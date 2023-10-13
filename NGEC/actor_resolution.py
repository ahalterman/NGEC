import os
import pandas as pd
import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import dateparser
import unidecode
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from collections import Counter
import logging
from textacy.preprocessing.remove import accents as remove_accents
from rich import print
from rich.progress import track
import time
import jsonlines
from scipy.spatial.distance import cdist
import pylcs
from sentence_transformers.util import cos_sim
import torch

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def setup_es():
    CLIENT = Elasticsearch()
    try:
        CLIENT.ping()
        #logger.info("Successfully connected to Elasticsearch.")
    except:
        ConnectionError("Could not locate Elasticsearch. Are you sure it's running?")
    conn = Search(using=CLIENT, index="wiki")
    return conn

def check_wiki(conn):
    """
    We changed the Wikipedia format on 2022-04-21, so make sure we're using the
    right one.
    """
    q = {"multi_match": {"query": "Massachusetts",
                                        "fields": ['title^2', 'alternative_names'],
                                        "type" : "phrase"}
                                        }
    res = conn.query(q)[0:1].execute()
    top = res['hits']['hits'][0].to_dict()['_source']
    if 'redirects' not in top.keys():
        raise ValueError("You seem to be using an outdated Wikipedia index that doesn't have a 'redirects' field. Please talk to Andy.")

def load_county_dict(base_path):
    """
    Construct a list of regular expressions to find countries by their name and nationality 
    (e.g. Germany, German).

    Update: sometimes we want countries from the categories, but those are much messier. For example:
    - countries recognized by Germany
    - Russian-speaking countries.

    To handle those, have a second set of patterns that start with "of" or "in". This is hacky!
    """
    file = os.path.join(base_path, "countries.csv")
    countries = pd.read_csv(file)
    #nat_dict = {}
    nat_list = []
    for n, i in countries.iterrows():
        nats = [i.strip() for i in i['Nationality'].split(",")]
        for nat in nats:
            patt = (re.compile(nat + "(?=[^a-z]|$)"), i['CCA3'])
            nat_list.append(patt)
        patt = (re.compile(i['Name']), i['CCA3'])
        nat_list.append(patt)
    nat_list_cat = []
    for prefix in ['of ', 'in ']: 
        for n, i in countries.iterrows():
            nats = [i.strip() for i in i['Nationality'].split(",")]
            for nat in nats:
                patt = (re.compile(prefix + nat), i['CCA3'])
                nat_list_cat.append(patt)
            patt = (re.compile(i['Name']), i['CCA3'])
            nat_list_cat.append(patt)
    return nat_list, nat_list_cat


def load_spacy_lg():
    nlp = spacy.load("en_core_web_lg", disable=["pos", "dep"])
    return nlp

def load_trf_model(model_dir='sentence-transformers/paraphrase-MiniLM-L6-v2'): ## Change to offline!!
    model = SentenceTransformer(model_dir)
    return model

def load_actor_sim_model(base_path, model_dir='actor_sim_model2'): 
    """
    This is the model that was trained on Wikipedia redirects to figure out if two names
    are equivalent. Used to identify Wikipedia articles if no exact matches are available.
    """
    combo_path = os.path.join(base_path, model_dir)
    model = SentenceTransformer(combo_path)
    return model


class ActorResolver:
    def __init__(self, 
                spacy_model=None,
                base_path="./assets",
                save_intermediate=False,
                wiki_sort_method="neural",
                gpu=False):
        if gpu:  
            self.device=0
        else:
            self.device = None
        self.conn = setup_es()
        if spacy_model:
            self.nlp = spacy_model
        else:
            self.nlp = load_spacy_lg()
        self.trf = load_trf_model()
        self.actor_sim = load_actor_sim_model(base_path)
        self.agents = self.clean_agents(base_path)
        self.trf_matrix = self.load_embeddings(base_path)
        self.nat_list, self.nat_list_cat = load_county_dict(base_path)
        self.base_path = base_path
        self.cache = {}
        self.save_intermediate=save_intermediate
        self.wiki_sort_method = wiki_sort_method


    def load_embeddings(self, base_path):
        """
        Load pre-computed embedding matrices from disk, or, if
        the precomputed matrices are out-of-date, re-compute and
        save them.
        """
        # Check if the agents file and bert matrix are mismatched.
        # If so, recompute bert matrix and save.
        hash_file = os.path.join(base_path, "PLOVER_agents.hash")
        try:
            with open(hash_file, "r") as f:
                existing_hash = f.read()
        except FileNotFoundError:
            existing_hash = ""
        agent_file = os.path.join(base_path, "PLOVER_agents.txt")
        with open(agent_file, "r", encoding="utf-8") as f:
            data = f.read() 
        hashed_agents = hash(data)
        if str(existing_hash) != str(hashed_agents):
            logger.info("Agents file and pre-computed matrix are mismatched. Recomputing...")
            patterns = [i['pattern'] for i in self.agents]
            trf_matrix = self.trf.encode(patterns, device=self.device)
            file_bert = os.path.join(base_path, "bert_matrix.pkl")
            with open(file_bert, "wb") as f:
                pickle.dump(trf_matrix, f)
            with open(hash_file, "w") as f:
                f.write(str(hashed_agents))

        # now read in the matrix
        logger.info("Reading in BERT matrix")
        file_bert = os.path.join(base_path, "bert_matrix.pkl")
        with open(file_bert, "rb") as f:
            trf_matrix = pickle.load(f)
        return trf_matrix


    def clean_agents(self, base_path):
        """
        Read, parse, and clean a PLOVER/CAMEO agents file.
        """
        file = os.path.join(base_path, "PLOVER_agents.txt")
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()

        data = re.sub(r"\{.+?\}", "", data)
        ags = data.split("\n")
        ags = [i for i in ags if i]
        ags = [i for i in ags if i[0] != "#"]
        logger.debug(f"Total agents: {len(ags)}")
        ags = [i for i in ags if i[0] != "!"]
        ags = [re.sub(r"#.+", "", i).strip() for i in ags]

        patterns = []
        for i in ags:
            try:
                code = re.findall(r"\[.+?\]", i)[0]
                code = re.sub(r"[\[\]]", "", code)
                code = re.sub(r"~", "", code).strip()
                # convert from CAMEO to PLOVER style
                patt = re.sub(r"(\[.+?\])", "", i)
                patt = re.sub(r"_", " ", patt).lower()
            except Exception as e:
                logger.info(f"Error loading {i}:", e)
            d = {"pattern": patt.strip(), "code_1": code[0:3], "code_2": code[3:]}
            patterns.append(d)

        cleaned = []
        for i in patterns:
            if 'code_1' not in i.keys():
                continue
            if re.search("!minist!", i['pattern']):
                for p in ["Minister", "Ministers", "Ministry", "Ministries"]:
                    new_p = {"code_1": i['code_1'], "code_2": i['code_2']}
                    new_p['pattern'] = re.sub(r"!minist!", p, i['pattern']).title()
                    cleaned.append(new_p)
            if re.search("!person!", i['pattern']):
                new_p = [re.sub(r"!person!", p, i['pattern']) for p in ["person", "man", "woman", "men", "women"]]
                new_p = [{"code_1": i['code_1'], "code_2": i['code_2'], 'pattern': p} for p in new_p]
                cleaned.extend(new_p)
            else:
                cleaned.append(i)
        return cleaned


    def trf_agent_match(self, 
                        non_ent_text, 
                        country="", 
                        method="cosine",
                        threshold=0.6):
        """
        Compare an input string to the agent file using a sentence transformer
        representation, returning the closest match

        Parameters
        ----------
        non_ent_text: str
          The input string to match
        country: str
          The name of the country (previously detected) to add to the resulting 
          entry.
        threshold: num
          Threshold below which matches won't be returned

        Returns
        ------
        match: dict
          {"country": the country name (passed in to function),
          "description": the pattern in the agents file that was the closest match,
          "code_1": PLOVER code 1 (e.g. GOV, MIL, etc),
          "code_2": any secondary PLOVER code}
        """
        if method not in ['cosine', 'dot']:
            raise ValueError("distance method must be one of ['cosine', 'dot']")
        if method == "dot" and threshold < 1:
            threshold = 45
            logger.info(f"Threshold is low high for dot. Setting to {threshold}")
        if method == "cosine" and threshold > 2:
            threshold = 0.1
            logger.info(f"Threshold is too high for cosine. Setting to {threshold}")
        if country is None:
            country = ""
        non_ent_text = self.clean_query(non_ent_text)
        #non_ent_text = non_ent_text.lower()

        if not non_ent_text:
            return None
        query_trf = self.trf.encode(non_ent_text, show_progress_bar=False)
        if method == "dot":
            sims = np.dot(self.trf_matrix, query_trf.T)
        if method == "cosine":
            sims = 1 - cdist(self.trf_matrix, np.expand_dims(query_trf.T, 0), metric="cosine")
        mmatch = self.agents[np.argmax(sims)].copy() # hopefully that fixes the weird wiki bug!!
        if np.max(sims) < threshold:
            logger.debug(f"Agents file comparsion. Closest result for {non_ent_text} is {mmatch['pattern']} with conf {np.max(sims)}")
            return None
        mmatch['country'] = country
        mmatch['description'] = mmatch['pattern']
        mmatch['query'] = non_ent_text
        mmatch['conf'] = np.max(sims)
        logger.debug(f"Match from trf_agent_match: {mmatch}")
        return mmatch

    def strip_ents(self, doc):
        """Strip out named entities from text"""
        skip_list = ['a', 'and', 'the', "'s", "'", "s"]
        non_ent_text = ''.join([i.text_with_ws for i in doc if i.ent_type_ == "" and i.text.lower() not in skip_list]).strip()
        return non_ent_text.strip()
    
    def get_noun_phrases(self, doc):
        skip_list = ['a', 'and', 'the']
        noun_phrases = [i for i in doc.noun_chunks if i[-1].ent_type_ == ""]
        short_text = ' '.join([j.text_with_ws.lower() for i in noun_phrases for j in i if j.text not in skip_list and j.ent_type_ not in ['CARDINAL', 'DATE', 'ORDINAL']]).strip()
        return short_text

    def get_noun_phrases_list(self, doc):
        skip_list = ['a', 'and', 'the']
        noun_phrases = [i for i in doc.noun_chunks if i[-1].ent_type_ == ""]
        return noun_phrases

    def short_text_to_agent(self, text, strip_ents=False, threshold=0.5):
        country, trimmed_text = self.search_nat(text)
        trimmed_text = self.clean_query(trimmed_text)
        if strip_ents:
            try:
                doc = self.nlp(text)
                trimmed_text = self.strip_ents(doc)
            except IndexError:
                # if NLPing fails, continue with trimmed_text as-is
                pass
            if trimmed_text == "s":
                return None
        code = self.trf_agent_match(trimmed_text, country=country, threshold=threshold)
        return code

#    def long_text_to_agent(self, text, method="cosine", threshold=0.7):
#        """
#        Keep only noun phrases and do BERT similarity lookup.
#        
#        Question: Better to look up each noun phrase separately, rather than
#        joining them all together and then looking it up?
#        """
#        country, trimmed_text = self.search_nat(text)
#        doc = self.nlp(trimmed_text)
#        short_text = self.get_noun_phrases(doc)
#        code = self.trf_agent_match(short_text, country=country, method=method, threshold=threshold)
#        return code       

    def search_nat(self, text, method="longest", categories=False):
        """
        Grep for a country name in plain text and return a canonical form
        
        TODO: Handle multiple country mentions
        """
        text = unidecode.unidecode(text)
        found = []
        #for k, v in self.nat_dict.items():
        if not categories:
            for k, v in self.nat_list:
                match = re.search(k, text)
                if match:
                    trimmed_text = re.sub(k, "", text).strip()
                    trimmed_text = self.clean_query(trimmed_text)
                    found.append((v, trimmed_text.strip(), match))
        if categories:
            for k, v in self.nat_list_cat:
                match = re.search(k, text)
                if match:
                    trimmed_text = re.sub(k, "", text).strip()
                    trimmed_text = self.clean_query(trimmed_text)
                    found.append((v, trimmed_text.strip(), match))
        if not found:
            return None, text
        elif method == "longest":
            # return the longest match to handle e.g. "Saudi", "Britain"
            found.sort(key=lambda x: len(x[1]))
            return found[0][0:2]
        elif method == "first":
            found.sort(key=lambda x: x[2].span()[0])
            return found[0][0:2]
        else:
            raise ValueError(f"search_nat sorting option must be one of ['longest', 'first']. You gave {method}")



    def parse_offices(self, infobox):
        """
        TODO: in some rare cases, offices are listed under 'title' instead of 'office'. E.g.:
        'box_type': 'officeholder',
 'infobox': {'name': 'Rosario Marin',
  'image': 'rosario marin.jpg',
  'caption': 'Official Portrait',
  'order': '41st',
  'title': 'Treasurer of the United States',
  'term_start': 'August 16, 2001',
  'term_end': 'June 30, 2003',
  'predecessor': 'Mary Ellen Withrow',
  'successor': 'Anna Escobedo Cabral',
  'president': 'George W. Bush',
  'order2': '',
  'title2': 'Mayor of Huntington Park, California',
  'term_start2': '1999',
  'term_end2': '2000',
  'president2': '',
  'predecessor2': 'Tom Jackson',
  'successor2': 'Jessica R. Maes',
  'order3': '',
        """
        offices = []
        office_keys = [i for i in infobox.keys() if re.search("office", i)]
        logger.debug(f"Office keys: {office_keys}")
        for i in office_keys:
            try:
                n = int(re.findall(r"\d+", i)[0])
            except:
                n = ""  # this is the most current one
            try:
                term_end = dateparser.parse(infobox[f"term_end{n}"])
            except KeyError:
                try:
                    term_end = dateparser.parse(infobox[f"termend{n}"])  # sometimes no underscore
                except KeyError:
                    term_end = None
            try:
                term_start = dateparser.parse(infobox[f"term_start{n}"])
            except KeyError:
                try:
                    term_start = dateparser.parse(infobox[f"termstart{n}"]) # sometimes no underscore
                except KeyError:
                    term_start = None
            try:
                d = {"office": infobox[f"office{n}"],
                    "office_num": n,
                    "term_start": term_start,
                    "term_end": term_end}
                offices.append(d)
            except KeyError:
                continue
        return offices



    def get_current_office(self, offices, query_date):
        """
        Parameters
        ----------
        offices: list of dir
          Parsed out office list from Wiki
        query_date: str or datetime
          Date to use to get "current" office
        """
        if type(query_date) is str:
            query_date = dateparser.parse(query_date)
        active_offices = []
        detected_countries = []
        for i in offices:
            if not i['term_start']:
                continue
            # REMOVE!!!!!
            country, trimmed_text = self.search_nat(i['office'])
            if country:
                detected_countries.append(country)

            try:
                if i['term_start'] < query_date and not i['term_end']:
                    active_offices.append(i)
                elif i['term_start'] < query_date and i['term_end'] > query_date:
                    active_offices.append(i)
            except Exception:
                logger.info("Term start or end error in current office")
        # handle the ELI/CVL/former official thing outside the function
        return active_offices, detected_countries

    def clean_query(self, qt):
        qt = str(qt).strip()
        if qt in ['The', 'the', 'a', 'an', '']:
            return ""
        qt = re.sub(' +', ' ', qt) # get rid of multiple spaces 
        qt = re.sub('\n+', ' ', qt) # newline to space
        # remove starting the, An, etc.
        qt = re.sub("^the ", "", qt.strip()) 
        qt = re.sub("^[Aa]n ", "", qt).strip()
        qt = re.sub("^[Aa] ", "", qt).strip()
        qt = re.sub(" of$", "", qt).strip()
        qt = re.sub("^'s", "", qt).strip()
        # remove ordinals (First a two-digit ordinal, then a 1 digit)
        qt = re.sub(r"(?<=\d\d)(st|nd|rd|th)\b", '', qt).strip()
        qt = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', qt).strip()
        qt = re.sub("^\d+? ", "", qt).strip()
        if len(qt) < 2:
            return ""
        return qt


    def search_wiki(self, 
                    query_term, 
                    limit_search_by_term="", 
                    fuzziness="AUTO", 
                    max_results=200,
                    fields=['title^50', 'redirects^50', 'alternative_names'],
                    score_type = "best_fields"):
        """
        Search Wikipedia for a given query term. Returns a list of dicts.

        Parameters
        ----------
        query_term: str
            Query term to search for
        limit_term: str
            Also search for this term to limit results. For example, country names.

        """
        query_term = self.clean_query(query_term)
        #if text:
        #    query_term = query_term + " " + text
        logger.debug(f"Using query term: '{query_term}'")
        if not limit_search_by_term:
            q = {"multi_match": {"query": query_term,
                             "fields": fields,
                             "type": score_type,
                             "fuzziness" : fuzziness,
                             "operator": "and"
                        }}
        else:
            limit_fields = ["title^100", "redirects^100", "alternative_names",
                            "intro_para", "categories", "infobox"]
            q = {"bool": {"must": [{"multi_match": {"query": query_term,
                             "fields": fields,
                             "type": score_type,
                             "fuzziness" : fuzziness,
                             "operator": "and"
                        }},
                          {"multi_match": {"query": limit_search_by_term,
                             "fields": limit_fields,
                             "type": "most_fields"}}]}}

        res = self.conn.query(q)[0:max_results].execute()
        results = [i.to_dict()['_source'] for i in res['hits']['hits']] 
        logger.debug(f"Number of hits for fuzzy ES/Wiki query: {len(results)}")
        return results


    def text_ranker_features(self, matches, fields):
        """
        Given a list of fields and a list of Wikipedia matches, pull the text
        out from the fields and combine into one big string for comparing.
        """
        wiki_text = []
        for m in matches:
            txt = ""
            for f in fields:
                try:
                    t = m[f]
                    if type(t) is str:
                        sents = t.split("\n")
                        if sents:
                            txt = " " + txt + " " + sents[0]
                    elif type(t) is list:
                        tj = ', '.join(t)
                        txt = txt + tj
                except KeyError:
                    logger.debug(f"Missing key {f} for {m['title']}")
                    continue
            wiki_text.append(txt.strip())
        return wiki_text

    def compute_lcs(self, query, matches, fields):
        max_strings = []
        mean_strings = []
        min_edit = [] # not implemented. Add edit distance?
        for m in matches:
            match_max = []
            match_mean = []
            match_edits = []
            for f in fields:
                lcs_list = [pylcs.lcs_sequence_length(query, i) for i in m[f] if i]
                if lcs_list:
                    match_max.append(np.max(lcs_list))
                    match_mean.append(np.mean(lcs_list))
                else:
                    match_max.append(0)
                    match_mean.append(0)
            max_strings.append(np.max(match_max) / len(query))
            mean_strings.append(np.mean(match_mean) / len(query))
        return max_strings, mean_strings

    def _trim_results(self, results):
        """
        Helper function to remove bad articles
        """
        good_res = [i for i in results if not re.search("(stub|User|Wikipedia\:)", i['title']) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("disambiguation", i['title']) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("Category\:", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("is the name of", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("may refer to", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("can refer to", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("most commonly refers to", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("usually refers to", i['intro_para'][0:80]) if i['intro_para']]
        good_res = [i for i in good_res if not re.search("is a surname", i['intro_para'][0:50]) if i['intro_para']]
        good_res = [i for i in good_res if len(i['intro_para']) > 50 if i['intro_para']]
        good_res = [i for i in good_res if i['intro_para'].strip()]
        return good_res


    def pick_best_wiki(self, 
                    query_term, 
                    results, 
                    text="", 
                    country="",
                    wiki_sort_method="neural",
                    rank_fields=['title', 'categories', 'alternative_names', 'redirects']):
        query_term = self.clean_query(query_term)
        logger.debug(f"Using query term '{query_term}'")
        if country:
            query_country = f"{query_term} ({country})"
            logger.debug(f"Secondary country query term {query_country}")
        else:
            query_country = query_term
        if not results:
            logger.debug("No wikipedia results. Returning None")
            return []
        #if len(results) == 1:
        #    best = results[0]
        #    best['wiki_reason'] = f"Only one hit."
        #    return best
        good_res = self._trim_results(results)
        if not good_res:
            return None
        logger.debug(f"Pared down to {len(good_res)} good results")

        exact_matches = []
        for i in good_res:
            if query_term == i['title'] or query_country == i['title']:
                exact_matches.append(i)
            elif query_term == remove_accents(i['title']) or query_country == remove_accents(i['title']):
                exact_matches.append(i)
            elif query_term.title() == i['title'] or query_country.title() == i['title']:
                exact_matches.append(i)


        logger.debug(f"Number of title matches: {len(exact_matches)}")
        if len(exact_matches) == 1:
            best = exact_matches[0]
            best['wiki_reason'] = "Only one exact title match"
            return best
        if len(exact_matches) == 2:
            if len(exact_matches[0]['intro_para']) > len(exact_matches[1]['intro_para']):
                best = exact_matches[0]
            else:
                best = exact_matches[1]
            best['wiki_reason'] = "Only two exact title matches, returning page with long intro"
            return best


        if exact_matches: 
            if wiki_sort_method == "alt_names":
                exact_matches.sort(key=lambda x: -len(x['alternative_names']))
                if exact_matches:
                    logger.debug(f"Wiki: returning exact title match (out of {len(exact_matches)} with longest alternative_names field")
                    best = exact_matches[0]
                    best['wiki_reason'] = f"Multiple title exact matches: Returning longest alt names"
                    return best
            elif wiki_sort_method in ["neural", "lcs"]: # TODO: add lcs to this part too
                try:
                    wiki_info = self.text_ranker_features(exact_matches, rank_fields)
                    category_trf = self.trf.encode(wiki_info, show_progress_bar=False, device=self.device)
                    query_trf = self.trf.encode(query_term, show_progress_bar=False)
                    sims = 1 - cdist(category_trf, np.expand_dims(query_trf.T, 0), metric="cosine")
                    exact_matches_sorted = [x for _, x in sorted(zip(sims, exact_matches), reverse=True)]
                    best = exact_matches_sorted[0]
                    best['wiki_reason'] = f"Multiple title exact matches: picking by neural similarity. Similarity = {sims[0]}"
                    return best
                except Exception as e:
                    logger.debug(f"{e}")
                    logger.debug(f"Exception on query term {query_term}")

        redirect_match = []
        for i in good_res:
            if query_term in i['redirects'] or query_country in i['redirects']:
                redirect_match.append(i)
            elif query_term.title() in i['redirects'] or query_country.title() in i['redirects']:
                redirect_match.append(i)
        logger.debug(f"Number of redirect matches: {len(redirect_match)}")

        if len(redirect_match) == 1:
            best = redirect_match[0]
            best['wiki_reason'] = "Single redirect exact match"
            return best

        if len(redirect_match) == 2:
            for i in redirect_match:
                if len(i['intro_para']) > 40:
                    best = i
                    best['wiki_reason'] = "Only two exact redirect matches, returning page with long intro"
                    return best
        
        if redirect_match:
            logger.debug(f"More than one redirect match. Using {wiki_sort_method} to try to pick one...")
            if wiki_sort_method == "alt_names":
                redirect_match.sort(key=lambda x: -len(x['alternative_names']))
                best = redirect_match[0]
                best['wiki_reason'] = "Redirect exact match; picking by longest alt names."
                logger.debug("Redirect exact match; picking by longest alt names.")
                return best
            elif wiki_sort_method in ["neural", "lcs"]:
                try:
                    query_context = query_term + text
                    wiki_info = self.text_ranker_features(redirect_match, rank_fields)
                    category_trf = self.trf.encode(wiki_info, show_progress_bar=False, device=self.device)
                    query_trf = self.trf.encode(query_context, show_progress_bar=False)
                    sims = 1 - cdist(category_trf, np.expand_dims(query_trf.T, 0), metric="cosine")
                    redirect_match = [x for _, x in sorted(zip(sims, redirect_match), reverse=True)]
                    best = redirect_match[0]
                    logger.debug("Redirect exact match; picking by neural similarity.")
                    best['wiki_reason'] = "Redirect exact match; picking by neural similarity."
                    return best
                except Exception as e:
                    logger.debug(f"{e}")
                    logger.debug(f"Exception on query term {query_context}")



        ## If no title or redirect matches, including fuzzy matches, do the neural
        ## similarity thing on titles
        logger.debug("Falling back to title neural similarity")
        titles = [i['title'] for i in good_res[0:50]] # just look at first 10? HACK
        if not titles:
            logger.debug("No titles. Returning None")
            return None
        enc_titles = self.actor_sim.encode(titles, show_progress_bar=False)
        enc_query = self.actor_sim.encode(query_term, show_progress_bar=False)
        actor_sims = cos_sim(enc_query, enc_titles)
        if torch.max(actor_sims) > 0.9:
            match = torch.argmax(actor_sims)
            best = good_res[match]
            best['wiki_reason'] = "High neural similarity between query and Wiki title"
            logger.debug(f"High neural similarity between query and Wiki title: {torch.max(actor_sims)}")
            return best
        else:
            logger.debug(f"Not a close enough match on neural title sim. Closest was {torch.max(actor_sims)} on {good_res[torch.argmax(actor_sims)]['title']}")
        

        alt_match = [i for i in good_res if query_term in i['alternative_names'] or query_country in i['alternative_names']]
        logger.debug(f"Number of alternative name matches: {len(alt_match)}")
        if len(alt_match) == 1: 
            best = alt_match[0]
            enc_titles = self.actor_sim.encode([best['title']], show_progress_bar=False)
            enc_query = self.actor_sim.encode(query_term, show_progress_bar=False)
            actor_sims = cos_sim(enc_query, enc_titles)
            if torch.max(actor_sims) > 0.8:
                best['wiki_reason'] = "Only one alternative names match + high title sim"
                return best
        elif alt_match:
            logger.debug("Falling back to title neural similarity on alt name matches")
            titles = [i['title'] for i in alt_match[0:10]] # just look at first 10? HACK
            if not titles:
                logger.debug("No titles. Returning None")
                return None
            enc_titles = self.actor_sim.encode(titles, show_progress_bar=False)
            enc_query = self.actor_sim.encode(query_term, show_progress_bar=False)
            actor_sims = cos_sim(enc_query, enc_titles)
            if torch.max(actor_sims) > 0.9:
                match = torch.argmax(actor_sims)
                best = alt_match[match]
                best['wiki_reason'] = "High neural similarity between query and Wiki title on alt name match"
                logger.debug(f"High neural similarity between query and Wiki title on alt name match: {torch.max(actor_sims)}")
                return best
            else:
                logger.debug(f"Not a close enough match on neural title sim. Closest was {torch.max(actor_sims)} on {alt_match[torch.argmax(actor_sims)]['title']}")
        
        return None

        
    def query_wiki(self, query_term, context = "", limit_term="", country="", max_results=200):
        print(f"limit_term in query_wiki: {limit_term}")

        results = self.search_wiki(query_term, limit_term, fuzziness=0, max_results=200)
        best = self.pick_best_wiki(query_term, results, limit_term, country=country)
        if best:
            return best
        results = self.search_wiki(query_term, limit_term, fuzziness=1, max_results=max_results)
        best = self.pick_best_wiki(query_term, results, limit_term, country=country)
        if best:
            return best


    def wiki_to_code(self, wiki, query_date="today", country=""):
        """
        Resolve a Wikipedia page to a PLOVER actor code, either using
        the sidebar info or the first sentence of the intro para.

        Parameters
        ---------
        page: dict
          Wiki article from ES
        query_date: str
          Optional, limit to a specific time
        """
        skip_types = [] #["War Faction"]

        if not wiki:
            return []

        #para_code = []
        box_codes = []
        countries = []
        sd_code = []
        b_code = None
        type_code = None
        country = None
        office_countries = []
        cat_countries = []

        intro_text = re.sub(r"\(.*?\)", "", wiki['intro_para']).strip()
        try:
            first_sent = intro_text.split("\n")[0] 
        except IndexError:
            first_sent_country = None
        first_sent_country, trimmed_text = self.search_nat(first_sent, method="first")
        country = first_sent_country

        if 'short_desc' in wiki.keys():
            country, trimmed_text = self.search_nat(wiki['short_desc'])
            countries.append(country)
            sd_code = self.trf_agent_match(trimmed_text, country=country)
            if sd_code:
                sd_code['source'] = "Wiki short description"
                sd_code['actor_wiki_job'] = wiki['short_desc']
                sd_code['wiki'] = wiki['title']
                sd_code['country'] = country
            sd_code = [sd_code]
        
        if 'infobox' in wiki.keys():
            infobox = wiki['infobox']
            if 'country' in infobox.keys():
                country = self.search_nat(infobox['country'])[0]
                countries.append(country)
            if 'box_type' in wiki.keys():
                b_code = self.trf_agent_match(wiki['box_type'])
                if b_code:
                    b_code['country'] = country
                    b_code['wiki'] = wiki['title']
                    b_code['source'] = "Infobox Title"
                    b_code['actor_wiki_job'] = wiki['box_type']
            if 'type' in infobox.keys():
                type_code = self.trf_agent_match(infobox['type'])
                if type_code:
                    type_code['country'] = country
                    type_code['wiki'] = wiki['title']
                    type_code['source'] = "Infobox Type"
                    type_code['actor_wiki_job'] = infobox['type']
            offices = self.parse_offices(infobox)
            logger.debug(f"All offices: {offices}")
            current_offices, office_countries = self.get_current_office(offices, query_date)
            logger.debug(f"Current offices: {current_offices}")

            # the ELI block
            if offices and not current_offices:
                logger.debug("Running ELI block")
                old_codes_raw = [self.trf_agent_match(self.clean_query(i['office'])) for i in offices]
                old_codes = []
                for i in old_codes_raw:
                    if not i:
                        continue
                    if 'code_1' not in i.keys():
                        logger.info(f"Weird pattern? {i}")
                        continue
                    old_codes.append(i['code_1'])
                old_countries = [self.search_nat(i['office'])[0] for i in offices]
                box_country = list(set([i for i in old_countries if i]))
                if "GOV" in old_codes:
                    code_1 = "ELI"
                    if len(box_country) == 0:
                        box_codes = [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 'country': country, 'description': "previously held a GOV role, so coded as ELI.", "source": "Infobox", "wiki": wiki['title']}] 
                    elif len(box_country) == 1 and (box_country[0] == country or country == ""):
                        box_codes = [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 'country': box_country[0], 'description': 'previously held a GOV role, so coded as ELI', "source": "Infobox", "wiki": wiki['title']}]
                    elif len(box_country) > 1 or (box_country[0] == country and country != ""):
                        box_codes = [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 'country': country, 'description': f"previously held a GOV role, so coded as ELI. Extracted countries don't match: {box_country}", "source": "Infobox", "wiki": wiki['title']}]
                    else:
                        box_codes = [{'pattern': 'NA', 'code_1': code_1, 'code_2': '', 'country': country, 'description': "previously held a GOV role, so coded as ELI.", "source": "Infobox", "wiki": wiki['title']}]
            else:  # they have a current office
                box_codes = []
                for co in current_offices:
                    office = self.clean_query(co['office'])
                    b = self.short_text_to_agent(office)
                    if b:
                        b['actor_wiki_job'] = office
                        b['source'] = "Infobox"
                        if not co['office_num']:
                            # current offices are sometimes blank
                            b['office_num'] = 0
                        else:
                            b['office_num'] = co['office_num']
                        b["wiki"] = wiki['title']
                        b['source'] = "Infobox"
                        logger.debug(f"{b}")
                        box_codes.append(b)
                        break ## Only get the first one!

        # TODO: get country from infobox? A few places to look are:
        #  - nationality
        #  - headquarters
        #  - jurisdiction
        #  - pushpin_map (somewhat noisy)
        #  - categories (noisy)

        
        #################  DISABLED FOR NOW ######################
        # This part is slow and *decreases* the accuracy on the eval set.
        # In theory, there are people without info boxes that we'd only get
        # from the first para, but there seem to be more where the first
        # para just messes things up.

        ## There was also a weird issue where the paragraph codes would
        # overwrite the info box codes. I have no idea how that's happening.
        # One option (implemented above) is to only run it if there are no box codes. 
#        if not box_codes:
#            try:
#                first_sent = intro_text.split("\n")[0] ## Speed up here!!
#            except IndexError:
#                first_sent = intro_text
#            first_sent_country, trimmed_text = self.search_nat(first_sent)
#
#            if trimmed_text: 
#                noun_chunks = self.get_noun_phrases_list(self.nlp(trimmed_text))
#                #if not first_sent_country:
#                #    first_sent_country, _  = self.search_nat(wiki['intro_para'])
#
#                # Seems to be messing up/overwriting the info box entries...
#                # NOTE: change so it only runs if there are no box codes
#                para_code = []
#                for i in noun_chunks:
#                    p_code = self.trf_agent_match(i.text, country=first_sent_country) 
#                    if p_code:
#                        p_code["source"] = "Intro paragraph--wiki"
#                        p_code["wiki"] = wiki['title']
#                        para_code.append(p_code)
#            else:
#                para_code = []
        ###################################
        if 'categories' in wiki.keys():
            cats = wiki['categories']
            for cat in cats:
                c, _ = self.search_nat(cat, categories=True)
                if c:
                    cat_countries.append(c)

        all_codes = box_codes + sd_code + [b_code] + [type_code]  # + para_code
        all_codes = [i for i in all_codes if i]
        logger.debug(f"All codes: {all_codes}")

        logger.debug(f"First sent country: {first_sent_country}")
        all_countries = [i['country'] for i in all_codes if i['country']]
        all_countries.extend(countries)
        all_countries.extend(office_countries)
        logger.debug(f"Category countries: {cat_countries}")
        all_countries.extend(cat_countries)
        if not all_countries:
            all_countries = [first_sent_country]

        top_country = Counter([i for i in all_countries if i])
        if top_country:
            top_country = top_country.most_common(1)[0][0]
        all_countries = list(set([i for i in all_countries if i]))
        logger.debug(f"All countries: {all_countries}")
        if len(all_countries) == 1:
            for i in all_codes:
                #if i['country'] == "":
                i['country'] = all_countries[0]
        elif top_country:
            for i in all_codes:
                if i['country'] == "":
                    i['country'] = top_country

        if not all_codes and len(all_countries) == 1:
            all_codes = [{'pattern': '', 
                        'code_1': '', 
                        'code_2': '', 
                        'country': all_countries[0], 
                        'description': "No code identified, but country found", 
                        "source": "Wiki", 
                        "wiki": wiki['title']}]
        elif not all_codes and top_country:
            all_codes = [{'pattern': '', 
                        'code_1': '', 
                        'code_2': '', 
                        'country': top_country, 
                        'description': "No code identified, but country found", 
                        "source": "Wiki", 
                        "wiki": wiki['title']}]

        return all_codes


    def pick_best_code(self, all_codes, country):
        logger.debug(f"Running pick_best_code with input country {country}")
        if len(all_codes) == 1:
            best = all_codes[0]
            best['best_reason'] = "only one code"
            if not best['country'] and country:
                best['country'] = country
            return best

        all_countries = [i['country'] for i in all_codes if i['country']]
        if country:
            all_countries.append(country)
        logger.debug(f"pick best code all_countries: {all_countries}")
        unique_code_1s = list(set([i['code_1'] for i in all_codes if i['code_1']]))

        wiki = [i['wiki'] for i in all_codes if 'wiki' in i.keys()]
        wiki = [i for i in wiki if i]
        if wiki:
            wiki = wiki[0]
        else:
            wiki = ""

        ####  Get country  #####
        if len(set(all_countries)) == 1:
            best_country = all_countries[0]
            if not all_codes:
                best = {"country": country,
                        "code_1": "",
                        "code_2": "",
                        "source": "country only",
                        "wiki": wiki,
                        "query": ''}
                return best
        elif not all_countries:
            best_country = ""
        elif country:
            best_country = country
        else:
            best_country = Counter(all_countries).most_common(1)[0][0]
        logger.debug(f"Identified as the best country: {best_country}")

        ####  Get role  #####
        code_sources = [i for i in all_codes if 'source' in i.keys()]
        box_type = [i for i in code_sources if i['source'] == "Infobox Type"]
        if box_type:
            if box_type[0]['query'] in ['settlement']:
                # if it's a city, then there's no agent/sector code, it's just the country
                best = box_type[0]
                best['code_1'] = ""
                if not best['country']:
                    best['country'] = best_country
                if 'wiki' not in best.keys():
                    best['wiki'] = wiki
                best['best_reason'] = "It's a city/settlement, so no code1 applies"
                return best 

        info_box = [i for i in code_sources if i['source'] == "Infobox"]
        if len(info_box) == 1:
            # sort by office_num, then take first one
            best = info_box[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Only one entry in the info box, going with that one"
            if 'wiki' not in best.keys():
                best['wiki'] = wiki
            return best
        elif len(info_box) > 1:
            info_box.sort(key=lambda x: x['office_num'])
            best = info_box[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Picking highest priority Wiki info box title"
            if 'wiki' not in best.keys():
                best['wiki'] = wiki
            return best

        if len(unique_code_1s) == 1:
            logger.debug("Only one unique code_1, so returning first one.")
            code1s = [i for i in all_codes if i['code_1']]
            wiki_codes = [i for i in code1s if i['wiki'] != '']
            logger.debug(f"Wiki codes: {wiki_codes}")
            if wiki_codes:
                best = wiki_codes[0]
                best['best_reason'] = "only one unique code1, returning wiki code."
                best['country'] = best_country
                if not best['country']:
                    best['country'] = best_country
                if 'wiki' not in best.keys():
                    best['wiki'] = wiki
            else:
                try:
                    code1s.sort(key=lambda x: -x['conf'])
                    best = code1s[0]
                    best['best_reason'] = "only one unique code1: returning highest conf"
                    if not best['country']:
                        best['country'] = best_country
                    if 'wiki' not in best.keys():
                        best['wiki'] = wiki
                    return best
                except KeyError:
                    logger.debug("Key Error on 'conf', proceeding to next code block")

        short_desc = [i for i in code_sources if i['source'] == "Wiki short description"]
        if short_desc:
            best = short_desc[0]
            if best['code_1'] == "IGO":
                best['country'] = "IGO"
            else:
                best['country'] = best_country
            best['best_reason'] = "Picking Wiki short description"
            return best

        non_wiki = []
        for i in all_codes:
            if i['source'] == "BERT matching on non-entity text":
                if i['country'] and i['code_1']:
                    non_wiki.append(i)

        if len(non_wiki) == 1:
            best = non_wiki[0]
            best['best_reason'] = "Using pre-wiki lookup"
            best['country'] = best_country
            if 'wiki' not in best.keys():
                best['wiki'] = wiki
            return best
        if len(set([i['country'] for i in non_wiki]))==1 and len(set([i['code_1'] for i in non_wiki]))==1:
            best = non_wiki[0]
            best['country'] = best_country
            best['best_reason'] = "All pre-wiki lookups are the same"
            if 'wiki' not in best.keys():
                best['wiki'] = wiki
            return best


        priority_dict = {"IGO": 200,
                        "ISM": 195,
                        "IMG": 192,
                        "PRE": 190,
                        "REB": 130,
                        "SPY": 110,
                        "JUD": 105,
                        "OPP": 102,
                        "GOV": 100,
                        "LEG": 90,
                        "MIL": 80,
                        "COP": 75,
                        "PRM": 72,
                        "ELI": 70,
                        "PTY": 65,
                        "BUS": 60,
                        "UAF": 50,
                        "CRM": 48,
                        "LAB": 47,
                        "MED": 45,
                        "NGO": 43,
                        "SOC": 42,
                        "EDU": 41,
                        "JRN": 40,
                        "ENV": 39,
                        "HRI": 38,
                        "UNK": 37,
                        "REF": 35,
                        "AGR": 30,
                        "RAD": 20,
                        "CVL": 10,
                        "JEW": 5,
                        "MUS": 5,
                        "BUD": 5,
                        "CHR": 5,
                        "HIN": 5,
                        "REL": 1,
                        "": 0,
                        "JNK": 51
                        }    
        logger.debug("Using code priority sorting")
        all_codes.sort(key=lambda x: -priority_dict[x['code_1']])
        logger.debug(all_codes)
        if all_codes:
            # get all the correct codes
            correct_codes = [i for i in all_codes if i['code_1'] == all_codes[0]['code_1']]
            wiki_codes = [i for i in correct_codes if i['wiki'] != '']
            logger.debug(f"Wiki codes: {wiki_codes}")
            if wiki_codes:
                best = wiki_codes[0]
                best['best_reason'] = "Ranked by code1 priority, returning wiki code."
            else:
                best = correct_codes[0]
                best['best_reason'] = "Ranked by code1 priority, returning first."
            best['country'] = best_country
            if 'wiki' not in best.keys():
                best['wiki'] = wiki_codes[0]
            return best 
        
        # not used anymore??
        trf_full_sent = [i for i in all_codes if i['source'] == 'BERT matching full text']
        if trf_full_sent:
            if trf_full_sent[0]['conf'] > 0.75: ## pretty arbitrary...
                best = trf_full_sent
                best['best_reason'] = "multiple codes, picking first high-confidence trf_full_sent"
                if 'wiki' not in best.keys():
                    best['wiki'] = wiki
                return best


    def clean_best(self, best):
        if not best:
            return None
        if best['code_1'] == 'JNK':
            return best
        if best['code_1'] in ["IGO", "MNC", "NGO", "ISM", "EUR", "UNO"]:
            best['country'] = best['code_1']
            best['code_1'] = ""
        if best['country'] is None or type(best['country'])is not str:
            best['country'] = ""
        if len(best['country']) == 6:
            best['code_1'] = best['country'][3:6]
            best['country'] = best['country'][0:3]
        return best
      

    def agent_to_code(self, 
                     text, 
                     context="", 
                     query_date="today", 
                     known_country="", 
                     search_limit_term=""):
        wiki_codes = [] 
        code_full_text = None
        code_non_ent = None
        #wiki_ent_codes = []

        cache_key = text + "_" + str(query_date)
        if cache_key in self.cache.keys():
            logger.debug("Returning from cache")
            return self.cache[cache_key]

        country, trimmed_text = self.search_nat(text)
        #trimmed_text = self.clean_query(trimmed_text)
        logger.debug(f"Identified country text: {country}")

        try:
            doc = self.nlp(trimmed_text)
            non_ent_text = self.strip_ents(doc)
            ents = [i for i in doc.ents if i.label_ in ['EVENT', 'FAC', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON']]
            token_level_ents = [i.ent_type_ for i in doc]
            ent_text = ''.join([i.text_with_ws for i in doc if i.ent_type_ != ""])
            logger.debug(f"Found the following named entities: {ents}")
        except IndexError:
            # usually caused by a mismatch between token and token embedding
            logger.info(f"Token alignment error on {trimmed_text}")
            non_ent_text = trimmed_text
            token_level_ents = ['']
            ent_text = ""
            ents = []

        if country and not trimmed_text:
            logger.debug("Country only, returning as-is")
            code_full_text = {"country": country,
                              "code_1": "",
                              "code_2": "",
                              "source": "country only",
                              "wiki": "",
                              'actor_wiki_job': "",
                              "query": text}
            self.cache[cache_key] = code_full_text
            return self.clean_best(code_full_text)
        
        # If a whole phrase is just a named entity, skip the lookup step completely???
        # I.e., only run the block below if there's at least one non-entity token.
        if trimmed_text: # and "" in token_level_ents:
            logger.debug(f"Running trf_agent_match on trimmed text: {trimmed_text}")
            code_full_text = self.trf_agent_match(trimmed_text, country=country, threshold=0.6)
            if code_full_text:
                logger.debug(f"Identified code using trf_agent_match on {trimmed_text}")
                code_full_text['source'] = "BERT matching full text"
                code_full_text['wiki'] = ""
                code_full_text['actor_wiki_job'] = "" 
                logger.debug(f"code_full_text: {code_full_text}")
                if code_full_text['conf'] > 0.6 and not ents:
                    logger.debug("High confidence on text-only and no named entities found. Returning w/o Wikipedia")
                    code_full_text = self.clean_best(code_full_text)
                    self.cache[cache_key] = code_full_text
                    return code_full_text
                elif code_full_text['conf'] > 0.90 and trimmed_text == trimmed_text.lower():
                    # Skip wikipedia if there's very high confidence and
                    # HACK: dumb NER finds no entities
                    logger.debug("High confidence on text-only and no ents. Returning w/o Wikipedia")
                    code_full_text = self.clean_best(code_full_text)
                    self.cache[cache_key] = code_full_text
                    return code_full_text
                elif code_full_text['conf'] > 0.95 == trimmed_text.lower():
                    code_full_text = self.clean_best(code_full_text)
                    logger.debug("Very high confidence on text-only. Returning w/o Wikipedia")
                    self.cache[cache_key] = code_full_text
                    return code_full_text
            else:
                logger.debug(f"No agent found for {trimmed_text}")

        # TODO: keep this at all?
        #if non_ent_text:
        #    if non_ent_text != trimmed_text:
        #        logger.debug(f"Running trf_agent_match on ent-stripped text: {non_ent_text}")
        #        code_non_ent = self.trf_agent_match(non_ent_text, country, threshold=0.7)
        #        if code_non_ent:
        #            logger.debug(f"Identified code in non_ent_text using trf_agent_match")
        #            code_non_ent['source'] = "BERT matching on non-entity text"
        #            #pattern = "(" + code_non_ent['query'] + "|" + code_non_ent['query'].title()
        #            #trimmed_text = 


        logger.debug(f"Querying Wikipedia with trimmed text: {trimmed_text}")
        wiki_codes = []
        wiki = self.query_wiki(query_term=trimmed_text, country=known_country, limit_term=search_limit_term)
        if wiki:
            logger.debug(f"Identified a Wiki page: {wiki['title']}")
            wiki_codes = self.wiki_to_code(wiki, query_date)
        else:
            if ent_text:
                logger.debug(f"No wiki results. Trying again with just proper nouns: {ent_text}")
                wiki = self.query_wiki(query_term=ent_text, country=known_country, limit_term=search_limit_term) 
                wiki_codes = self.wiki_to_code(wiki, query_date)
            

        # Doing this increased time by 50% and only decreased errors from 94 to 93.
       # ents = [i for i in doc.ents if i.label_ in ['EVENT', 'FAC', 'GPE', 'LOC', 'NORP', 'ORG', 'PERSON']]
       # if ents:
       #     ent_list = [j for i in ents for j in i]
       #     ent_text = ''.join([j.text_with_ws for j in ent_list]).strip()
       #     if ent_text:
       #         logger.debug(f"Ents found. Processing ent_text: {ent_text}")
       #         wiki_ents = self.query_wiki(ent_text)
       #     if wiki_ents:
       #         logger.debug(f"Wiki page: {wiki_ents['title']}")
       #         wiki_ent_codes = self.wiki_to_code(wiki_ents, query_date)
       #     else:
       #         wiki_ent_codes = []
        
        all_codes = wiki_codes + [code_full_text] + [code_non_ent]
        all_codes = [i for i in all_codes if i]
        
        logger.debug("--- ALL CODES ----")
        logger.debug(all_codes)
        unique_code1s = list(set([i['code_1'] for i in all_codes if i]))
        unique_code1s = [i for i in unique_code1s if i not in ["IGO"]]
        unique_code2s = list(set([i['code_2'] for i in all_codes if i]))
        #print([(i['source'], i['code_1'], i['country']) for i in all_codes if i])
        # try picking the one best...
        #print("\n\nRETURNING BEST:")
        best = self.pick_best_code(all_codes, country)
        best = self.clean_best(best)
        if best:
            best['all_code1s'] = unique_code1s
            best['all_code2s'] = unique_code2s
        self.cache[cache_key] = best
        return best

    def process(self, event_list):
        """

        Returns
        -------
        event_list
          For ACTOR and RECIP, adds the following to the 'attributes' for each:
            - 'wiki'
            - 'country'
            - 'code_1'
            - 'code_2'
            - 'actor_role_query'
            - 'actor_resolved_pattern'
            - 'actor_pattern_conf'
            - 'wiki_actor_job'
            - 'actor_resolution_reason'
        
        Example
        ------
        event = {'id': '20190801-2227-8b13212ac6f6_SANCTION', 
                'date': '2019-08-01', 
                'event_type': 'SANCTION', 
                'event_mode': [], 
                'event_text': 'The Liberal Party, the largest opposition in Paraguay, announced in the evening of Wednesday the decision to submit an application of impeachment against the president of the country, Mario Abdo Benítez, and vice-president Hugo Velázquez, by polemical agreement with Brazil on the purchase of energy produced in Itaipu. According to the president of the Liberal Party, Efraín Alegre, the opposition also come tomorrow with penal action against all those involved in the negotiations of the agreement with Brazil, signed on confidentiality in May and criticized for being detrimental to the interests of the country. The Liberal Party has the support of the front Guasú, Senator and former President Fernando Lugo, he himself target of an impeachment, decided in less than 24 hours, in June 2012. According to legend, the reasons for the opening of the proceedings against Abdo Benítez are bad performance of functions, betrayal of the homeland and trafficking of influence. Alegre also announced the convocation of demonstrations throughout the country on Friday. ', 
                'story_id': 'EFESP00020190801ef8100001:50066618', 
                'publisher': 'translateme2-pt', 
                'headline': '\nOposição confirma que pedirá impeachment de presidente do Paraguai; PARAGUAI GOVERNO (Pauta)\n', 
                'pub_date': '2019-08-01', 'contexts': ['corruption'], 
                'version': 'NGEC_coder-Vers001-b1-Run-001', 
                'attributes': {'ACTOR': [{'text': 'Mario Abdo Benítez', 'score': 0.1976235955953598}], 
                                'RECIP': [{'text': 'Fernando Lugo', 'score': 0.10433810204267502}], 
                                'LOC': [{'text': 'Paraguay', 'score': 0.24138706922531128}]}}
        ag.process([event])
        """
        for event in track(event_list, description="Resolving actors..."):
            ## get the date
            query_date = event['pub_date']
            if not query_date:
                query_date = "today"
            for k, block in event['attributes'].items():
                if k in ["LOC", "DATE"]:
                    continue
                for v in block:  # ACTOR, RECIP, and LOC are lists, but current just length 1
                    wiki = None
                    actor_text = v['text']
                    if type(actor_text) is not str:
                        print(actor_text)
                        actor_text = actor_text[0]
                    ## TO DO: get the country here
                    limit_word = ""
                    res = self.agent_to_code(actor_text, query_date=query_date, search_limit_term=limit_word)
                    if res:
                        if 'wiki' in res.keys():
                            v['wiki'] = res['wiki']
                        else:
                            v['wiki'] = ""
                        if 'actor_wiki_job' in res.keys():
                            v['actor_wiki_job'] = res['actor_wiki_job']
                        else:
                            v['actor_wiki_job'] = ""
                        if 'all_code1s' in res.keys():
                            v['all_code1s'] = res['all_code1s']
                        else:
                            v['all_code1s'] = []
                        if 'all_code2s' in res.keys():
                            v['all_code2s'] = res['all_code2s']
                        else:
                            v['all_code2s'] = []
                        v['country'] = res['country']
                        v['code_1'] = res['code_1']
                        v['code_2'] = res['code_2']
                        if 'query' in res.keys():
                            v['actor_role_query'] = res['query']
                        else:
                            v['actor_role_query'] = ""
                            #print(v)
                        if 'pattern' in res.keys():
                            v['actor_resolved_pattern'] = res['description']
                        else:
                            v['actor_resolved_pattern'] = ""
                        if 'conf' in res.keys():
                            v['actor_pattern_conf'] = float(res['conf'])
                        else:
                            v['conf'] = ""
                        if 'best_reason' in res.keys():
                            v['actor_resolution_reason'] = res['best_reason']
                        else:
                            v['actor_resolution_reason'] = ""
                    else:
                        v['wiki'] = ""
                        v['actor_wiki_job'] = ""
                        v['country'] = ""
                        v['code_1'] = ""
                        v['code_2'] = ""
                        v['actor_role_query'] = ""
                        v['actor_resolved_pattern'] = ""
                        v['actor_pattern_conf'] = ""
                        v['actor_resolution_reason'] = ""

        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_actor_resolution_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)
        return event_list



if __name__ == "__main__":
    import jsonlines

    ag = ActorResolver()
    with jsonlines.open("PLOVER_coding_201908_with_attr.jsonl", "r") as f:
        data = list(f.iter())

    out = ag.process(data)
    with jsonlines.open("PLOVER_coding_201908_with_actor.jsonl", "w") as f:
        f.write_all(out)

    """
    {'id': '20190801-2309-4e081644904c_COOPERATE_R',
 'date': '2019-08-01',
 'event_type': 'R',
 'event_mode': [],
 'event_text': 'Delegates of the Venezuelan president, Nicolas Maduro, and the leader objector Juan Guaidó resumed on Wednesday (31) conversations on the island of Barbados, sponsored by Norway, to seek a way out of the crisis in their country, announced the parties. "We started another round of sanctions under the mechanism of Oslo," indicated on Twitter Mr Stalin González, one of the envoys of Guaidó, parliamentary leader recognized as interim president by half hundred countries. The vice-president of Venezuela, Delcy Rodríguez, confirmed in a press conference that representatives of mature traveled to Barbados for the meetings with the opposition. Mature reaffirmed in a message to the nation that the government seeks to establish a "bureau for permanent dialog with the opposition, and called entrepreneurs and social movements to be added to the process. After exploratory approximations and a first face to face in Oslo in mid-May, the parties have transferred the dialog on 8 July for the caribbean island. The opposition search in the negotiations the output of mature and a new election, by considering that his second term, started last January, resulted from fraudulent elections, not recognized by almost 60 countries, among them the United States. ',
 'story_id': 'AFPPT00020190801ef81000jh:50066619',
 'publisher': 'translateme2-pt',
 'headline': '\nGoverno e oposição da Venezuela retomam diálogo em Barbados\n',
 'pub_date': '2019-08-01',
 'contexts': ['pro_democracy'],
 'version': 'NGEC_coder-Vers001-b1-Run-001',
 'attributes': {'ACTOR': {'text': 'Nicolas Maduro',
   'score': 0.23675884306430817,
   'wiki': 'Nicolás Maduro',
   'country': 'VEN',
   'code_1': 'ELI',
   'code_2': ''},
  'RECIP': {'text': 'Juan Guaidó',
   'score': 0.13248120248317719,
   'wiki': 'Juan Guaidó',
   'country': 'VEN',
   'code_1': 'REB',
   'code_2': ''},
  'LOC': {'text': 'Barbados', 'score': 0.4741457998752594}}}
    """ 