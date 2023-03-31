import multiprocessing
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import mwxml
import mwparserfromhell
import re
from tqdm import tqdm
from textacy.preprocessing.remove import accents as remove_accents
from bz2 import BZ2File as bzopen
import pickle
import plac
import os
import redis
import json
import datetime

import logging

logger = logging.getLogger()
handler = logging.FileHandler("wiki_es.log")
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

es_logger = elasticsearch.logger
es_logger.setLevel(elasticsearch.logging.WARNING)

def get_redirect(page, title=None, text=None):
    if not title and not text and page:
        text = next(page).text
        if not page:
            logger.debug("not page")
            return None
        title = page.title
        if not page:
            return None
    if not text:
        return None

    wikicode = mwparserfromhell.parse(str(text))

    raw_intro = wikicode.get_sections()[0]
    intro_para = raw_intro.strip_code()
    if re.match("#?(REDIRECT|redirect)", intro_para):
        # skip/ignore redirects for now
        return None


redirect_pattern = re.compile("#?(REDIRECT|redirect|Redirect)")

def get_page_redirect(page, title, text):
    """Returns (original page, new page to redirect to)"""
    wikicode = mwparserfromhell.parse(text)
    raw_intro = wikicode.get_sections()[0]
    if re.match(redirect_pattern, str(raw_intro)):
        new_page = re.findall(r"\[\[(.+?)\]\]", str(raw_intro))
        try:
            new_page = new_page[0]
        except:
            return None
        # Too many false positives come from this splitting. Keep as-is instead, even
        # if that means it won't get added to any articles.
        #new_page = new_page.split("#")[0]
        return (str(title), str(new_page))


def clean_names(name_list):
    if not name_list:
        return []
    name_list = [re.sub("\|.+?\]\]", "", i).strip() for i in name_list]
    name_list = [re.sub("\[|\]", "", i).strip() for i in name_list]
    # There are some weird entries here like "son:"
    name_list = [i for i in name_list if not i.endswith(":")]
    de_accent = [remove_accents(i) for i in name_list]
    name_list = name_list + de_accent
    name_list = list(set(name_list))
    return name_list


def parse_wiki_article(page, title=None, text=None, use_redis=True):
    """
    Go through a Wikipedia dump and format the article so it's useful for us.

    Pull out the article's:
    - title
    - short desc: (new!) it's similar to the Wikidata short description
    - first para
    - redirects (from Redis)
    - alternative names (anything bold in the first para)
    - info box
    """
    # These had errors earlier: pull them out separately for inspection.
    if title in ['Kyle Rittenhouse', 'Dmitry Peskov', 'Warsaw', 'Brasília', 'Beirut', 'Muhammadu Buhari',
                'Anil Deshmukh', 'Viktor Orbán']:
        print(f"found article: {title}")
        with open(f"error_articles/list/{title}.txt", "w") as f:
            f.write(text)
    if not title and not text and page:
        if not page:
            logger.debug(f"not page: {title}")
            return None
        text = next(page).text
        title = page.title
    if not text:
        logger.debug(f"No text for {title}")
        return None

    # There are a whole bunch of article types that we want to skip
    if title.endswith(".jpg") or title.endswith(".png"):
        logger.debug(f"Skipping image: {title}")
        return None
    if re.search("\-stub", title):
        logger.debug(f"Skipping Stub: {title}")
        return None
    if re.match("(User|Selected anniversaries)", title):
        logger.debug(f"Skipping User: {title}")
        return None
    if re.search("\([Dd]isambiguation\)", title):
        logger.debug(f"Skipping Disambig: {title}")
        return None
    if re.search("Articles for deletion", title):
        logger.debug(f"Skipping For deletion: {title}")
        return None
    if re.match("List ", title):
        logger.debug(f"Skipping List: {title}")
        return None
    if re.match("Portal ", title):
        logger.debug(f"Skipping Portal: {title}")
        return None
    if re.search("Today's featured article", title):
        logger.debug(f"Skipping featured article: {title}")
        return None
    if re.search("Featured article candidates", title):
        logger.debug(f"Skipping featured article candidate: {title}")
        return None
    if title.startswith("Peer review/"):
        logger.debug(f"Skipping peer review article: {title}")
        return None
    if title.startswith("Requests for adminship/"):
        logger.debug(f"Skipping adminship: {title}")
        return None
    if title.startswith("Featured list candidates/"):
        logger.debug(f"Skipping list candidates: {title}")
        return None
    if title.startswith("Sockpuppet investigations/"):
        logger.debug(f"Skipping sockpuppt: {title}")
        return None
    # clean up intro para? [[File:Luhansk raions eng.svg|thumb|100px|Raions of Luhansk]]
    # also delete the leftover alt names parentheses? 
    # "[[File:Luhansk raions eng.svg|thumb|100px|Raions of Luhansk]]\nLuhansk,(, ; , , , ; , ), also known as Lugansk and formerly known as Voroshilovgrad (1935-1958)"    

    wikicode = mwparserfromhell.parse(str(text))

    raw_intro = wikicode.get_sections()[0]
    intro_para_raw = raw_intro.strip_code()
    # strip out the occasional stuff that slips through
    intro_para = re.sub("(\[\[.+?\]\])", "", intro_para_raw).strip()
    # delete thumbs (not removed by strip_code()):
    intro_para = re.sub("^thumb\|.+?\n", "", intro_para)
    # do it again, the lazy way
    intro_para = re.sub("^thumb\|.+?\n", "", intro_para)
    # delete the first set of paratheses
    intro_para = re.sub("\(.+?\)", "", intro_para, 1)
    if not intro_para:
        logger.debug(f"No intro para for {title}.")
        #logger.debug(f"{wikicode.get_sections()[:2]}")
        return None
    if re.match("#?(REDIRECT|redirect|Redirect)", intro_para):
        logger.debug(f"Detected redirect in first para: {title}")
        # skip/ignore redirects for now
        return None
    if re.search("\*?\n?Category\:", intro_para):
        logger.debug(f"Category: {title}")
        return None
    if intro_para.startswith("Category:"):
        logger.debug(f"Category: {title}")
        return None
    if intro_para.startswith("<noinclude>"):
        logger.debug(f"Sneaky category? {title}")
        return None
    if re.search("may refer to", intro_para[0:100]):
        logger.debug(f"may refer to: {title}")
        return None
    if re.search("most often refers", intro_para[0:100]):
        logger.debug(f"most often refers: {title}")
        return None
    if re.search("most commonly refers", intro_para[0:100]):
        logger.debug(f"most commonly refers: {title}")
        return None
    if re.search("[Pp]ortal\:", intro_para[0:100]):
        logger.debug(f"Portal: {title}")
        return None
    alternative_names = re.findall("'''(.+?)'''", str(raw_intro))

    redirects = []
    if use_redis:
        redis_db = redis.StrictRedis(host="localhost", port=6379, db=0, charset="utf-8", decode_responses=True)
        redirects = redis_db.get(title)
        if redirects:
            redirects = redirects.split(";")

    if re.match("Categories for", title):
        return None

    try:
        short_desc = re.findall("\{\{[Ss]hort description\|(.+?)\}\}", str(raw_intro))[0].strip()
    except:
        logger.debug(f"Error getting short desc for {title}")
        #title_mod = re.sub("/", "_", title)
        #with open(f"error_articles/short_desc/{title_mod}.txt", "w") as f:
        #    f.write(str(raw_intro))
        short_desc = ""


    params = {"title": title,
             "short_desc": short_desc,
             "intro_para": intro_para.strip(),
             "alternative_names": clean_names(alternative_names),
             "redirects": clean_names(redirects),
             "affiliated_people": [],
             "box_type": None}

    for template in wikicode.get_sections()[0].filter_templates():
        if re.search("[Ii]nfobox", template.name.strip()):
            # do it this way to prevent overwriting
            info_box = {p.name.strip(): p.value.strip_code().strip() for p in template.params}
            params['infobox'] = info_box
            params['box_type'] = re.sub("Infobox", "", str(template.name)).strip()
            break

    if 'infobox' in params.keys():
        for k in ['name', 'native_name', 'other_name', 'alias', 'birth_name', 'nickname', 'other_names']:
            if k in params['infobox'].keys():
                newline_alt = [i.strip() for i in params['infobox'][k].split("\n") if i.strip()]
                new_alt = [j.strip() for i in newline_alt for j in i.split(",")]
                params['alternative_names'].extend(new_alt)

        affiliated_people = []
        for k in ['leaders', 'founded_by', 'founder']:
            if k in params['infobox'].keys():
                aff_people = [i.strip() for i in params['infobox'][k].split("\n") if i.strip()]
                aff_people = [j.strip() for i in aff_people for j in i.split(",")]
                affiliated_people.extend(aff_people) 

        params['affiliated_people'] = clean_names(affiliated_people)
        params['alternative_names'] = clean_names(params['alternative_names'])


    raw_categories = wikicode.get_sections()[-1].strip_code()
    categories = re.findall("Category:(.+?)\n", raw_categories)
    params['categories'] = categories

    if 'infobox' in params.keys():
        for k in ['map']:
            if k in params['infobox'].keys():
                del params['infobox'][k]
    
    params['update'] = datetime.date.today().isoformat()
    logger.debug(f"Good article: {title}")

    if title in ['Kyle Rittenhouse', 'Dmitry Peskov', 'Warsaw', 'Brasília', 'Beirut', 'Muhammadu Buhari',
                'Anil Deshmukh', 'Viktor Orbán']:
        with open(f"error_articles/list/{title}.json", "w") as f:
            json.dump(params, f)
    return params

def wrapper_loader(title, text, page=None):
    res = parse_wiki_article(page, title, text)
    if not res:
        return None
    action = {"_index" : "wiki",
                      #"_id" : res['title'], # it turns out the titles aren't globally unique, so can't use as an ID
                      "_source" : res}
    return action


def load_batch_es(page_batch, p, es):
    actions = [p.apply_async(wrapper_loader, (title, text)) for title, text in page_batch if title]
    actions = [i.get() for i in tqdm(actions, leave=False) if i]
    actions = [i for i in actions if i]
    try:
        helpers.bulk(es, actions, chunk_size=-1, raise_on_error=False)
        logger.info("Bulk loading success")
    except Exception as e:
        logger.info(f"Error in loading Wiki batch!!: {e}. Loading stories individually...")
        for i in actions:
            try:
                response = helpers.bulk(es, i, chunk_size=-1, raise_on_error=False)
                if response[1]:
                    logger.info(f"Error on loading story {i}: {response[1]}")
            except Exception as e:
                logger.info(f"Skipping single Wiki story {e}")



def redirect_wrapper(title, text):
    redir = get_page_redirect(None, title, text)
    if redir:
        if redir[1] not in redirect_dict.keys():
            redirect_dict[redir[1]] = [redir[0]]
        else:
            redirect_dict[redir[1]] = list(set(redirect_dict[redir[1]] + [redir[0]]))


def read_clean_redirects():
    files = os.listdir()
    versions = [int(re.findall("dict_(\d+)\.", i)[0]) for i in files if re.match("redirect_dict", i)]
    max_file = f"redirect_dict_{max(versions)}.0.pkl"
    logger.info(f"Loading {max_file} into redis")
    with open(max_file, "rb") as f:
        redirect_dict = pickle.load(f)

    # Merge lowercase versions of keys with their non-lowercase
    #len = 1132887
    del_list = []
    for k in redirect_dict.keys():
        if k.lower() in redirect_dict.keys():
            redirect_dict[k] = redirect_dict[k] + redirect_dict[k.lower()]
            del_list.append(k.lower())

    for d in del_list:
        if d in redirect_dict.keys():
            del redirect_dict[d]
    # len = 1106119
    return redirect_dict
    

@plac.pos('process', "Which process to run?", choices=['build_links', 'load_redis', 'load_es'])
@plac.pos('file', "Wikiepdia dump location")
@plac.pos('es_batch', "Elasticsearch batch size")
@plac.pos('threads', "number of threads to use")
def process(process, file="enwiki-latest-pages-articles.xml.bz2", es_batch=5000, threads=10):
    p = multiprocessing.Pool(threads)
    logger.info(f"Reading from {file}")
    if re.search("bz2", file):
        dump = mwxml.Dump.from_file(bzopen(file, "r"))
    else:
        dump = mwxml.Dump.from_file(file)

    #dump = mwxml.Dump.from_file(open("Wikipedia-protest-export.xml"))
    # 1 core   = 11.077 total
    # 5 cores  = 3.254 total
    # 10 cores = 3.075 total
    
    if process == "build_links":
        redirect_dict = {}
        logger.info("Building redirect link dictionary...")
        page_batch = []
        for n, page in tqdm(enumerate(dump), total=22373694):
            if n % 1000000 == 0 and n > 0:
                k = n / 1000000
                with open(f"redirect_dict_{k}.pkl", "wb") as f:
                    pickle.dump(redirect_dict, f)
                    logger.info(f"Dumped at {k} x 1,000,000")
                #break
            #    continue
            if page:
                page_batch.append((page.title, next(page).text))
            if len(page_batch) % 5000 == 0:
                if re.search("bz2", file):
                    #actions = [p.apply_async(wrapper_loader, (title, text)) for title, text in page_batch if page]
                    actions = [p.apply_async(get_page_redirect, (None, title, text)) for title, text in page_batch if page]
                else:
                    actions = [p.apply_async(get_page_redirect, (page, None, None)) for page in page_batch if page]
                actions = [i.get() for i in tqdm(actions, leave=False) if i]
                #actions = [i for i in actions if i]
                for redir in tqdm(actions, leave=False):
                    if not redir:
                        continue
                    if redir[1] not in redirect_dict.keys():
                        redirect_dict[redir[1]] = [redir[0]]
                    else:
                        redirect_dict[redir[1]] = list(set(redirect_dict[redir[1]] + [redir[0]]))
                page_batch = []
        # get the final batch
        # This one isn't wrapped in a function to make sure redirect_dict stays in the right scope
        for redir in tqdm(actions, leave=False):
            if not redir:
                continue
            if redir[1] not in redirect_dict.keys():
                redirect_dict[redir[1]] = [redir[0]]
            else:
                redirect_dict[redir[1]] = list(set(redirect_dict[redir[1]] + [redir[0]]))
        with open(f"redirect_dict_{k+1}.pkl", "wb") as f:
            pickle.dump(redirect_dict, f)


    elif process == "load_redis":
        logger.info("Reading redirect dict...")
        redirect_dict = read_clean_redirects()
        redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
        pipe = redis_db.pipeline()
        for n, item in tqdm(enumerate(redirect_dict.items()), total=len(redirect_dict)):
            k, v = item
            v_str = ";".join(v)
            pipe.set(k, v_str)
            if n % 1000 == 0:
                pipe.execute()
        # get the final batch
        pipe.execute()

    elif process == "load_es":
        logger.info("Loading Wikipedia into Elasticsearch")
        es = Elasticsearch(urls='http://localhost:9200/', timeout=60, max_retries=2)

        page_batch = []
        for n, page in tqdm(enumerate(dump), total=21726007):
            if page:
                page_batch.append((page.title, next(page).text))
            if len(page_batch) % es_batch == 0:
                #logger.debug(f"Loaded {page.title}")
                load_batch_es(page_batch, p, es)
                page_batch = []
        # load final batch
        load_batch_es(page_batch, p, es)


if __name__ == '__main__':
    plac.call(process)


