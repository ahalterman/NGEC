import pandas as pd
from datasets import Dataset
from transformers import pipeline
from rich.progress import track
import time
import jsonlines
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import traceback
import re
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _load_questions(base_path="assets/"):
    """
    Load in a CSV of questions for each event type and mode.

    Assumes that the CSV is titled "event_mode_questions.csv" and is in the base_path.

    Parameters
    ---------
    base_path: str

    Returns
    -------
    q_lookup: dict of dicts
      Each key is a string of the form "{event_type}_{mode}", e.g. "PROTEST_riot".
      Each value is a dictionary with the question types (e.g. ACTOR) as keys and the questions as values.
    """
    questions = pd.read_csv(os.path.join(base_path, "event_mode_questions.csv"))
    questions = questions.replace(np.nan, "")
    q_dict = questions.to_dict("records")
    q_lookup = {}
    for i in q_dict:
        if 'mode' in i.keys():
            key = i['event_type'] + "_" + i['mode']
        else:
            key = i['event_type'] + "_" 
        val = {"ACTOR": i['ACTOR'].split("\n"),
                "RECIP": i['RECIP'].split("\n"),
                "LOCATION": i['LOCATION'].split("\n"),
                "DATE": i['TIME'].split("\n")}
        q_lookup[key] = val
    return q_lookup


def expand_tokens(orig_span, mods=[]):
    """
    Expand out a token to include compound or amod tokens.

    For example, "States" --> "United States"

    Parameters
    ---------
    orig_span: spaCy Token

    mods: spaCy Token
      Anything that's been identified previously as an amod child, e.g. "solider" in 
        "Russian soldier"
    """
    if mods:
        mods = mods[0]
    final_list = []
    pop_list = [orig_span]

    # Any time we find a child token that's a compound or amod, add it to the list.
    # If that new token also has compound or amod children, keep adding them to the list.
    while pop_list:
        a = pop_list.pop()
        new = [i for i in a.children if i.dep_ in ['compound', 'amod']]
        pop_list.extend(new)
        final_list.append(a)

    if mods:
        final_list.append(mods)
    # remove duplicates (possible with amods?)
    final_list = list(set(final_list))
    # put it in sentence order
    final_list.sort(key=lambda x: x.i)
    return final_list


def make_dataset(qs):
    """
    Make a dataset from from the text ("context") and the questions.
    """
    df = pd.DataFrame(qs)
    prod_df = df.reset_index(drop=True)
    prod_df['context'] = prod_df['event_text']
    prod_df = prod_df[['context', 'question']] 
    prod_ds = Dataset.from_pandas(prod_df)
    return prod_ds



class AttributeModel:
    def __init__(self, 
                 model_dir="./assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457",
                 threshold=0.6, # we can set stuff like this here
                 silent=False, # whether to silence progress bars and logs
                 batch_size=8,
                 save_intermediate=False,
                 expand_actors=True,
                 gpu=False,
                 base_path="assets/"
                 ):
        """
        Intialize the attribute model

        Paramaters
        ---------
        """
        
        if gpu:
            self.device="cuda:0"
        else:
            self.device=-1
        logger.info(f"Device (-1 is CPU): {self.device}")
        print("Loading model")
        if model_dir == "allenai/unifiedqa-t5-large":
            print("Using T5 model")
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            self.qa_tokenizer =  T5Tokenizer.from_pretrained(model_dir)
            self.qa_tokenizer
            self.qa_model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.qa_model.to(self.device)
            self.t5 = True
        else:
            print("Using normal pipeline")
            self.qa_pipeline = pipeline('question-answering', 
                                    model=model_dir, 
                                    tokenizer=model_dir,
                                    handle_impossible_answer=True)
            self.t5 = False
        self.threshold = threshold
        self.silent=silent
        self.batch_size=batch_size
        self.expand_actors=expand_actors
        self.save_intermediate=save_intermediate
        self.q_lookup = _load_questions(base_path)
    
    def make_attr_q(self, att, event_type, mode="", actor_text=None):
        """
        Given an event type (from earlier step) and an attribute type,
        generate a question for the QA model.

        Parameters
        ---------
        att: str
            one of ['ACTOR', 'LOC', 'RECIP']
        event_type: str
            one of the PLOVER codes, e.g. "PROTEST"
        mode: str
            one of the PLOVER modes or ""

        Returns
        -------
        str: the question to give to the QA model
        """
        if mode is None:
            mode = ""
        lookup_key = event_type + "_" + mode

        if not actor_text:
            actor_text = "someone"

        if att == "ACTOR":
            try:
                raw_qs = self.q_lookup[lookup_key]['ACTOR']
                return [i.format(recip_text=actor_text) for i in raw_qs]
            except:
                return [f"Who did the {event_type.lower()} to someone?"]

        elif att == "LOC":
            try:
                raw_qs = self.q_lookup[lookup_key]['LOCATION']
                return [i.format(recip_text=actor_text) for i in raw_qs]
            except:
                return [f"Where did the {event_type.lower()} take place?"]
        elif att == "DATE":
            try:
                raw_qs = self.q_lookup[lookup_key]['DATE']
                return [i.format(recip_text=actor_text) for i in raw_qs]
            except:
                return [f"When did the {event_type.lower()} take place?"]
        elif att == "RECIP":
            try:
                raw_qs = self.q_lookup[lookup_key]['RECIP']
                return [i.format(actor_text=actor_text) for i in raw_qs]
            except:
                return [f"Who was the target of the {event_type.lower()}?"]
        else:
            raise ValueError(f"Attribute must be one of ACTOR,LOC,RECIP but you provided {att}.")


    def find_co_actors(self, qa, doc):
        """
        The QA model usually returns just one actor, but sometimes the story reports
        multiple actors and we can find them with dependency parses.

        Parameters
        ----------
        qa: dict
          The dict is the output of a QA model and takes the following form:
                    {"text": i['answer'],
                    "qa_score": float(i['qa_score']),
                    "qa_start_char": i['qa_start_char'],
                    "qa_end_char": i['qa_end_char'],
                    "_doc_position": i['_doc_position'],
                    "question": i['question']}

        Returns
        -------
        qa: dict
            A modified version of the input dict with the text field updated to include
            all actors found in the dependency parse. Also adds a new field, "qa_expanded_actor",
            which is a boolean indicating whether the actor was expanded.
        """
        try:
            spacy_token_start = [i for i in doc if i.idx == qa['qa_start_char']][0]
        except IndexError:
            spacy_token_start = [i for i in doc if abs(i.idx - qa['qa_start_char']) == 1][0]
        try:
            spacy_token_end = [i for i in doc if i.idx == (qa['qa_end_char'] - len(i))][0]
        except:
            try:
                spacy_token_end = [i for i in doc if abs(i.idx - (qa['qa_end_char'] - 1 - len(i))) == 1][0]
            except:
                try:
                    spacy_token_end = [i for i in doc if abs(i.idx - (qa['qa_end_char'] - 1 - len(i))) < 3][0]
                except:
                    return []

        spacy_answer = doc[spacy_token_start.i:spacy_token_end.i+1]

        # split conjunction
        conjs = []
        mods = []
        for c in spacy_answer:
            if c.dep_ == "conj":
                conjs.append(c)
            #
            subtree = list(c.subtree)
            if subtree:
                for s in subtree:
                    if s.dep_ == "conj":
                        conjs.append(list(s.ancestors)[0])
                        conjs.append(s)
            children = list(c.children)
            if children:
                for cc in children:
                    if cc.dep_ == "amod":
                        mods.append(c)

        conjs = list(set(conjs))

        cleaned = [expand_tokens(c, mods) for c in conjs]
        if cleaned:
            formatted = []
            for clean in cleaned:
                text = ''.join([i.text_with_ws for i in clean]).strip()
                q = qa.copy() # keep score, expanded actors, etc.
                q["text"] = text
                q["qa_start_char"] = clean[0].idx
                q["qa_end_char"] = clean[-1].idx + len(clean[-1])
                q["qa_expanded_actor"] = True
                formatted.append(q)
            return formatted
        else:
            return [qa]

    def expand_actor(self, qa, doc):
        try:
            spacy_token_start = [i for i in doc if i.idx == qa['qa_start_char']][0]
        except IndexError:
            spacy_token_start = [i for i in doc if abs(i.idx - qa['qa_start_char']) == 1][0]
        try:
            spacy_token_end = [i for i in doc if i.idx == (qa['qa_end_char'] - len(i))][0]
        except IndexError:
            spacy_token_end = [i for i in doc if abs(i.idx - (qa['qa_end_char'] - 1 - len(i))) == 1][0]
        spacy_answer = doc[spacy_token_start.i:spacy_token_end.i+1]

        first_expanded = [c for i in spacy_answer for c in i.children if c.dep_ in ["compound"]]
        first_expanded = [c for i in first_expanded for c in i.subtree]
        #appos = [i.head for i in spacy_answer[0].ancestors if i.dep_ == 'appos']# if i.dep_ in ["compound", "appos"]]
        appos = [i.head for i in spacy_answer if i.dep_ == 'appos']# if i.dep_ in ["compound", "appos"]]
        parent_subtree = []
        for i in appos:
            parent_subtree.extend(i.subtree)
        both = first_expanded + parent_subtree
        full_expanded = both.copy()
        for i in both:
            full_expanded.extend(list(i.subtree))
        full_expanded.sort(key=lambda x: x.i, reverse=False)
        extra_text = [i for i in full_expanded if i not in spacy_answer and i.pos_ != "PUNCT"]
        extra_text = list(set([i for i in extra_text]))
        extra_text.sort(key=lambda x: x.i, reverse=False)
        extra_text = ''.join([i.text_with_ws for i in extra_text]).strip()
        return extra_text

    def run_t5_model(self, input_string):
        input_ids = self.qa_tokenizer.encode(input_string, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        #res = model.generate(input_ids, num_beams=2, output_scores=True, return_dict_in_generate=True)
        if self.device.startswith("cuda"):
            input_ids.to(self.device)
        outputs = self.qa_model.generate(input_ids, 
                                         max_new_tokens=20,
                                         return_dict_in_generate=True, output_scores=True)
        spans = self.qa_tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        answer_text = spans[0]
        transition_scores = self.qa_model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
                )
        answer_score = np.mean(np.exp(transition_scores.cpu().numpy()))
        context = input_string.split("\\n")[1]
        if answer_text == 'no answer>':
            answer_text = ""
            start_char = 0
            end_char = 0
        elif not re.search(re.escape(answer_text.lower()), context.lower()):
            answer_text = ""
            start_char = 0
            end_char = 0
        else:
            start_char, end_char = re.search(re.escape(answer_text.lower()), context.lower()).span()

        output = {"answer": answer_text, 
                  "score": answer_score, 
                  "start": start_char, 
                  "end": end_char}
        return output

    def do_qa(self, event_list):
        """
        Iterate through the event list, generate questions for each event, and
        run the questions through the QA pipeline.
        """
        qs = []
        for event in event_list:
            for att in ['ACTOR', 'LOC', 'DATE', 'RECIP']:
                questions = self.make_attr_q(att, event_type=event['event_type'], mode=event['event_mode'])
                for q in questions:
                    if not q:
                        continue
                    d = event.copy()
                    d['question'] = q
                    d['attribute'] = att
                    if self.t5:
                        d['t5_question'] = d['question'] + "\\n" + d['event_text']
                    qs.append(d)

        # Step 2: put the data into a pandas dataframe
        # and from there into a huggingface dataset
        prod_ds = make_dataset(qs)
        
        # run the dataset through the pipeline. This should
        # handle batching automatically
        logger.debug(f"Running QA first pipeline on {len(qs)} questions")
        if not self.t5:
            # replace the code below with a non-pipeline version that applies
            # the model to the constructed dataset
            #all_out = self.qa_pipeline(prod_ds, batch_size=self.batch_size, device=self.device)
            # NOTE: use batch size of 16 for now. Don't use pipeline.
            all_out = self.qa_model(prod_ds['question'], prod_ds['context'], batch_size=16, device=self.device)
            

        else:
            all_out = []
            for i in qs:
                # batch this!
                out = self.run_t5_model(i['t5_question'])
                all_out.append(out)


        # all_out is just the raw output, so we need to add it back to the original inputs
        for n, i in enumerate(qs):
            i['answer'] = all_out[n]['answer']
            i['qa_score'] = float(all_out[n]['score'])
            i['qa_start_char'] = all_out[n]['start']
            i['qa_end_char'] = all_out[n]['end']

        ### ADD RECIPS
        rs = []
        for i in qs:
            if i['attribute'] == "ACTOR":
                att = "RECIP"
                questions = self.make_attr_q(att, event_type=i['event_type'], mode=i['event_mode'], actor_text=i['answer'])
                for q in questions:
                    if not q:
                        continue
                    d = i.copy()
                    d['question'] = q
                    d['attribute'] = att
                    d['qa_score'] = 0
                    d['qa_end_char'] = 0
                    d['qa_start_char'] = 0
                    d['answer'] = "---"
                    rs.append(d)
        # uniquify
        rs = list({v['question']:v for v in rs}.values())
        recip_ds = make_dataset(rs)
        recip_out = []
        logger.debug(f"Running QA recip step on {len(rs)} stories.")
        if not self.t5:
            for out in tqdm(self.qa_pipeline(recip_ds, batch_size=self.batch_size, device=self.device), 
                            total=len(rs), disable=self.silent):
                recip_out.append(out)
        else:
            for i in rs:
                # batch this!
                out = self.run_t5_model(i['t5_question'])
                recip_out.append(out)

        #for out in track(self.qa_pipeline(recip_ds, batch_size=self.batch_size, device=self.device)):
        #    recip_out.append(out)

        # Make sure it's in a list so we can iterate
        if len(recip_out) == 1:
            recip_out = [recip_out]
        # put the recip raw answers back into the rs list
        for n, i in enumerate(rs):
            try:
                i['answer'] = recip_out[n]['answer']
                i['qa_score'] = float(recip_out[n]['score'])
                i['qa_start_char'] = recip_out[n]['start']
                i['qa_end_char'] = recip_out[n]['end']
            except Exception as e:
                logger.debug(e)
                logger.debug(traceback.print_tb(e.__traceback__))
                logger.debug(recip_out)

        ### NOW, the actors one more time. REPEATED CODE SRY
        actor2 = []
        for i in rs:
            if i['attribute'] == "RECIP":
                att = "ACTOR"
                questions = self.make_attr_q(att, event_type=i['event_type'], mode=i['event_mode'], actor_text=i['answer'])
                for q in questions:
                    if not q:
                        continue
                    d = i.copy()
                    d['question'] = q
                    d['attribute'] = att
                    d['qa_score'] = 0
                    d['qa_end_char'] = 0
                    d['qa_start_char'] = 0
                    d['answer'] = "---"
                    actor2.append(d)
        # uniquify
        actor2 = list({v['question']:v for v in actor2}.values())
        actor2_ds = make_dataset(actor2)
        actor2_out = []
        logger.debug(f"Running second actor QA step on {len(actor2)} questions.")
        if not self.t5:
            for out in tqdm(self.qa_pipeline(actor2_ds, batch_size=self.batch_size, device=self.device), 
                            total=len(actor2), disable=self.silent):
                actor2_out.append(out)
        else:
            for i in actor2:
                # batch this!
                out = self.run_t5_model(i['t5_question'])
                actor2_out.append(out)

        #for out in track(self.qa_pipeline(recip_ds, batch_size=self.batch_size, device=self.device)):
        #    recip_out.append(out)

        if len(actor2_out) == 1:
            actor2_out = [actor2_out]
        for n, i in enumerate(actor2):
            try:
                i['answer'] = actor2_out[n]['answer']
                i['qa_score'] = float(actor2_out[n]['score'])
                i['qa_start_char'] = actor2_out[n]['start']
                i['qa_end_char'] = actor2_out[n]['end']
            except Exception as e:
                logger.debug(e)
                logger.debug(traceback.print_tb(e.__traceback__))

        both_qs = qs + rs + actor2
        return both_qs

    def pick_best_answers(self, q_dict_entry):
        """
        We ask multiple questions for each event type (both variants of the original question
        and different versions with the actor/recip filled in). This function picks the best
        answer for each attribute.

        Details: we sum the returned by the QA model for each answer. Then, for each span,
        we assign it to the best attribute using linear_sum_assignment from scipy.optimize.

        When there are overlapping spans (e.g. "Hindu nationalists" and "a group of Hindu nationalists"),
        we pick the version that's most common in the set of answers.

        TODO: allow "" to be an an answer for multiple categories
        """
        # if overlapping spans, ("Hindu nationalists", "a group of Hindu nationalists"), pick the 
        # one that's most common and just use that one.
        # BUT!! Examples like LOC = "Delhi" and DATE = "Dehli last week"
        # Current status: skip this, and use the SUM of the scores rather than the 
        # mean to over-weight common answers. Could play around with this. \sum{score}/log(len(scores))?

        # create a dictionary keyed on the text span, e.g.:
        # 'Muslim shops': {'ACTOR': [0.13081131875514984, 0.10325650125741959],
        #          'RECIP': [0.17858384549617767,...
        actor_text_dict = {}

        for att, v in q_dict_entry.items():
            for span in v:
                if span['text'] not in actor_text_dict.keys(): 
                    actor_text_dict[span['text']] = {att: [span['qa_score']]}
                else:
                    if att not in actor_text_dict[span['text']].keys():
                        actor_text_dict[span['text']][att] = [span['qa_score']]
                    else:
                        actor_text_dict[span['text']][att] = actor_text_dict[span['text']][att] + [span['qa_score']]

        # Create a second dictionary, also keyed to the text, to store the
        # QA info for later expansion
        actor_info_dict = {}
        for att, v in q_dict_entry.items():
            for span in v:
                if span['text'] not in actor_info_dict.keys():
                    raw_qa = span.copy()
                    del raw_qa['question']
                    del raw_qa['qa_score']
                    actor_info_dict[span['text']] = raw_qa

        # ~~average~~ SUM for now
        for span, v in actor_text_dict.items():
            for att, scores in v.items():
                #v[att] = -np.mean(scores)
                v[att] = -np.sum(scores)

        df = pd.DataFrame(actor_text_dict)
        df = df.fillna(0)
        assignment = linear_sum_assignment(df)
        # Returns a tuple. The second element is ordered by rows and gives the best column.
        # array([1, 0, 2, 3])) --> row 1 (ACTOR) should be column 1, row 2 (RECIP) should be column 0, etc.
        try:
            # this code is confusing, so here's what's going on:
            # - iterate through the second part of the `assignment` tuple, which is for each of the
            #   attributes.
            final_assign = {}
            for n, a in enumerate(assignment[1]):
                # Left side: pull out the *name* of the attribute (df.index[n]), based on the element 
                # in the assignment matrix. Why not hard code the names of the attributes? 
                # We can't guarentee their order.  dataframe corresponding
                # Right side: Get the original, full QA data (actor_info_dict) using
                # the span named that's stored in the (attribute, span) dataframe.
                final_assign[df.index[n]] = actor_info_dict[df.columns[a]]
        except:
            logger.debug("Error on assignment step")
            logger.debug(assignment)
        return final_assign
                

    def process(self, event_list, doc_list, all_qs=None, return_raw_spans=False,
                show_progress=False):
        """
        Given event records from the previous steps in the NGEC pipeline,
        run the QA model to identify the spans of text corresponding with
        each of the event attributes (e.g. ACTOR, RECIP, LOC, DATE.)

        Parameters
        --------
        event_list: list of event dicts. 
          At a minimum, it should entries the following keys:
            - event_text
            - id (id for the event)
            - _doc_position (needed to link back to the nlped list)
            - event_type
            - mode
        doc_list: list of spaCy NLP docs
        expand: bool
          Expand the QA-returned answer to include appositives or compound words?
        all_qs: list of dicts, optional
          Use this to pass in the output of the QA step. Useful for running experiments
          where the QA output is saved separately.
        return_raw_spans: bool
            If True, return the raw spans from the QA model, rather than the "best" answer
        show_progress: bool
            If True, show a tqdm progress bar.

        Returns
        -----
        event_list: list of dicts
          Adds 'attributes', which looks like: {'ACTOR': [{'text': 'Mario Abdo Benítez', 'score': 0.19762}], 
                                                'RECIP': [{'text': 'Fernando Lugo', 'score': 0.10433}], 
                                                'LOC': [{'text': 'Paraguay', 'score': 0.24138}]}
        """
        if self.expand_actors and not doc_list:
            raise ValueError("If 'expand_actors' is True, you must provide a list of nlped docs.")
        # Step 1: further lengthen the data to generate separate elements
        # for each attribute/question, so we have unique (ID, event_cat, attribute) 
        logger.debug("Starting attribute process")

        # check if all event types have a question
        event_types = set([i['event_type'] for i in event_list])
        event_questions = set([i[:-1] for i in self.q_lookup.keys()])
        diff = event_types - event_questions


        if not all_qs:
            all_qs = self.do_qa(event_list)

        # now we need to reverse the steps we did at the beginning, and go
        # from (ID, event_cat, question) to just (ID, event_cat)
        q_dict = {}
        for i in tqdm(all_qs, disable=not show_progress):
            #doc = doc_list[i['_doc_position']] # 
            #try:
            #    exp = self.expand_actor(i, doc)
            #except:
            #    exp = ""
            entry = {"text": i['answer'],
                    "qa_score": float(i['qa_score']),
                    "qa_start_char": i['qa_start_char'],
                    "qa_end_char": i['qa_end_char'],
                    "_doc_position": i['_doc_position'],
                    "question": i['question']}
            entries = [entry]
            #try:
            #    entries = self.find_co_actors(entry, doc)
            #except Exception as e:
            #    logger.info(f"Error on find_co_actors: {e}")
            #    entries = [entry]
            #except:
            #    #logger.debug(f"expand_actor error on {i, doc}")
            #    pass
            # Create a dictionary keyed to the event id for merging
            if i['id'] in q_dict.keys():
                if i['attribute'] not in q_dict[i['id']].keys():
                    q_dict[i['id']][i['attribute']] = entries
                else:
                    q_dict[i['id']][i['attribute']] = q_dict[i['id']][i['attribute']] + entries
            else:
                q_dict[i['id']] = {i['attribute']: entries}

        if return_raw_spans:
            return q_dict
        # Pick the best one here
        # we want something keyed to the actor spans, e.g. ["YPG militants", "Turkish forces"]
        # For each of those, we want to know whether it's a better fit for ACTOR or RECIP
        # Also disallow having the same span in both roles. That's what `pick_best_answers` does.
        final_attributes = {}
        for event_id, v in q_dict.items():
            final_attributes[event_id] = {}
            best_attr = self.pick_best_answers(v)
            for attr, qa in best_attr.items():
                try:
                    exp = self.find_co_actors(qa, doc_list[qa['_doc_position']])
                except:
                    exp = []
                for i in exp:
                    for k in ['_doc_position', 'qa_expanded_actor']:
                        if k in i.keys():
                            del i[k]
                final_attributes[event_id][attr] = exp 


        # Now, at the very end, put the results back into the event list.
        for i in event_list:
            i['attributes'] = final_attributes[i['id']]

        if self.save_intermediate:
            fn = time.strftime("%Y_%m_%d-%H") + "_attribute_output.jsonl"
            with jsonlines.open(fn, "w") as f:
                f.write_all(event_list)

        return event_list


if __name__ == "__main__":
    import jsonlines
    import utilities
    import spacy
    nlp = spacy.load("en_core_web_sm") 

    data = [
        {"event_text": "A group of Hindu nationalists rioted in Dehli last week, burning Muslim shops.",
        "id": 123,
        "_doc_position": 0,
        "event_type": "PROTEST",
        "event_mode": "riot"},
        {"event_text": "Turkish forces battled with YPG militants in Syria.",
        "id": 456,
        "_doc_position": 1,
        "event_type": "ASSAULT",
        "event_mode": ""},
        {"event_text": "Turkish forces and Turkish-backed militias battled with YPG militants in Syria.",
        "id": 789,
        "_doc_position": 2,
        "event_type": "ASSAULT",
        "event_mode": ""}
    ]

    doc_list = list(track(nlp.pipe([i['event_text'] for i in data])))

    event_list = utilities.stories_to_events(data, doc_list)
    qa_model = AttributeModel(model_dir = "NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457",
                             base_path = "NGEC/assets",
                             silent=False)

    output = qa_model.process(event_list, doc_list)

    print(output)   