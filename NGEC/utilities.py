import pandas as pd
from spacy.tokens import Token
from spacy.language import Language
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#def make_country_dict():
#    country = pd.read_csv("assets/wikipedia-iso-country-codes.txt")
#    country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
#    country_dict["CUW"] = len(country_dict)
#    country_dict["XKX"] = len(country_dict)
#    country_dict["SCG"] = len(country_dict)
#    country_dict["SSD"] = len(country_dict)
#    country_dict["BES"] = len(country_dict)
#    country_dict["NULL"] = len(country_dict)
#    country_dict["NA"] = len(country_dict)
#    return country_dict
#
#
#with open("assets/feature_code_dict.json", "r") as f:
#    feature_code_dict = json.load(f)
#

def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass

    try:
        @Language.component("token_tensors")
        def token_tensors(doc):
            chunk_len = len(doc._.trf_data.tensors[0][0])
            token_tensors = [[]]*len(doc)

            for n, i in enumerate(doc):
                wordpiece_num = doc._.trf_data.align[n]
                for d in wordpiece_num.dataXd:
                    which_chunk = int(np.floor(d[0] / chunk_len))
                    which_token = d[0] % chunk_len
                    ## You can uncomment this to see that spaCy tokens are being aligned with the correct 
                    ## wordpieces.
                    #wordpiece = doc._.trf_data.wordpieces.strings[which_chunk][which_token]
                    #print(n, i, wordpiece)
                    token_tensors[n] = token_tensors[n] + [doc._.trf_data.tensors[0][which_chunk][which_token]]
            for n, d in enumerate(doc):
                if token_tensors[n]:
                    d._.set('tensor', np.mean(np.vstack(token_tensors[n]), axis=0))
                else:
                    d._.set('tensor',  np.zeros(doc._.trf_data.tensors[0].shape[-1]))
            return doc
    except ValueError:
        pass

def stories_to_events(story_list, doc_list=None):
    if not doc_list:
        logger.warning("Missing doc list...")
    if doc_list:
        if len(doc_list) != len(story_list):
            raise ValueError("the story list and list of spaCy docs must be the same length")
        for n, story in enumerate(story_list):
            doc = doc_list[n]
            story['story_people'] = list(set([i.text for i in doc.ents if i.label_ == "PERSON"]))
            story['story_organizations'] = list(set([i.text for i in doc.ents if i.label_ == "ORG"]))
            story['story_places'] = list(set([i.text for i in doc.ents if i.label_ in ["GPE", "LOC", "FAC"]]))
            story['_doc_position'] = n
    # "lengthen" the story-level data to generate a separate element
    # for each event type
    event_list = []
    for n, ex in enumerate(story_list):
        # event modes are formatted ["ACCUSE-disapprove", "ACCUSE-allege", "CONSULT-third-party"]
        modes = [i.split("-") for i in ex['event_mode']]
        events_with_modes = list(set([i[0] if i else None for i in modes]))
        for event_type in ex['event_type']:
            if event_type not in events_with_modes:
                event_mode = ""
                d = ex.copy() # note: the copy is important!
                d['event_type'] = event_type
                d['orig_id'] = d['id']
                d['event_mode'] = event_mode
                d['id'] = d['id'] + "_" + event_type + "_" # generate a new ID
                event_list.append(d)
            else:
                for et, *event_mode in modes:
                    # annoyingly, the event and mode are separated by a hyphen, but
                    # there are also hyphens within certain mode names. Merge those back
                    # together
                    event_mode = '-'.join([*event_mode])
                    if et != event_type:
                        # skip modes that are attached to the wrong event type
                        continue
                    d = ex.copy() # note: the copy is important!
                    d['event_type'] = event_type
                    d['orig_id'] = d['id']
                    d['event_mode'] = event_mode
                    d['id'] = d['id'] + "_" + event_type + "_" + event_mode # generate a new ID
                    event_list.append(d)
    return event_list