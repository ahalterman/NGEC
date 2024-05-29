import pandas as pd
from spacy.tokens import Token
from spacy.language import Language
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def spacy_doc_setup():
    try:
        Token.set_extension('tensor', default=False)
    except ValueError:
        pass
    try:
        @Language.component("token_tensors")
        def token_tensors(doc):
            tensors = doc._.trf_data.last_hidden_layer_state
            for n, d in enumerate(doc):
                if tensors[n]:
                    d._.set('tensor', tensors[n])
                else:
                    d._.set('tensor',  np.zeros(tensors[0].shape[-1]))
            return doc
    except ValueError:
        pass

### TESTING ###
### Comment this out and run to verify that the new 3.7+ version of spaCy works
#import spacy
#nlp = spacy.load("en_core_web_trf")
#spacy_doc_setup()
#nlp.add_pipe("token_tensors")
#
#doc = nlp("We visited Berlin and Alexanderplatz.")
#doc[3]._.tensor
####

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