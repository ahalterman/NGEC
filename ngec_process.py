from NGEC import  AttributeModel
from NGEC import ActorResolver
from NGEC import GeolocationModel
from NGEC import Formatter
from NGEC import utilities

import spacy
from tqdm import tqdm
from rich import print
from rich.progress import track
import plac
from pathlib import Path
import re

import logging
from rich.logging import RichHandler

logger = logging.getLogger('main')
handler = RichHandler() 
#formatter = logging.Formatter(
#        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for i in loggers:
    if re.search("NGEC\.", i.name):
        i.addHandler(handler) 
        i.setLevel(logging.INFO)
        i.propagate = False
    if re.search("elasticsearch", i.name):
        i.addHandler(handler) 
        i.setLevel(logging.WARNING)

#loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
#print(loggers)

# we need to keep the raw tensors for each token

def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp


def read_input(input_file="NGEC/PLOVER_coding_201908_220302-1049.jsonl", max_stories=10):
    import jsonlines
    """
    Read in Factiva stories and return a list of stories for processing

    TODO:
    - clean up new lines/whitespace at the beginning and end of headlines and stories
    - do Phil's dateline remover here?

    Parameters
    ----------
    ????: ????
      Probably from a file, but possibly from a DB

    Returns
    -------
    stories: list of dicts
      - text
      - title
      - publication
      - date
    """
    if max_stories > 0:
        logger.info(f"Limiting to the first {max_stories} stories.")
    with jsonlines.open(input_file, "r") as f:
        data = list(f.iter())
    return data[:max_stories]

@plac.pos('input_file', "JSONL input file with events, modes, and contexts")
@plac.opt('max_stories', "Max stories to code", type=int)
@plac.opt('attribute_dir', "Location of the QA attribute model", type=str)
@plac.opt('base_path', "Location of the other models and files", type=Path)
@plac.opt('save_intermediate', "Write output of each intermediate step?", type=bool)
@plac.opt('geo_model', "Location of the geolocation model", type=Path)
@plac.opt('gpu', "Set to True if GPU is available", abbrev='d', type=bool)
def ngec(input_file="NGEC/PLOVER_coding_201908_220302-1049.jsonl",
        max_stories=-1,
        attribute_dir="NGEC/assets/roberta-base-squad2_2022-08-02",
        base_path="NGEC/assets/",
        save_intermediate=False,
        expand_actors=True,
        geo_model="../mordecai3/mordecai_2023-02-07_good.pt",
        gpu=False):

    utilities.spacy_doc_setup()
    nlp = load_nlp()

    # Initialize the processing models/objects
    #event_model = EventClassModel()
    #context_model = ContextModel()
    #mode_model = ModeModel()
    logger.info("Loading geolocation model...")
    geolocation_model = GeolocationModel(geo_model, 
                                        geo_path = "../mordecai3/mordecai3/assets/",
                                        save_intermediate=save_intermediate)
    attribute_model = AttributeModel(attribute_dir, 
                                    silent=True, 
                                    gpu=gpu, 
                                    save_intermediate=save_intermediate, 
                                    expand_actors=expand_actors,
                                    base_path=base_path)
    actor_resolution_model = ActorResolver(spacy_model=nlp, 
                                           base_path=base_path, 
                                           save_intermediate=save_intermediate, 
                                           gpu=gpu)
    formatter = Formatter(base_path=base_path)

    # Read in the stories
    story_list = read_input(input_file, max_stories)

    just_text = [i['event_text'] for i in story_list]
    doc_list = list(track(nlp.pipe(just_text), total=len(just_text), description="nlping docs..."))

    #story_list = event_model.process(story_list)
    #story_list = mode_model.process(story_list)
    #story_list = context_model.process(story_list)
    logger.info("Geolocating events...")
    story_list = geolocation_model.process(story_list, doc_list)

    event_list = utilities.stories_to_events(story_list, doc_list)
    logger.debug("Post-event split")
    logger.debug(f"{event_list[0]}")
    #event_list = mode_model(event_list)

    logger.info("Running attribute model...")
    event_list = attribute_model.process(event_list, doc_list)
    #print(event_list[0])
    logger.info("Running actor resolution model...")
    event_list = actor_resolution_model.process(event_list, doc_list)
    #print(event_list[0])

    logger.info("Formatting results...")
    cleaned_events = formatter.process(event_list)
    logger.info("Completed processing.")

if __name__ == "__main__":
    plac.call(ngec)