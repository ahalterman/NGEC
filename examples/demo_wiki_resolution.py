import jsonlines
import spacy
from NGEC import ActorResolver
from tqdm import tqdm
from pprint import pprint
import pandas as pd

# NOTE: Make sure you have NGEC installed.
# From the main NGEC repo, install the requirements then run `pip install -e .`
# Also make sure you have the offline Wikipedia index installed.
# See https://github.com/ahalterman/NGEC#quick-start

# Change the logging levels--Elasticsearch is very verbose
import logging
logging.getLogger("NGEC.actor_resolution").setLevel(logging.WARNING)

es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)

# Load the spaCy model we'll use for named entity recognition
nlp = spacy.load("en_core_web_sm")

# Load the sample data
data = pd.read_csv("Guardian_SDF_sample.csv.zip", compression='zip')

# Instantiate the model.
# This assumes that you're running the code from the NGEC/docs directory.
actor_resolution_model = ActorResolver(spacy_model=nlp, 
                                           base_path="../NGEC/assets/", 
                                           save_intermediate=False, 
                                           gpu=False) # Set to True if you have a GPU

# Run spaCy over the docs
docs = list(nlp.pipe([i['text'] for i in data]))

# iterate through the docs, making a list of lists with the PERSON and ORG entities
entities = []
for doc in docs:
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG']:
            ent_text = ent.text
            d = {"entity": ent_text,
                 # make sure to append the sentence text--we'll use this for context
                 "context": ent.sent.text}
            entities.append(d)

# Now iterate through the extracted entities and resolve them to Wikipedia
wikis = []
for ent in tqdm(entities):
    wiki = actor_resolution_model.query_wiki(ent['entity'], context = ent['context'])
    if not wiki:
        wiki = {"search_term": ent['entity'],
                "title": None}
    else:
        wiki['search_term'] = ent['entity']
    wikis.append(wiki)


# Print out an example 50 results
for i in wikis[400:450]:
    try:
        short_desc = i['short_desc']
    except KeyError:
        short_desc = "None"
    print(f"{i['search_term']:<30} ---> {i['title']} ({short_desc})")



## Example of categorizing actors using their linked Wikipedia pages

wiki = actor_resolution_model.query_wiki("Ben Rhodes", context = "The former Obama adviser Ben Rhodes said: “We all owe him our gratitude – he literally made us safer.”")
code = actor_resolution_model.wiki_to_code(wiki)
pprint(code)

wiki = actor_resolution_model.query_wiki("Niloufar Hamedi", context = "The two journalists are Niloufar Hamedi, who broke the news of Amini’s death for wearing her headscarf too loose, and Elaheh Mohammadi, who wrote about Amini’s funeral.")
code = actor_resolution_model.wiki_to_code(wiki)
pprint(code)

### The code below lets you explore the output of the model in a little more detail. ###

## Print out the full logs--this will give you more detail on how many
## candidate wikipedia matches there are.
logging.getLogger("NGEC.actor_resolution").setLevel(logging.DEBUG)

# Experiment with upper/lower case, including/excluding context, etc.
actor_resolution_model.query_wiki("the International Rescue Committee")
actor_resolution_model.query_wiki("Isis")
actor_resolution_model.query_wiki("Isis", context="Fighting continues in Syria with the terrorist group Isis.")
actor_resolution_model.query_wiki("ISIS", context="Fighting continues in Syria with the terrorist group ISIS.")
actor_resolution_model.query_wiki("ISIS")


# Example where coding fails without context
sdf = actor_resolution_model.search_wiki("SDF", fuzziness=0)


## Code to explore how the context similarity model works

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def load_trf_model(model_dir='sentence-transformers/paraphrase-MiniLM-L6-v2'): ## Change to offline!!
    model = SentenceTransformer(model_dir)
    return model
trf = load_trf_model()

doc = "The SDF (the Kurdish led force raised by Washington to fight Isis) and the United States are sitting on a volcano in north-east Syria, with tens of thousands of foreign fighters and families in cramped detention centres."
encoded = trf.encode(doc)
res = actor_resolution_model.search_wiki("SDF")

intro_paras = [i['intro_para'][0:200] for i in res]
encoded_intros = trf.encode(intro_paras)

sims = cos_sim(encoded, encoded_intros)[0]
res[sims.argmax()]