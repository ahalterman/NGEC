from NGEC import  AttributeModel
from NGEC import ActorResolver
from NGEC import GeolocationModel
from NGEC import Formatter
from NGEC import utilities

import streamlit as st


import spacy
from tqdm import tqdm
from rich import print
from rich.progress import track
import plac
from pathlib import Path
import re

# stuff that's just used to allow streamlit cacheing
import preshed
import cymem
import spacy_transformers
import thinc

st.markdown("## NGEC test interface")

st.markdown("Put in some story text, an event type, and (optionally) a mode to see what NGEC produces.")
st.markdown("Intermediate output is also returned but hidden by default.")
st.markdown("The attribute step uses the faster tinybert model rather than the more accurate but slow BERT model.")

@st.cache(allow_output_mutation = True)
def load_nlp():
    utilities.spacy_doc_setup()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

nlp = load_nlp()


save_intermediate=False
attribute_dir="NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457"
base_path="./NGEC/assets/"
save_intermediate=False
expand_actors=True
geo_model="../mordecai3/mordecai_new.pt"
gpu=False

@st.cache(allow_output_mutation = True)
def load_geo(save_intermediate=save_intermediate):
    geolocation_model = GeolocationModel(geo_model, save_intermediate=save_intermediate)
    return geolocation_model

@st.cache(allow_output_mutation = True)
def load_attr(attribute_dir=attribute_dir, silent=True, gpu=gpu, save_intermediate=save_intermediate, expand_actors=expand_actors,
             base_path=base_path):
    attribute_model = AttributeModel(attribute_dir,
                                    silent=silent,
                                    gpu=gpu,
                                    save_intermediate=save_intermediate,
                                    base_path=base_path,
                                    expand_actors=expand_actors)
    return attribute_model

@st.cache(allow_output_mutation = True, hash_funcs={preshed.maps.PreshMap: lambda _: None,
                                                    cymem.cymem.Pool: lambda _: None,
                                                    spacy_transformers.layers.transformer_model.TransformerModel: lambda _: None,
                                                    spacy_transformers.layers.listener.TransformerListener: lambda _: None,
                                                    thinc.model.Model: lambda _: None})
def load_resolution(nlp=nlp, base_path=base_path, save_intermediate=save_intermediate, gpu=gpu):
    actor_resolution_model = ActorResolver(spacy_model=nlp, base_path=base_path, save_intermediate=save_intermediate, gpu=gpu)
    return actor_resolution_model

@st.cache(allow_output_mutation = True)
def load_formatter(base_path=base_path):
    formatter = Formatter(base_path=base_path)
    return formatter

geolocation_model = load_geo()
attribute_model = load_attr(base_path=base_path)
actor_resolution_model = load_resolution()
formatter = load_formatter()

text = st.text_area("Input text", "German troops withdrew from their area of operations in Kandahar last week.")
st.markdown("We don't have the event and mode models so input those manually")
event_type = st.text_input("Event type", "RETREAT")
event_mode = st.text_input("Mode type", "")

if not text or not event_type:
    st.warning("You must add both text and an event type before the model will run.")

if text and event_type:
    doc_list = [nlp(text)]

    story_list = [{"event_text": text, "id": "123", "event_type": [event_type], "event_mode": [event_mode]}]

    story_list = geolocation_model.process(story_list, doc_list)

    event_list = utilities.stories_to_events(story_list, doc_list)

    with st.expander("Show geolocation step output", expanded=False):
        st.write(event_list)

    event_list = attribute_model.process(event_list, doc_list)
    with st.expander("Show attribute step output", expanded=False):
        st.write(event_list)

    event_list = actor_resolution_model.process(event_list)
    with st.expander("Show actor resolution step output", expanded=False):
        st.write(event_list)

    st.markdown("### Final output")
    cleaned_events = formatter.process(event_list, return_raw=True)
    st.write(cleaned_events)
