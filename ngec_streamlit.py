from NGEC import EventClass
from NGEC import  AttributeModel
from NGEC import ActorResolver
from NGEC import GeolocationModel
from NGEC import Formatter
from NGEC import utilities

import streamlit as st

import spacy
import pandas as pd

# stuff that's just used to allow streamlit cacheing
import preshed
import cymem
import spacy_transformers
import thinc

st.markdown("## NGEC test interface")

st.markdown("Put in some story text to see what NGEC produces.")
st.markdown("The event classifier step uses the open source models that are trained on synthetic documents. The accuracy is not as good as the proprietary models used to produce the POLECAT dataset. To manually override the event classification, set the event type (and mode) on the sidebar.")
st.markdown("Intermediate output is also returned but hidden by default.")

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_nlp():
    utilities.spacy_doc_setup()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

nlp = load_nlp()

def format_output(cleaned_events):
    for event in cleaned_events:
        if 'ACTOR' in event['attributes'].keys() and event['attributes']['ACTOR']:
            actors = '; '.join([i['text'] for i in event['attributes']['ACTOR']])
            actor_codes = '; '.join([f"{i['country']} {i['code_1']}" for i in event['attributes']['ACTOR']])
            actor_wikis = '; '.join([i['wiki'] for i in event['attributes']['ACTOR']])
        else:
            actors = ""
            actor_codes = ""
            actor_wikis = ""
        if 'RECIP' in event['attributes'].keys() and event['attributes']['RECIP']:
            recipients = '; '.join([i['text'] for i in event['attributes']['RECIP']])
            recipient_codes = '; '.join([f"{i['country']} {i['code_1']}" for i in event['attributes']['RECIP']])
            recip_wikis = '; '.join([i['wiki'] for i in event['attributes']['RECIP']])
        else:
            recipients = ""
            recipient_codes = ""
            recip_wikis = ""
        if event['event_geolocation']['geo']:
            resolved_placename = event['event_geolocation']['geo']['resolved_placename']
            adm1 = event['event_geolocation']['geo']['admin1_name']
            country = event['event_geolocation']['geo']['country_name']
        else:
            resolved_placename = ""
            adm1 = ""
            country = ""
        #st.success(actors)
        d = {"Raw Actors": actors,
                "Actor Codes": actor_codes,
                "Actor Wikis": actor_wikis,
                "Event Type": event['event_type'],
                "Event Mode": event['event_mode'],
                "Raw Recipients": recipients,
                "Recipient Codes": recipient_codes,
                "Recipient Wikis": recip_wikis,
                "Resolved Placename": resolved_placename,
                "Admin1": adm1,
                "Country": country,
                "Date": event['date_resolved']}
        df = pd.DataFrame(d, index=[0]).transpose()
        df = df.reset_index()
        df.columns = ["Attribute", "Value"]
        # disable row numbers
        df.index = [""] * len(df)
        st.table(df)



save_intermediate=False
attribute_dir="NGEC/assets/deberta_squadnewsqa_2023-05-22"
base_path="./NGEC/assets/"
save_intermediate=False
expand_actors=True
geo_model="/home/andy/projects/mordecai/mordecai3/assets/mordecai_2023-02-07_good.pt"
geo_path="/home/andy/projects/mordecai/mordecai3/assets/"

gpu=True

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_event_class():
    event_model = EventClass()
    return event_model

pub_date = st.sidebar.text_input("Publication date", "today")
event_type = st.sidebar.text_input("Event type", "")
event_mode = st.sidebar.text_input("Mode type", "")
show_intermediate = st.sidebar.checkbox("Show intermediate output", False)
event_model = load_event_class()

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_geo(save_intermediate=save_intermediate):
    geolocation_model = GeolocationModel(geo_model, 
                                         geo_path=geo_path,
                                         save_intermediate=save_intermediate)
    return geolocation_model

#@st.cache(allow_output_mutation = True)
@st.cache_resource()
def load_attr(attribute_dir=attribute_dir, silent=True, gpu=gpu, save_intermediate=save_intermediate, expand_actors=expand_actors,
             base_path=base_path):
    attribute_model = AttributeModel(attribute_dir,
                                    silent=silent,
                                    gpu=gpu,
                                    save_intermediate=save_intermediate,
                                    base_path=base_path,
                                    expand_actors=expand_actors)
    return attribute_model


@st.cache_resource()
def load_resolution(nlp=nlp, base_path=base_path, save_intermediate=save_intermediate, gpu=gpu):
    actor_resolution_model = ActorResolver(spacy_model=nlp, base_path=base_path, save_intermediate=save_intermediate, gpu=gpu)
    return actor_resolution_model

@st.cache_resource()
def load_formatter(base_path=base_path):
    formatter = Formatter(base_path=base_path)
    return formatter

geolocation_model = load_geo()
attribute_model = load_attr(base_path=base_path)
actor_resolution_model = load_resolution()
formatter = load_formatter()

text = st.text_area("Input text", "German troops withdrew from their area of operations in Kandahar last week.")




if text:
    doc_list = [nlp(text)]

    story_list = [{"event_text": text, "id": "123", "event_type": [event_type], "event_mode": [event_mode], "pub_date": pub_date}]
    
    if not event_type:
        story_list = event_model.process(story_list)
        if show_intermediate:
            with st.expander("Show event class step output", expanded=False):
                st.write(story_list)
    if not story_list[0]['event_type']:
        st.error("No event type detected.")
        st.stop()
    story_list = geolocation_model.process(story_list, doc_list)

    event_list = utilities.stories_to_events(story_list, doc_list)

    if show_intermediate:
        with st.expander("Show geolocation step output", expanded=False):
            st.write(event_list)

    event_list = attribute_model.process(event_list, doc_list)
    if show_intermediate:
        with st.expander("Show attribute step output", expanded=False):
            st.write(event_list)

    event_list = actor_resolution_model.process(event_list)
    if show_intermediate:
        with st.expander("Show actor resolution step output", expanded=False):
            st.write(event_list)

    st.markdown("### Final output")
    cleaned_events = formatter.process(event_list, return_raw=True)
    
    st.markdown(text)
    format_output(cleaned_events)

    with st.expander("Show raw final output", expanded=False):
        st.write(cleaned_events)
