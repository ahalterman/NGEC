import numpy as np
from rich import print
import jsonlines
import numpy as np
import pandas as pd
import os
import dateparser
from datetime import datetime
import re

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# silence dateparser warning. https://github.com/scrapinghub/dateparser/issues/1013
import warnings
warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


def country_name_dict(base_path):
    file = os.path.join(base_path, "countries.csv")
    countries = pd.read_csv(file)
    country_name_dict = {i:j for i, j in zip(countries['CCA3'], countries['Name'])}
    country_name_dict.update({"": ""})
    country_name_dict.update({"IGO": "Intergovernmental Organization"})
    return country_name_dict

def resolve_date(event):
    """
    Create a new 'date_resolved' key with a date in YYYY-MM-DD format

    TODO:
    include granularity details (e.g. month, year.)?
    >>> DateDataParser().get_date_data('March 2015')
    DateData(date_obj=datetime.datetime(2015, 3, 16, 0, 0), period='month', locale='en')
    """
    if 'DATE' not in event['attributes'].keys():
        pub_date = dateparser.parse(event['pub_date']).strftime("%Y-%m-%d")
        event['date_resolved'] = pub_date
        event['date_raw'] = "No date detected--using publication date"
        return event
    if not event['attributes']['DATE']:
        pub_date = dateparser.parse(event['pub_date']).strftime("%Y-%m-%d")
        event['date_resolved'] = pub_date
        event['date_raw'] = "<No date detected--using publication date>"
        return event
    
    base_date = dateparser.parse(event['pub_date'])
    raw_date = event['attributes']['DATE'][0]['text']
    print(f"raw_date: {raw_date}")
    
    resolved_date = dateparser.parse(date_string=raw_date, settings={'RELATIVE_BASE': base_date,
                                                                    'PREFER_DATES_FROM': "past"})
    if not resolved_date:
        if re.search("next|later", raw_date):
            raw_date = re.sub(r"next|later", "", raw_date).strip()
            resolved_date = dateparser.parse(date_string=raw_date, settings={'RELATIVE_BASE': base_date,
                                                                    'PREFER_DATES_FROM': "future"})
            if resolved_date:
                event['date_resolved'] = resolved_date.strftime("%Y-%m-%d")
                event['date_raw'] = raw_date
                return event
    if not resolved_date:
        event['date_resolved'] = event['pub_date']
        event['date_raw'] = "<dateparser failed to convert relative date--using pub date>"
        return event
    else:
        event['date_resolved'] = resolved_date.strftime("%Y-%m-%d")
        event['date_raw'] = raw_date
        return event


class Formatter:
    def __init__(self, quiet=False, base_path="assets", geolocation_threshold=0.85):
        self.quiet = quiet
        self.base_path = base_path
        self.iso_to_name = country_name_dict(self.base_path)
        self.geo_threshold = geolocation_threshold

    """
    event = {   'attributes': {   'ACTOR': [{   'qa_end_char': 53,
                                   'qa_score': 0.31743326783180237,
                                   'qa_start_char': 39,
                                   'text': 'Nicolas Maduro',
                                   'score': 0.23675884306430817,
                                   'wiki': 'Nicolás Maduro',
                                   'country': 'VEN',
                                   'code_1': 'ELI',
                                   'code_2': ''}],
                      'LOC': [{   'qa_end_char': 156,
                                 'qa_score': 0.4355418384075165,
                                 'qa_start_char': 148,
                                 'text': 'Barbados'}],
                      'RECIP': [{   'qa_end_char': 90,
                                   'qa_score': 0.1324695497751236,
                                   'qa_start_char': 79,
                                   'score': 0.13248120248317719,
                                   'wiki': 'Juan Guaidó',
                                   'country': 'VEN',
                                   'code_1': 'REB',
                                   'code_2': '',
                                   'text': 'Juan Guaidó'}]},
    'contexts': ['pro_democracy'],
    'date': '2019-08-01',
    'event_geolocation': {   'admin1_code': '00',
                             'admin1_name': '',
                             'admin2_code': '',
                             'admin2_name': '',
                             'country_code3': 'BRB',
                             'end_char': 156,
                             'event_location_overlap_score': 1.0,
                             'feature_class': 'A',
                             'feature_code': 'PCLI',
                             'geonameid': '3374084',
                             'lat': 13.16453,
                             'lon': -59.55165,
                             'resolved_placename': 'Barbados',
                             'score': 1.0,
                             'search_placename': 'Barbados',
                             'start_char': 148},
    'event_mode': [],
    'event_text': 'Delegates of the Venezuelan president, Nicolas Maduro, and '
                  'the leader objector Juan Guaidó resumed on Wednesday (31) '
                  'conversations on the island of Barbados, sponsored by '
                  'Norway, to seek a way out of the crisis in their country, '
                  'announced the parties. "We started another round of '
                  'sanctions under the mechanism of Oslo," indicated on '
                  'Twitter Mr Stalin González, one of the envoys of Guaidó, '
                  'parliamentary leader recognized as interim president by '
                  'half hundred countries. The vice-president of Venezuela, '
                  'Delcy Rodríguez, confirmed in a press conference that '
                  'representatives of mature traveled to Barbados for the '
                  'meetings with the opposition. Mature reaffirmed in a '
                  'message to the nation that the government seeks to '
                  'establish a "bureau for permanent dialog with the '
                  'opposition, and called entrepreneurs and social movements '
                  'to be added to the process. After exploratory '
                  'approximations and a first face to face in Oslo in mid-May, '
                  'the parties have transferred the dialog on 8 July for the '
                  'caribbean island. The opposition search in the negotiations '
                  'the output of mature and a new election, by considering '
                  'that his second term, started last January, resulted from '
                  'fraudulent elections, not recognized by almost 60 '
                  'countries, among them the United States. ',
    'event_type': 'RETREAT',
    'geolocated_ents': [   {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'BRB',
                               'end_char': 156,
                               'event_location_overlap_score': 1.0,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3374084',
                               'lat': 13.16453,
                               'lon': -59.55165,
                               'resolved_placename': 'Barbados',
                               'score': 1.0,
                               'search_placename': 'Barbados',
                               'start_char': 148},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'NOR',
                               'end_char': 177,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3144096',
                               'lat': 62.0,
                               'lon': 10.0,
                               'resolved_placename': 'Kingdom of Norway',
                               'score': 1.0,
                               'search_placename': 'Norway',
                               'start_char': 171},
                           {   'admin1_code': '12',
                               'admin1_name': 'Oslo',
                               'admin2_code': '0301',
                               'admin2_name': 'Oslo',
                               'country_code3': 'NOR',
                               'end_char': 318,
                               'feature_class': 'P',
                               'feature_code': 'PPLC',
                               'geonameid': '3143244',
                               'lat': 59.91273,
                               'lon': 10.74609,
                               'resolved_placename': 'Oslo',
                               'score': 1.0,
                               'search_placename': 'Oslo',
                               'start_char': 314},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'VEN',
                               'end_char': 502,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3625428',
                               'lat': 8.0,
                               'lon': -66.0,
                               'resolved_placename': 'Bolivarian Republic of '
                                                     'Venezuela',
                               'score': 1.0,
                               'search_placename': 'Venezuela',
                               'start_char': 493},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'BRB',
                               'end_char': 604,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '3374084',
                               'lat': 13.16453,
                               'lon': -59.55165,
                               'resolved_placename': 'Barbados',
                               'score': 1.0,
                               'search_placename': 'Barbados',
                               'start_char': 596},
                           {   'admin1_code': '12',
                               'admin1_name': 'Oslo',
                               'admin2_code': '0301',
                               'admin2_name': 'Oslo',
                               'country_code3': 'NOR',
                               'end_char': 918,
                               'feature_class': 'P',
                               'feature_code': 'PPLC',
                               'geonameid': '3143244',
                               'lat': 59.91273,
                               'lon': 10.74609,
                               'resolved_placename': 'Oslo',
                               'score': 1.0,
                               'search_placename': 'Oslo',
                               'start_char': 914},
                           {   'admin1_code': '00',
                               'admin1_name': '',
                               'admin2_code': '',
                               'admin2_name': '',
                               'country_code3': 'USA',
                               'end_char': 1259,
                               'feature_class': 'A',
                               'feature_code': 'PCLI',
                               'geonameid': '6252001',
                               'lat': 39.76,
                               'lon': -98.5,
                               'resolved_placename': 'United States',
                               'score': 1.0,
                               'search_placename': 'United States',
                               'start_char': 1239}],
    'headline': 'Governo e oposição da Venezuela retomam diálogo em Barbados\n',
    'id': '20190801-2309-4e081644904c_COOPERATE_R',
    'pub_date': '2019-08-01',
    'publisher': 'translateme2-pt',
    'story_id': 'AFPPT00020190801ef81000jh:50066619',
    'story_people': ['Delcy Rodríguez', 'Guaidó', 'Nicolas Maduro', 'Stalin González', 'Juan Guaidó'],
    'story_orgs': ['Mature'],
    'story_locs': ['Norway', 'United States', 'Barbados', 'Oslo', 'Venezuela'],
    'version': 'NGEC_coder-Vers001-b1-Run-001'}
    """

    def find_event_loc(self, event, geo_overlap_thresh=0.5):
        if 'LOC' not in event['attributes'].keys():
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        try:
            event_loc_raw = event['attributes']['LOC'][0] ## NOTE!! Assuming just one location from the QA model
        except IndexError:
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        if not event_loc_raw:
            event['event_geolocation'] = {"reason": "No LOC attribute found by the QA/attribute model",
                                          "geo": None}
            return event
        if 'geolocated_ents' not in event.keys():
            event['event_geolocation'] = {"reason": "No story locations were geolocated (Missing 'geolocated_ents' key).",
                                          "geo": None}
            return event
        event_loc_chars = set(range(event_loc_raw['qa_start_char'], event_loc_raw['qa_end_char']))
        geo_ent_ranges = [set(range(i['start_char'], i['end_char'])) for i in event['geolocated_ents']]
        # calculate intersection-over-union/Jaccard
        ious = np.array([len(event_loc_chars.intersection(i)) / len(event_loc_chars.union(i)) for i in geo_ent_ranges])
        if len(ious) == 0:
            event['event_geolocation'] = {"reason": f"No geolocated entities",
                                              "geo": None}
            return event
        try:
            if np.max(ious) < geo_overlap_thresh:
                event['event_geolocation'] = {"reason": f"Attribute placename ({event_loc_raw['text']}) [doesn't overlap enough with any placenames: {str(np.max(ious))}",
                                              "geo": None}
                return event
        except ValueError:
            event['event_geolocation'] = {"reason": f"Problem with intersection-overlap vector. No elements?",
                                              "geo": None}
            return event
        best_match = event['geolocated_ents'][np.argmax(ious)]
        if not best_match:
            event['event_geolocation'] = {"reason": f"No 'best_match' geolocated entity",
                                              "geo": None}
            return event
        best_match['event_location_overlap_score'] = float(np.max(ious))
        if 'score' not in best_match.keys():
            event['event_geolocation'] = {"reason": f"'best_match' identified but no 'score' key. Returning best_match anyway",
                                        "geo": best_match} 
            return event
        if best_match['score'] > self.geo_threshold:
            event['event_geolocation'] = {"reason": f": Successful overlap between attribute placename and one of the geoparser results",
                                        "geo": best_match}
            return event
        else:
            event['event_geolocation'] = {"reason": f": Successful overlap between attribute placename and one of the geoparser results BUT geoparser score was too low ({best_match['score']})",
                                        "geo": None}
            return event




    def add_meta(self, event):
        """
        Add optional metadata to the event dictionary (e.g. alternative country codes, country names,
        event intensity, event quad class, etc.)
        """
        for k, att in event['attributes'].items():
            # add stuff to actors and recipients
            if k in ["LOC", "DATE"]:
                continue
            for v in att:
                try:
                    v['country_name'] = self.iso_to_name[v['country']]
                except:
                    print(v['country'])
                    v['country_name'] = ""

        return event


    def process(self, event_list, return_raw=False):
        """
        Create and write out a final cleaned dictionary/JSON file of events.

        Parameters
        ----------
        event_list: list of dicts
          list of events after being passed through each of the processing steps
        return_raw: bool
          If true, don't write to a final and instead return the final version. Useful for 
          debugging. Defaults to False.
        """
        for n, event in enumerate(event_list):
            #if n == 0:
            #    print(e)
            event = self.find_event_loc(event)
            event = self.add_meta(event)
            try:
                event = resolve_date(event)
            except Exception as exception:
                logger.warning(f"{exception} parsing date for event number {n}")
        if return_raw:
            return event_list
        else:
            with jsonlines.open("events_processed.jsonl", "w") as f:
                f.write_all(event_list)

