# Make sure you've installed Mordecai, which is in a separate package.
# E.g, pip install mordecai3
from mordecai3 import Geoparser
from pprint import pprint

# Create the Geoparser object
# Make sure the path to the model is correct
geo = Geoparser("NGEC/assets/mordecai_2023-03-28.pt")

output = geo.geoparse_doc("The Mexican government sent 300 National Guard troopers to bolster the southern state of Guerrero on Tuesday, where a local police chief and 12 officers were shot dead in a brutal ambush the day before.")

pprint(output)
#{'doc_text': 'The Mexican government sent 300 National Guard troopers to '
#             'bolster the southern state of Guerrero on Tuesday, where a local '
#             'police chief and 12 officers were shot dead in a brutal ambush '
#             'the day before.',
# 'event_location_raw': '',
# 'geolocated_ents': [{'admin1_code': '12',
#                      'admin1_name': 'Guerrero',
#                      'admin2_code': '',
#                      'admin2_name': '',
#                      'city_id': '',
#                      'city_name': '',
#                      'country_code3': 'MEX',
#                      'end_char': 97,
#                      'feature_class': 'A',
#                      'feature_code': 'ADM1',
#                      'geonameid': '3527213',
#                      'lat': 17.66667,
#                      'lon': -100.0,
#                      'name': 'Estado de Guerrero',
#                      'score': 1.0,
#                      'search_name': 'Guerrero',
#                      'start_char': 89}]}
