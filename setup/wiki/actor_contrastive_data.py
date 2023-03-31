import os
import re
import pickle
from tqdm import tqdm
import pylcs
import random
from itertools import combinations
import itertools
import jsonlines

files = os.listdir()
versions = [int(re.findall("dict_(\d+)\.", i)[0]) for i in files if re.match("redirect_dict", i)]

with open(f"redirect_dict_{max(versions)}.0.pkl", "rb") as f:
    redirect_dict = pickle.load(f)

query = "Donald Trump"

keys = list(redirect_dict.keys())
sims = [pylcs.lcs_sequence_length(query, i) for i in tqdm(keys) if i]
#np.argmax(sims, K=5)
np.argpartition(sims, -4)[-4:]
## Build training data


redirect_list = list(redirect_dict.items())

item = redirect_list[33333]

skip_patt = re.compile(r"#|/")
disambig_patt = re.compile(r"\(disambiguation\)")

def clean_entries(entries):
    entries = [i for i in entries if not re.search(skip_patt, i)]
    entries = [re.sub(disambig_patt, "", i).strip() for i in entries]
    entries = list(set(entries))
    return entries

pos_data = []
pos_file = "redirect_sim_pos.jsonl"
for n, item in tqdm(enumerate(redirect_list), total=len(redirect_list)):
    entries = [item[0]] + item[1]
    entries = clean_entries(entries)
    if len(entries) > 50:
        entries = random.choices(entries, k=50)
    res = list(combinations(entries, 2))
    random.shuffle(res)
    pos_data.extend(res[:10])
    if n % 5000 == 0:
        #print(n)
        if pos_data:
            with jsonlines.open(pos_file, "a") as f:
                f.write_all(pos_data)
            pos_data = []
if pos_data:
    with jsonlines.open(pos_file, "a") as f:
        f.write_all(pos_data)


140156 * (4328876 / 18106)


neg_data = []
neg_file = "redirect_sim_neg.jsonl"
for n, item in tqdm(enumerate(redirect_list), total=len(redirect_list)):
    entries = [item[0]] + item[1]
    entries = clean_entries(entries)
    for entry in entries:
        other_item = random.choice(redirect_list)
        # avoid the vanishing rare chance of sampling the same entry
        if other_item[0] == item[0]:
            other_item = random.choice(redirect_list)
        other_entries = [other_item[0]] + other_item[1]
        other_entries = clean_entries(other_entries)
        if other_entries:
            neg_sample = random.choice(other_entries)
            neg_data.append((entry, neg_sample))
    if n % 5000 == 0:
        #print(n)
        if neg_data:
            with jsonlines.open(neg_file, "a") as f:
                f.write_all(neg_data)
            neg_data = []
if neg_data:
    with jsonlines.open(neg_file, "a") as f:
        f.write_all(neg_data)


## Wiki version

conn = setup_es()

box_types = ['officeholder', 'settlement',  'official post', 'company',
            'war faction', 'government agency', 'military unit', 'person',
            'aircraft begin', 'ship begin', 'weapon', 'military person',
            'politician', 'Minister', 'criminal', 'company']

'honorific-prefix'

box_type = box_types[0]
q = {"multi_match": {"query": box_type,
                                        "fields": ['box_type'],
                                        "type" : "phrase"}
                                        }
res = conn.query(q)[0:40].execute()
results = [i.to_dict()['_source'] for i in res['hits']['hits']] 

for page in results:


page = results[4]

def page_to_entries(page):
    entries = [page['title']] + page['redirects'] + page['alternative_names']
    if 'infobox' in page.keys():
        if 'name' in page['infobox'].keys():
            entries.append(page['infobox']['name'])
            if 'honorific-suffix' in page['infobox'].keys():
                nn = page['infobox']['name'] + " " + page['infobox']['honorific-suffix']
                entries.append(nn)
            if 'honorific-prefix' in page['infobox'].keys():
                nn = page['infobox']['honorific-prefix'] + " " + page['infobox']['name']
                entries.append(nn)
            if 'office' in page['infobox'].keys():
                nn = page['infobox']['office'] + " " + page['infobox']['name']
                entries.append(nn)
            if 'office1' in page['infobox'].keys():
                nn = page['infobox']['office1'] + " " + page['infobox']['name']
                entries.append(nn) 
            if 'rank' in page['infobox'].keys():
                nn = page['infobox']['rank'] + " " + page['infobox']['name']
                entries.append(nn) 
    entries = clean_entries(entries)
    return entries


def make_pos_combos(entries, max_pairs=30):
    # First, limit to 50 redirects
    if len(entries) > 50:
        entries = random.choices(entries, k=50)
    res = list(combinations(entries, 2))
    random.shuffle(res)
    return res[0:max_pairs]

def get_close_match(query, max_results=3, conn=conn):
    q = {"multi_match": {"query": query,
                             "fields": ['title', 'redirects'],
                        }}
    res = conn.query(q)[0:max_results].execute()
    results = [i.to_dict()['_source'] for i in res['hits']['hits']] 
    results = [i for i in results if i['title'] != query]
    return results

def get_neg_pairs(entries, page):
    other_names = []
    close_matches = get_close_match(page['title'], 5)
    for cm in close_matches:
        nes = page_to_entries(cm)
        other_names.extend(nes)
    
    samp_size = min(len(entries), 2)
    if samp_size == 0:
        return []
    neg_pairs = []
    for n in other_names:
        es = random.sample(entries, samp_size)
        for e in es:
            neg_pairs.append((e, n))
    return neg_pairs

all_pos = []
all_neg = []
neg_file = "redirect_sim_neg2.jsonl"
pos_file = "redirect_sim_pos2.jsonl"

box_types = ['officeholder', 'settlement',  'official post', 'company',
            'war faction', 'government agency', 'military unit', 'person',
            'aircraft begin', 'ship begin', 'weapon', 'military person',
            'politician', 'Minister', 'criminal', 'company', 'infobox company',
            'country', 'geopolitical organization']


for box_type in box_types[]:
    q = {"multi_match": {"query": box_type,
                                            "fields": ['box_type'],
                                            "type" : "phrase"}
                                            }
    res = conn.query(q)[0:10000]

    for i in tqdm(res):
        page = i.to_dict() 
        entries = page_to_entries(page)
        pos = make_pos_combos(entries)
        neg = get_neg_pairs(entries, page)
    
        all_pos.extend(pos)
        all_neg.extend(neg)
    
        if len(all_pos) > 5000:
            with jsonlines.open(neg_file, "a") as f:
                f.write_all(all_neg)
            with jsonlines.open(pos_file, "a") as f:
                f.write_all(all_pos)
            all_pos = []
            all_neg = []

#para_terms = ['officer', 'politician', 'diplomat', 'country', 'province', 'municipalities', 'city', 'municipality',  'non-governmental organization']#
para_terms = ["Arab", "Andorra","United Arab Emirates","Afghanistan","Antigua and Barbuda","Anguilla","Albania","Armenia","Angola","Argentina","American Samoa","Austria","Australia","Aruba","Azerbaijan","Bosnia and Herzegovina","Barbados","Bangladesh","Belgium","Burkina Faso","Bulgaria","Bahrain","Burundi","Benin","Saint Barthélemy","Bermuda","Brunei","Bolivia","Brazil","Bahamas","Bhutan","Botswana","Belarus","Belize","Canada","Cocos [Keeling] Islands","Democratic Republic of the Congo","Central African Republic","Congo","Switzerland","Côte d’Ivoire","Cook Islands","Chile","Cameroon","China","Colombia","Costa Rica","Cuba","Cape Verde","Curaçao","Christmas Island","Cyprus","Czech Republic","Germany","Djibouti","Denmark","Dominican Republic","Algeria","Ecuador","Estonia","Egypt","Western Sahara","Eritrea","Spain","Ethiopia","European Union","Finland","Fiji","Falkland Islands","Micronesia","Faroe Islands","France","Gabon","United Kingdom","Grenada","Georgia","Ghana","Gibraltar","Greenland","Gambia","Guinea","Guadeloupe","Equatorial Guinea","Greece","Guatemala","Guam","Guinea-Bissau","Guyana","Hong Kong SAR China","Heard Island and McDonald Islands","Honduras","Croatia","Haiti","Hungary","Indonesia","Ireland","Israel","Isle of Man","India","Iraq","Iran","Iceland","Italy","Jersey","Jamaica","Jordan","Japan","Kenya","Kosovo","Kyrgyzstan","Cambodia","Kiribati","Comoros","Saint Kitts and Nevis","North Korea","South Korea","Kuwait","Cayman Islands","Kazakhstan","Laos","Lebanon","Saint Lucia","Liechtenstein","Sri Lanka","Liberia","Lesotho","Lithuania","Luxembourg","Latvia","Libya","Morocco","Monaco","Moldova","Montenegro","Saint Martin","Madagascar","Marshall Islands","Macedonia","Mali","Myanmar [Burma]","Mongolia","Mauritania","Montserrat","Malta","Mauritius","Maldives","Malawi","Mexico","Malaysia","Mozambique","Namibia","New Caledonia","Niger","Norfolk Island","Nigeria","Nicaragua","Netherlands","Norway","Nepal","Nauru","Niue","New Zealand","Oman","Panama","Peru","French Polynesia","Papua New Guinea","Philippines","Pakistan","Poland","Pitcairn Islands","Puerto Rico","Palestinian Territories","Portugal","Palau","Paraguay","Qatar","Romania","Serbia","Russia","Rwanda","Saudi Arabia","Solomon Islands","Seychelles","Sudan","Sweden","Singapore","Saint Helena","Slovenia","Slovakia","Sierra Leone","San Marino","Senegal","Somalia","Suriname","São Tomé and Príncipe","El Salvador","Syria","Swaziland","Turks and Caicos Islands","Chad","Togo","Thailand","Tajikistan","Tokelau","Timor-Leste","Turkmenistan","Tunisia","Tonga","Turkey","Trinidad and Tobago","Tuvalu","Taiwan","Tanzania","Ukraine","Uganda","United Nations","United States","Uruguay","Uzbekistan","Vatican City","Saint Vincent and the Grenadines","Venezuela","British Virgin Islands","U.S. Virgin Islands","Vietnam","Vanuatu","Wallis and Futuna","Samoa","Yemen","South Africa","Zambia","Zimbabwe"]
for para_term in tqdm(para_terms):
    q = {"multi_match": {"query": para_term,
                                            "fields": ['intro_para']
                                            }}
    res = conn.query(q)[0:10000]
    for i in tqdm(res, leave=False, total=10000):
        page = i.to_dict() 
        entries = page_to_entries(page)
        pos = make_pos_combos(entries)
        neg = get_neg_pairs(entries, page)
    
        all_pos.extend(pos)
        all_neg.extend(neg)
    
        if len(all_pos) > 5000:
            with jsonlines.open(neg_file, "a") as f:
                f.write_all(all_neg)
            with jsonlines.open(pos_file, "a") as f:
                f.write_all(all_pos)
            all_pos = []
            all_neg = []

with jsonlines.open(neg_file, "a") as f:
    f.write_all(all_neg)
with jsonlines.open(pos_file, "a") as f:
    f.write_all(all_pos)

    q = {"multi_match": {"query": "officer",
                                            "fields": ['intro_para'],
                                            }}