# NGEC -- Next generation political event coder


Currently, this processing pipeline only performs the following steps:

1. geolocation and event geoloction
2. the event attribute model (identifying actors, recipients, and locations)
3. the actor resolution model

The event categorization, context detection, and mode detection are currently outside of this repo.

## Running

The main script is `ngec_process.py`.

```
python ngec_process.py

usage: ngec_process.py [-h] [-m -1] [-a NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457] [-b NGEC/assets/]
                       [-g ../mordecai3/mordecai_new.pt]
                       [input_file]

positional arguments:
  input_file            JSONL input file with events, modes, and contexts

options:
  -h, --help            show this help message and exit
  -m, --max-stories -1
                        Max stories to code
  -a, --attribute-dir NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457
                        Location of the QA attribute model
  -b, --base-path NGEC/assets/
                        Location of the other models and files
  -g, --geo-model ../mordecai3/mordecai_new.pt
                        Location of the geolocation model
```


<details>
  <summary>Click to view example input</summary>

```
{
  "id": "20190801-2227-8b13212ac6f6",
  "date": "2019-08-01",
  "event_type": [
    "ACCUSE",
    "REJECT",
    "THREATEN",
    "SANCTION"
  ],
  "event_mode": [],
  "event_text": "The Liberal Party, the largest opposition in Paraguay, .... ",
  "story_id": "EFESP00020190801ef8100001:50066618",
  "publisher": "translateme2-pt",
  "headline": "\nOposição confirma q...",
  "pub_date": "2019-08-01",
  "contexts": [
    "corruption"
  ],
  "version": "NGEC_coder-Vers001-b1-Run-001"
}
```
</details>


## Requirements 

First, create a new Conda environment and install the required libraries:

```
conda create -y --name ngec python=3.9 
conda activate ngec

pip install spacy
python -m spacy download en_core_web_trf
pip install textacy sentence-transformers
pip install elasticsearch elasticsearch_dsl unidecode dateparser
pip install jsonlines tqdm datasets rich plac 
```

The coder requires the following libraries and services to be available:

- Mordecai3, specifically the update from 12 March 2022 ([repo](https://github.com/ahalterman/mordecai3)
- Wikipedia running in Elasticsearch (see code in the `setup` directory)
- Geonames running in Elasticsearch ([repo](https://github.com/openeventdata/es-geonames/tree/es_7)).

## Acknowledgements

This research was sponsored by the Political Instability Task Force (PITF). The PITF is funded by
the Central Intelligence Agency. The views expressed in this paper are the authors’ alone and do not
represent the views of the U.S. Government.