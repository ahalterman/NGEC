# NGEC -- Next generation political event coder

This repository contains the code for the Next Generation Event Coder (NGEC), a Python library for creating custom event data.
It accompanies the working paper, "Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks".

```
@article{halterman_et_al2023creating,
  title={Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks},
  author={Andrew Halterman and Philip A. Schrodt and Andreas Beger and Benjamin E. Bagozzi and Grace I. Scarborough},
  journal={arXiv preprint arXiv:2304.01331},
  year={2023}
}
```

Currently, this processing pipeline only performs the following steps:

1. event classification
2. geolocation and event geoloction
3. the event attribute model (identifying actors, recipients, and locations)
4. the actor resolution model

The context detection and mode detection are currently outside of this repo.

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

- the [Mordecai3 geoparser](https://github.com/ahalterman/mordecai3)
- Wikipedia running in Elasticsearch (see code in the `setup` directory)
- [Geonames running in Elasticsearch](https://github.com/openeventdata/es-geonames/).

## Note on the models

Because of conditions imposed by our funder and the proprietary data used in the project, we cannot share the training data or the trained event, mode, and context models used to produce the POLECAT event dataset. However, we provide example code for training classifiers on your own data in the [setup](https://github.com/ahalterman/NGEC/tree/main/setup/train_classifiers) directory. We also provide demonstration pretrained models for the event categories used in the POLECAT dataset that draw on a corpus of pseudo-labeled synthetic news stories using an approach described in [Halterman (2023)](https://arxiv.org/abs/2303.16028). These classifiers are not as accurate as the ones used in the POLECAT dataset, but work pretty well and could easily be improved with additional training data. 

## Acknowledgements

This research was sponsored by the Political Instability Task Force (PITF). The PITF is funded by
the Central Intelligence Agency. The views expressed in this paper are the authors’ alone and do not
represent the views of the U.S. Government.