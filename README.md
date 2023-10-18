# NGEC -- Next generation political event coder

This repository contains the code for the Next Generation Event Coder (NGEC), a
Python library for extracting event data from news text. The pipeline works out-of-the-box
to code events using the [PLOVER event ontology](https://osf.io/preprints/socarxiv/rm5dw/), but can 
be easily customized to produce events with a custom ontology.

It accompanies the working paper, ["Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks"](https://arxiv.org/pdf/2304.01331.pdf).

## Overview

We break the problem of event extraction into six steps:

1. Event classification: identify the event described in a document (e.g., PROTEST, ASSAULT, AGREE,...) using a transformer classifier trained on new data.
2. Sub-event (``mode'') classification: identify a more specific event type (e.g., PROTEST-riot, ASSAULT-aerial), also using a transformer-based classifier.
3. Context classification: identify themes or topics in a document (e.g., "human rights", "environment") using a classifier.
4. Event attribute identification: identifying the spans of text that report who carried out the event, who it was directed against, where it occurred, etc. We do this with a fine-tuned question-answering model trained on newly annotated text.
5. Actor, location, and date resolution: we resolve extracted named actors and recipients to their Wikipedia page using an offline Wikipedia index and a custom neural similarity model.
6. Entity categorization: Finally, we map the actor to their country and their "sector" code as defined by the PLOVER ontology (e.g., "GOV", "MIL", etc.)

![](docs/pipeline_figure.png)

Currently, this processing pipeline only performs the following steps:

*Note*: This repo has basic pretrained models for event detection, but does *not*
currently include context and mode models.

## Running

The main script is `ngec_process.py`.

```
python ngec_process.py

usage: ngec_process.py [-h] [-m -1] [-a NGEC/assets/PROP-SQuAD-trained-tinybert-6l-768d-squad2220302-1457] [-b NGEC/assets/]
                       [-g ../mordecai3/mordecai_new.pt]
                       [input_file]

positional arguments:
  input_file            JSONL input file. At a minimum, this should have keys for "id", "date", and
                        "event_text"

options:
  -h, --help            show this help message and exit
  -m, --max-stories -1
                        Max stories to code. -1 is all stories
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

## Citing

The steps that this pipeline implements are described in more detail in the [paper](https://arxiv.org/pdf/2304.01331.pdf). If you use the pipeline or the techniques we introduce, please cite the following:

```
@article{halterman_et_al2023creating,
  title={Creating Custom Event Data Without Dictionaries: A Bag-of-Tricks},
  author={Andrew Halterman and Philip A. Schrodt and Andreas Beger and Benjamin E. Bagozzi and Grace I. Scarborough},
  journal={arXiv preprint arXiv:2304.01331},
  year={2023}
}
```

## Acknowledgements

This research was sponsored by the Political Instability Task Force (PITF). The PITF is funded by
the Central Intelligence Agency. The views expressed in this paper are the authors’ alone and do not
represent the views of the U.S. Government.