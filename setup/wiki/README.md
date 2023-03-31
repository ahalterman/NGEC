## Wikipedia in Elasticsearch

The files in this repo will help you set up a reduced Wikipedia in Elasticsearch for easy querying.

After ingesting, each Wikipedia article will be stored in the following form:

- `title`: the title of the Wikipedia page (no underscores)
- `redirects': every page that redirects to this page
- `alternative_names`: alternative names for the article, identified from bold phrases in the first sentence 
- `categories`: the Wikipedia categories associated with this page
- `intro_para`: the cleaned text of the first paragraph of the article. All text after the intro paragraph is discarded for space reasons.
- `infobox`: if the article includes a side infobox, it will be stored here.
- `box_type`: articles can have different box formats, e.g. "legislature", "military unit", "settlement" 

## Setup

First, make sure that Redis is installed and running:

```
sudo apt-get install redis-server
```

## Running

1. If you're starting from scratch, run

```
bash create_index.sh
```

This will run all commands to:

- set up Elasticsearch
- download the English Wikipedia dump
- go through the Wikipedia dump and identify all redirects
- store those redirects in Redis for easy querying
- go through Wikipedia again,
  - parsing each article
  - looking up alternative names in Redis
  - loading the formatted article into Elasticsearch


## 2. Updating the index

To update the index with a new copy of Wikipedia (and adding in the changes from issue #26 and #29):

Once it's downloaded, you should be able to do the following:
1. Re-download the NGEC repo and put the redirect pickle file in `NGEC/setup/wiki`
2. delete the existing Wikipedia index, but don't destroy the entire Elasticsearch container (I did this, and then realized that I'd also nuked the Geonames index, which thankfully only takes about 30 minutes to rebuild)
3. delete the old Wikipedia file and re-download: `wget "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"`
4. Re-do the Elasticsearch mapping with the new wiki_mapping.json file that separates out alternative names and redirects (and changes the way the indexing is done): `curl -XPUT 'localhost:9200/wiki' -H 'Content-Type: application/json' -d @wiki_mapping.json`
5. SKIP the "build links" step: that's already taken care of with the downloaded file. This step is absurdly slow.
6. Load the redirects file into Redis: `python load_wiki_es.py load_redis`
7. Load Wikipedia into elasticsearch: `python load_wiki_es.py load_es`


**A few caveats**:

- it assumes you'll run Redis directly on the machine, not in a container. You may want to switch to running Redis in a container.
- it assumes you'll run Elasticsearch in a Docker container. You may instead want to run it directly, in which case you should remove the Docker step and make sure that it's running on the port that the script expects.
- the Python `requirements.txt` files has specific package version numbers. To prevent overriding existing package versions, you may want to set up a virtual enviroment to install into.

You may want to run each command separately in a terminal so you don't have to re-run everything if you encounter an error somewhere.

