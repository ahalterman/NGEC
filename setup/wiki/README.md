## Wikipedia in Elasticsearch

The files in this repo will help you set up a reduced Wikipedia in Elasticsearch for easy querying.

After ingesting, each Wikipedia article will be stored in the following form:

- `title`: the title of the Wikipedia page (no underscores)
- `redirects`: every page that redirects to this page
- `alternative_names`: alternative names for the article, identified from bold phrases in the first sentence 
- `short_desc`: Wikipedia's "short description" of the article
- `categories`: the Wikipedia categories associated with this page
- `intro_para`: the cleaned text of the first paragraph of the article. All text after the intro paragraph is discarded for space reasons.
- `infobox`: if the article includes a side infobox, it will be stored here.
- `box_type`: articles can have different box formats, e.g. "legislature", "military unit", "settlement" 
- `affiliated_people`: the contents of the 'leaders', 'founded_by', or 'founder' fields if present in the infobox. (This ends up not being used)

## Setup

First, make sure that Redis is installed and running:

```
sudo apt-get install redis-server
```

Alternatively, you can use a Docker container:

```bash
docker run -d -p 6379:6379 --name redis redis
```

Then, install the Python requirements:

```bash
pip install -r requirements.txt
```


## Running

**NOTE**: If you want, you can skip the first step (building the Wiki redirect file) by downloading a prebuilt pickle of the redirect dictionary from Google Drive. This is by far the slowest step (it can take up to 24 hours on a slow machine), but there are security concerns about downloading pickle files from the internet. If you're comfortable with that, you can download the file from [here](https://drive.google.com/file/d/1zJviHKAm0bQH9xaq5p-dUrVnknDlFgJK/view?usp=sharing) and place it in the `setup` directory.

To run the entire process, you can run the following command:

```bash
bash create_index.sh
```

This will run all commands to:

- set up Elasticsearch
- download the English Wikipedia dump
- go through the Wikipedia dump and identify all redirects (see note above--this is slow)
- store those redirects in Redis for easy querying
- go through Wikipedia again,
  - parsing each article
  - looking up alternative names in Redis
  - loading the formatted article into Elasticsearch

Alternatively, you can run each command in the bash file separately in a terminal so you don't have to re-run everything if you encounter an error somewhere.

## 2. Updating the index

To update the index with a new copy of Wikipedia, you should be able to do the following:

1. delete the existing Wikipedia index, but don't destroy the entire Elasticsearch container (I did this, and then realized that I'd also nuked the Geonames index, which thankfully only takes about 30 minutes to rebuild)
2. delete the old Wikipedia file and re-download: `wget "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"`
3. Re-do the Elasticsearch mapping with the new wiki_mapping.json file that separates out alternative names and redirects (and changes the way the indexing is done): `curl -XPUT 'localhost:9200/wiki' -H 'Content-Type: application/json' -d @wiki_mapping.json`
4. SKIP the "build links" step: that's already taken care of with the downloaded file. This step is absurdly slow.
5. Load the redirects file into Redis: `python load_wiki_es.py load_redis`
6. Load Wikipedia into elasticsearch: `python load_wiki_es.py load_es`


**A few caveats**:

- it assumes you'll run Redis directly on the machine, not in a container. You may want to switch to running Redis in a container.
- it assumes you'll run Elasticsearch in a Docker container. You may instead want to run it directly, in which case you should remove the Docker step and make sure that it's running on the port that the script expects.
- the Python `requirements.txt` files has specific package version numbers. To prevent overriding existing package versions, you may want to set up a virtual enviroment to install into.


