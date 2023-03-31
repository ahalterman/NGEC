echo "Installing Redis..."
sudo apt-get install redis-server

echo "Starting Docker container and data volume..."
# create the directory first to avoid permission issues when Docker is running as root
mkdir $PWD/wiki_index/
docker run -d -p 127.0.0.1:9200:9200 -e "discovery.type=single-node" -v $PWD/wiki_index/:/usr/share/elasticsearch/data elasticsearch:7.10.1 

echo "Downloading Wikipedia..."
wget "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

echo "Creating mappings for the fields in the Wikipedia index..."
curl -XPUT 'localhost:9200/wiki' -H 'Content-Type: application/json' -d @wiki_mapping.json

echo "Change disk availability limits..."
curl -X PUT "localhost:9200/_cluster/settings" -H 'Content-Type: application/json' -d'
{
  "transient": {
    "cluster.routing.allocation.disk.watermark.low": "10gb",
    "cluster.routing.allocation.disk.watermark.high": "5gb",
    "cluster.routing.allocation.disk.watermark.flood_stage": "4gb",
    "cluster.info.update.interval": "1m"
  }
}
'

echo "\nBuilding redirect links..."
python load_wiki_es.py build_links

echo "\nBuilding links..."
python load_wiki_es.py load_redis

echo "\nLoading Wikipedia into Elasticsearch..."
python load_wiki_es.py load_es

echo "Done"
