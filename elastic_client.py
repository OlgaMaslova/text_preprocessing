from datetime import datetime
import json
from elasticsearch import Elasticsearch

client = None

def connect_elasticsearch():
    global client
    _es = None
    _es = Elasticsearch([{'host': '192.168.1.10', 'port': 9200}], http_auth=('aboutgoods', 'Vintage*'))
    if _es.ping():
        print('Yay Connected!')
    else:
        print('Cannot connect to ElasticSearch!')
    client = _es


def search(es_object, index_name, search):
    res = es_object.search(index=index_name, body=search)
    return res

def get_uuid_by_name(name):
    search_object = {'query': {'match': {'name': name}}}
    hits = get_hits(search(client, 'dm_ag_category', json.dumps(search_object)))    
    return hits[0]['_source']['code']

def get_hits(res):
    return res['hits']['hits']

if __name__ == '__main__':
    connect_elasticsearch()
    print(client)
    if client is not None:
        search_object = {'query': {'match': {'name': 'creamery'}}}
        res = get_hits(search(client, 'dm_ag_category', json.dumps(search_object)))
        print(res[0]['_source']['code'])