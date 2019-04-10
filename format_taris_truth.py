import pandas as pd
import os
import re 
import unidecode
import elastic_client as es_client

def extract_ground_truth():
    file_path="ground_truth_taris_3.csv"
    df = pd.read_csv(file_path, header=0, delimiter=";")
    products = df.products.tolist()
    all_labels = []
    all_cat = []
    for product in products:
        raw_data = product.split("],[")
        print(raw_data)
        raw_data_2 = [clean(x) for x in raw_data]    
        labels = [extract_words(x) for x in raw_data_2]
        categories = [extract_cat(x) for x in raw_data_2]
        all_labels += labels
        all_cat += categories
    
    ground_truth = pd.DataFrame({'label': all_labels, 'category_uid':all_cat})
    print(ground_truth.info())
    ground_truth.to_csv("ground_truth_es_3.csv" , index=False, columns=['label','category_uid'])

def clean(text):
    text = re.sub("\[", '', text)
    text = re.sub("\]", '', text)    
    return text

def extract_words(text):
    match = re.findall(r"([a-zA-Z]{2,})", text)
    label=''
    if match is not None:
        label=' '.join(match)
    return label

def drop_duplicates(file):
    df = pd.read_csv(file, header =0)
    print(df.info())
    df.drop_duplicates(inplace=True, subset='label')
    print(df.info())
    df.to_csv(file, index=False, columns=['label','category_uid'])

def set_uid(file):
    es_client.connect_elasticsearch()
    df_vt = pd.read_csv(file, header =0)
    df_category = pd.read_csv("../../data/scrapped/products_spain/category_name_id.csv", delimiter=";", header=0)
    cat_uid = dict()
    for _, row in df_category.iterrows():
        uid = es_client.get_uuid_by_name(row['category_name'])
        cat_uid[row['id']]=uid
    
    df_vt['category_uid'] = df_vt['category_uid'].apply(lambda x: cat_uid[x])
    df_vt.to_csv(file, index=False, columns=['label','category_uid'])


def extract_cat(text):
    match = re.findall(r"(?:(\"\,\"))(\d{1,2})", text)
    cat=''
    if match is not None:
        cat=match[0][1]     
    return cat

if __name__ == "__main__":
    #extract_ground_truth()
    #drop_duplicates("../word2vec-cnn/truth/es/ground_truth_es.csv")
    set_uid("../word2vec-cnn/truth/es/ground_truth_es.csv")
   
    
