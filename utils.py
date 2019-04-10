import pandas as pd
import os
import re 
import unidecode
from langdetect import detect
import nltk
import random
import preprocess_scrap as augm
#nltk.download('stopwords')



def read_and_detect_lang():
    df_lang = pd.read_csv('experiments/all_brand_language_count.csv', header=0)
    df_gs1 = pd.read_csv('experiments/all_brands.csv', header=0)

    df_lang['gs1_code'] = df_gs1['gs1_code'].where(df_lang['name'] == df_gs1['name'])
    """
    df_language =  pd.read_csv('experiments/all_brand_language.csv', header=0)
    df = df[df['count'] > 7]
    df['language'] = df_language['name'].apply(lambda x: detect_land(x))
    """
    df.to_csv("all_brand_language_count.csv")

def detect_land(x):
    lang = detect(x)
    print("{0} - language is {1}".format(x,lang))
    return lang

def get_difference_between_csv():
    #what we want to remove
    dir_path1 = "../../data/scrapped/products_spain/alcampo/prepared/shelf_chicken.csv"
    #from where we want to remove 1
    dir_path2 = "../../data/scrapped/products_spain/alcampo/prepared/fresh_prepared.csv"
    
    df1 = pd.read_csv(dir_path1, header=0)
    df2 = pd.read_csv(dir_path2, header=0)
    print(df2.info())
    # find elements in df2 that are not in df1
    df2not_in1 = df2[~(df2['label'].isin(df1['label']))].reset_index(drop=True)
    print(df2not_in1.info())
    df2not_in1.to_csv("../../data/scrapped/products_spain/alcampo/prepared/fresh_prepared.csv", index=False, columns=['web-scraper-order','web-scraper-start-url','label'])