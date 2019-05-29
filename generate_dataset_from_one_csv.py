import pandas as pd
import os
import numpy as np
import re 
import unidecode
import random
import preprocess_scrap as augm

from preprocess_scrap import augmentation_labels, reduce_labels, clean_vocab_label

def read_scraped_csv(filepath):
    """
    Reads csv this all scrapped labels and their categories.
    Applies data augmentation if less than 1000 labels in a given category.
    Final csv with 'category_uid' and 'label' columns.
    """   
    # creating a DataFrame from csv 
    df_all_labels = pd.read_csv(filepath)
    locale='it'
    df_new_all_labels = pd.DataFrame([])
    #print(df_all_labels.info())
    df_all_labels['label']=df_all_labels['label'].apply(lambda x: clean_vocab_label(x, locale))   
    print("labels are cleaned")
    #labels per category
    categories = df_all_labels['ag_category_uid'].unique()
    for category in categories:
        print("Processing {} category".format(category))
        labels_per_category = df_all_labels[df_all_labels['ag_category_uid'] == category]
        labels_per_category.drop(columns=['ag_category_uid'], inplace=True)
        # too many labels, keep random 2000
        if labels_per_category.count()['label'] > 4000:
            print("reducing labels for category {}".format(category))
            reduced_labels = reduce_labels(labels_per_category['label'])
            labels_per_category = pd.DataFrame(reduced_labels) 

        if labels_per_category.count()['label'] < 1400:
            print("data augmentation for category {}".format(category))
            new_labels = augmentation_labels(labels_per_category['label'], locale)
            labels_per_category = pd.DataFrame(new_labels) 
        else:
            print("random modification for {}".format(category))
            # apply random modification on label (50% of crop, lemmatize, shuffle, remove generic)
            labels_per_category['label'] = labels_per_category['label'].apply(lambda x: augm.modify_label(x, locale) if bool(random.getrandbits(1)) else x)    
            #print(labels_per_category.info())   
        labels_per_category.insert(0,'category_uid', category)
        print(labels_per_category.head())   
        df_new_all_labels = pd.concat([df_new_all_labels, labels_per_category], sort=True)
        print(df_new_all_labels.info())
    df_new_all_labels.dropna(inplace=True)  
    df_new_all_labels.to_csv("../../data/scrapped/products_italy/all_labels_italy_processed_lemma_syn_290519.csv", index=False, columns=['category_uid','label'])

if __name__ == "__main__":
    read_scraped_csv("../../data/scrapped/products_italy/scrapped_products_italy.csv")
    