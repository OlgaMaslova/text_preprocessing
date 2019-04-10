import numpy as np
import pandas as pd
import unidecode
import re 
import os
import pickle
import itertools
import random
from sklearn.utils import shuffle
from nltk.stem.snowball import SpanishStemmer
from nltk.stem.porter import PorterStemmer
import spacy
from es_lemmatizer import lemmatize
from nltk.corpus import stopwords

nlp = spacy.load('es_core_news_sm')
nlp.add_pipe(lemmatize, after="tagger")

GENERIC_WORDS = ['bebida', 'queso', 'refresco']
CUSTOM_STOPWORDS = ['unid', 'unidades', 'aprox', 'ud']

def clean_vocab_label(text):
    text = text.lower()
    # remove commas and dots
    text = re.sub("(\.|,)", '', text)
    # remove sign
    text = clean_from_distributor_brand(text)
    # regex replace for all units
    text = preprocess_spanish_scrap(text)
    text = remplace_unit_bundle(text)    
    #remove stopwords
    text = remove_stopwords(text)    
    text = unidecode.unidecode(text)
    return text

def remplace_unit_bundle(word):
    word = unidecode.unidecode(word)        
    #unit_liquid
    word = re.sub(r"([0-9]+[.,]?)+[ ]{0,1}(litros|ml|cl|l)(\b)", ' <UNIT_LIQUID> ', word, flags=re.IGNORECASE)
    #unit_solid
    match_solid = re.findall(r"(?P<solid>([0-9]+[.,]?)+[ ]?(gramos|grs|gr|mg|kg|g))(?:\w{1,3}|\b)", word, flags=re.IGNORECASE)
    if match_solid is not None:
        for match in match_solid:
            # replace first element of match - it's a full match
            word = re.sub(match[0], ' <UNIT_SOLID> ', word, flags=re.IGNORECASE)
    #bundle
    word = re.sub(r"((\d{1,3})(x))|(x\d{1,3})", ' <BUNDLE> ', word, flags=re.IGNORECASE)
    #pack
    word = re.sub(r"(\d)+( )(pcs|pieces)", ' <PACK> ', word, flags=re.IGNORECASE)    
    #percent
    word = re.sub(r"(\d)+(%)|(\d)+( )+(%)", ' <PERCENTAGE> ', word, flags=re.IGNORECASE)
    #fraction
    word = re.sub(r"(\d+)(\/)(\d+)", ' <FRACTION> ', word, flags=re.IGNORECASE)
    #number
    word = re.sub(r"(\d)+", ' <NUMBER> ', word, flags=re.IGNORECASE)
    #remove puntuation (except for commas and points)
    word = word = re.sub(r"[^\w\s\.\,<>\/]", '', word, flags=re.IGNORECASE)
    word = word.lower()
    return word

def preprocess_spanish_scrap(label):
    # package
    match_package = re.findall(r"(?P<package>(pack|caja|bolsa|bandeja|tarrina))(\s\d+)", label)
    if match_package is not None:
        for match in match_package:
            # replace first element of match - it's a full match
            label = re.sub(match[0], ' <PACK> ', label)
    return label

def remove_stopwords(label):
    label = ' '.join([word for word in label.split(' ') if not word in stopwords.words('spanish')])
    label = ' '.join([word for word in label.split(' ') if not word in CUSTOM_STOPWORDS])
    return label

def clean_from_distributor_brand(label):
    brands = ["auchan", "eroski", "carrefour", "carref", "alcampo"] 
    label = re.sub('|'.join(brands), '', label, flags=re.IGNORECASE)
    return label    

def stem_label(label):
    # stemming
    #print("stemming...")
    stemmer = PorterStemmer()
    stemword = lambda s: stemmer.stem(s)
    label = ' '.join([stemword(stemmed_token) for stemmed_token in label.split(' ')])
    #print(label)
    return label

def remove_generic_words(label):
    label = re.sub('|'.join(GENERIC_WORDS), '', label, flags=re.IGNORECASE)
    return label

def shuffle_words(label):
    # don't keep empty strin gas tokens when split
    shuffled_labels_list = list(filter(None, shuffle(label.split(' '))))
    shuffled_label = ' '.join(shuffled_labels_list)   
    return shuffled_label.strip()

def my_lemmatize(label):  
    #print("lemmatizing...")
    doc = nlp(label)
    lemmas = [token.lemma_ for token in doc]
    label = ' '.join(lemmas)
    #print(label)
    return label

def crop_words(label):
    #keep from 1 to 4 first words
    #print("cropping...")
    words = label.split(' ')
    nmb_words_to_keep = np.random.randint(1,5)
    words = words[:nmb_words_to_keep]      
    label = ' '.join(words)
    #print(label)
    return label

def modify_label(original_label):
    label = original_label
    # crop
    #print("original label: {}".format(label))
    to_crop = bool(random.getrandbits(1))
    label = crop_words(label) if to_crop else label

    # shuffle words 
    label = shuffle_words(label) if bool(random.getrandbits(1)) else label

    # lemmas
    to_lemmatize = bool(random.getrandbits(1))
    label = my_lemmatize(label) if to_lemmatize else label 
    
    # remove generic words
    label = remove_generic_words(label) if bool(random.getrandbits(1)) else label  

    label = unidecode.unidecode(label.strip())
    #check if we have word in final label
    match = re.findall(r'([a-zA-Z]+)', label)

    #print("final label: {}".format(label))
    return label if match else original_label

def augmentation_labels(original_series):
    modified_series = original_series.copy()
    modified_series = modified_series.apply(lambda x: modify_label(x))
    series = pd.concat([original_series,modified_series], ignore_index=True, sort=True)  
    series = series.drop_duplicates().reset_index(drop=True)
    print("Total labels in augmentation {}".format(series.count()))
    return series   

def reduce_labels(original_series):
    series = original_series.sample(2000)
    return series

if __name__ == "__main__":
    remove_generic_words("arefresco verde arizona miel botella  <unit_liquid> ")