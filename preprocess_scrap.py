import numpy as np
import pandas as pd
import unidecode
import re 
import os
import pickle
import itertools
import random
import json
from pattern.it import singularize, pluralize
from sklearn.utils import shuffle
from nltk.stem.snowball import SpanishStemmer
from nltk.stem.snowball import ItalianStemmer
from nltk.stem.porter import PorterStemmer
import spacy
from es_lemmatizer import lemmatize
from nltk.corpus import stopwords

nlp = spacy.load('it_core_news_sm') #python -m spacy download it_core_news_sm
nlp.add_pipe(lemmatize, after="tagger")

GENERIC_WORDS = ['bebida', 'queso', 'refresco']
CUSTOM_STOPWORDS = ['unid', 'unidades', 'aprox', 'ud']
BRANDS_ES=["auchan", "eroski", "carrefour", "carref", "alcampo"]
BRANDS_IT=['auchan', 'carrefour', 'bennet', 'tigota', 'coop', 'crf']

LOCALE_REGEX_PACK = {
        "it": r"(\d)+\s?(pz|pezzi)",
        "es": r"(\d)+( )(pcs|pieces)",
        "fr": r"(\d)+( )(pcs|pieces)"
    }    

def clean_vocab_label(text, locale):
    text = text.lower()
    # remove commas and dots
    text = re.sub("(\.|,)", '', text)
    # remove sign
    text = clean_from_distributor_brand(text, locale)
    # regex replace for all units
    if locale == 'es':
        text = preprocess_spanish_scrap(text)
    text = remplace_unit_bundle(text, locale)    
    #remove stopwords
    text = remove_stopwords(text, locale)    
    text = unidecode.unidecode(text)
    #print(text)
    return text

def remplace_unit_bundle(word, locale):
    word = unidecode.unidecode(word)        
    #unit_liquid
    word = re.sub(r"([0-9]+[.,]?)+[ ]{0,1}(litros|ml|cl|l)(\b)", ' <UNIT_LIQUID> ', word, flags=re.IGNORECASE)
    #unit_solid
    # match_solid_1 - format "number unit"
    # match_solid_2 - format "unit number"
    match_solid_1 = re.findall(r"(?P<solid>([0-9]+[.,]?)+[ ]?(gramos|grs|(gr(\s|$))|mg|kg|(g(\s|$)))(?:\w{1,3}|\b))", word, flags=re.IGNORECASE)
    match_solid_2 = re.findall(r"(?P<solid>(\b(gramos|grs|gr|mg|kg|g)(\.?)(\s)?[0-9]+[.,]?)+)", word, flags=re.IGNORECASE)
    if match_solid_1 is not None:
        for match in match_solid_1:
            # replace first element of match - it's a full match
            word = re.sub(match[0], ' <UNIT_SOLID> ', word, flags=re.IGNORECASE)
    if match_solid_2 is not None:
        for match in match_solid_2:
            word = re.sub(match[0], ' <UNIT_SOLID> ', word, flags=re.IGNORECASE)
    #bundle
    word = re.sub(r"((\d{1,3})\s?(x))|(x\s?\d{1,3})", ' <BUNDLE> ', word, flags=re.IGNORECASE)
    #pack
    word = re.sub(LOCALE_REGEX_PACK[locale], ' <PACK> ', word, flags=re.IGNORECASE)    
    #percent
    word = re.sub(r"(\d)+(%)|(\d)+( )+(%)", ' <PERCENTAGE> ', word, flags=re.IGNORECASE)
    #fraction
    word = re.sub(r"(\d+)(\/)(\d+)", ' <FRACTION> ', word, flags=re.IGNORECASE)
    #number
    word = re.sub(r"(\b((\d+\b)|(\d{1,2}\.\d{1,2})))", ' <NUMBER> ', word, flags=re.IGNORECASE)
    #remove puntuation (except for commas and points)
    word = re.sub(r"[^\w\s\.\,<>\/]", '', word, flags=re.IGNORECASE)
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

def remove_stopwords(label, locale):
    if locale == "es":
        label = ' '.join([word for word in label.split(' ') if not word in stopwords.words('spanish')])
        label = ' '.join([word for word in label.split(' ') if not word in CUSTOM_STOPWORDS])
    if locale == "it":
        label = ' '.join([word for word in label.split(' ') if not word in stopwords.words('italian')])   
    return label

def clean_from_distributor_brand(label, locale):
    locale_brands = {
        "it": BRANDS_IT,
        "es": BRANDS_ES
    }
    label = re.sub('|'.join(locale_brands[locale]), '', label, flags=re.IGNORECASE)
    return label    

def stem_label(label):
    # stemming
    #print("stemming...")
    stemmer = ItalianStemmer()
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
    print("lemmatizing...")
    doc = nlp(label)
    lemmas = [token.lemma_ for token in doc]
    label = ' '.join(lemmas)
    print(label)
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

def modify_label(original_label, locale):
    label = original_label
    # crop
    #print("original label: {}".format(label))
    to_crop = bool(random.getrandbits(1))
    label = crop_words(label) if to_crop else label

    # shuffle words 
    label = shuffle_words(label) if bool(random.getrandbits(1)) else label
    
    # lemmas
    to_lemmatize = bool(random.getrandbits(1))
    label = singularize_it(label) if to_lemmatize else label 
    """
    # stem 
    if category != 'c123e16b-0683-4ed0-942a-a9866c90b85c':
        to_stem = bool(random.getrandbits(1))
        label = stem_label(label) if to_stem else label
    
    # remove generic words
    label = remove_generic_words(label) if bool(random.getrandbits(1)) else label  
    """
    # replace with synonim/abbreviation
    to_synonim = bool(random.getrandbits(1))
    label = replace_synonim(label, locale) if to_synonim else label

    label = unidecode.unidecode(label.strip())
    #check if we have word in final label
    match = re.findall(r'([a-zA-Z]+)', label)

    #print("final label: {}".format(label))
    return label if match else original_label

def augmentation_labels(original_series, locale):
    modified_series = original_series.copy()
    modified_series = modified_series.apply(lambda x: modify_label(x, locale))
    series = pd.concat([original_series,modified_series], ignore_index=True, sort=True)  
    series = series.drop_duplicates().reset_index(drop=True)
    print("Total labels in augmentation {}".format(series.count()))
    return series   

def reduce_labels(original_series):
    series = original_series.sample(2000)
    return series


def singularize_it(label):
    label = ' '.join([singularize(word) for word in label.split(' ')])
    return label

def replace_synonim(label,locale):
    with open('resources/synonims.json') as synonims_file:
        synonims_list = json.load(synonims_file)
        for locale_synonims in synonims_list:
            if locale in locale_synonims:
                for original_word in list(locale_synonims[locale].keys()):
                    if original_word in label:
                        index =  np.random.randint(0,len(locale_synonims[locale][original_word]))
                        synonim = locale_synonims[locale][original_word][index]
                        label = re.sub(original_word,synonim,label)
    return label                    


if __name__ == "__main__":
    replace_synonim("gelato multi frutti", "it")