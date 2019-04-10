from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
import shorttext
import numpy as np
import pandas as pd
import pickle
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import utils as my_utils
import w2v_cnn

SEED = 500
MODEL_SIZE = 300
MAX_NB_WORDS = 15000

def train_word2vec():
    #x_train, x_validation, x_test, y_train, y_validation, y_test = my_utils.load_data("experiments/custom_w2v/iri_product_labels.csv")
    #all_x = pd.concat([x_train,x_validation,x_test])
    #all_x_w2v = labelize(all_x)   
  
    df = pd.read_csv("experiments/custom_w2v/data/iri/pps_product_es.csv", header = 0)  
    all_x = df.name      # "name" is the name of column literally     
    vocabulary = [my_utils.preprocess_vocab(label) for label in tqdm(all_x.tolist())]
   
    model_ug_cbow = Word2Vec(sg=0, 
        size=MODEL_SIZE, 
        window=4, 
        negative=1, 
        min_count=2, alpha=0.065, min_alpha=0.065)
    model_ug_cbow.build_vocab(vocabulary)

    model_ug_cbow.train(utils.shuffle(vocabulary), total_examples=len(vocabulary), epochs=30)
    
    model_ug_sg = Word2Vec(sg=1, size=MODEL_SIZE, window=4, negative=10, min_count=2,
                        alpha=0.05, min_alpha=0.001)
    model_ug_sg.build_vocab(vocabulary)

    model_ug_sg.train(vocabulary, total_examples=len(vocabulary), epochs=30)
   
    print('size of vocabulary {}'.format(len(model_ug_sg.wv.vocab)))
    model_ug_cbow.wv.save_word2vec_format('model_cbow.bin', binary=True)
    model_ug_sg.wv.save_word2vec_format('model_sg.bin', binary=True)


def train_log_reg(wvmodel): 
    x_train, x_validation, x_test, y_train, y_validation, y_test = my_utils.load_data("experiments/290119_custom_w2v/food_no_acc.csv")
    #y_train, y_validation, y_test = w2v_cnn.hot_encode_labels(y_train, y_validation, y_test)
    train_vecs_average = preprocessing.scale(np.concatenate([my_utils.get_w2v_general(z, MODEL_SIZE, wvmodel,aggregation='mean') for z in x_train]))
    clf = LogisticRegression(solver='liblinear', multi_class='ovr')
    clf.fit(train_vecs_average, y_train)  

    #save
    f = open("ft_log_reg_model_av.pickle", "wb")
    f.write(pickle.dumps(clf))
    f.close()

def predict_log_reg(wvmodel, file_truth):
    log_reg_model = pickle.load(open('experiments/120219_log_reg/prep_meat_sav/ft_log_reg_model_av.pickle', 'rb')) 
    x_truth, y_truth= pd.read_csv(file_truth, header=0, names=['label', 'category_uid'])  
    embeddings = preprocessing.scale(np.concatenate([my_utils.get_w2v_general(z, MODEL_SIZE, wvmodel) for z in x_truth]))
    predicted_categories = []
    y_truth = y_truth.apply(lambda x: my_utils.uid_to_explicit(x)).tolist()
    for pred in log_reg_model.predict(embeddings):
        print(my_utils.uid_to_explicit(pred))
        predicted_categories.append(my_utils.uid_to_explicit(pred))
    
    my_utils.create_confusion_matrix(y_truth, predicted_categories, my_utils.get_food_category_names())
    df = pd.DataFrame({'label': x_truth.tolist(), 'predicted_category': predicted_categories, 'truth': y_truth})
    df.to_csv('prediction.csv', index=False) 

def explore(wvmodel):
    result = wvmodel.most_similar(positive=['chocolat'])
    #result = wvmodel.similarity('curly', 'donuts')
    print(result)
    result = wvmodel.most_similar(positive=['nutella'])
    print(result)

def labelize(labels):
    result = []
    prefix = 'all'
    for i, t in zip(labels.index, labels):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result

if __name__ == "__main__":
    translate('pommes vrac')
   