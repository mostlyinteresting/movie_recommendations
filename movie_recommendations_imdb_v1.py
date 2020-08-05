
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #using cpu no gpu

inDir = '<YOUR_DIRECTORY>'
os.chdir(inDir)

import pandas as pd
import numpy as np

import re
import sqlite3

import tensorflow_hub as hub

import tokenization
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

from fuzzywuzzy import process

random_state = 12345 #reproducibility

pd.options.mode.chained_assignment = None

conn = sqlite3.connect(f'{inDir}imdb.sqlite')

bracketText = re.compile(r"[\(\[].*?[\)\]]")
whitespace = re.compile(r"\s+")

def process_text(text):
    text = text.replace('\n',' ')
    text = bracketText.sub('', text)
    text = whitespace.sub(' ', text)
    return text.strip().lower()

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def getMostSimilar(name, s, g, data, n = 5, t = True):
    
    ID = data['imdbId'][ data['title'] == name ].values[0]
    year = data['year'][ data['title'] == name ].values[0]
    y = s.loc[ID]
    y.sort_values(ascending = False, inplace = True)
    
    diff = abs(data['year'] - year)
    diff.index = data['imdbId']
    diff = diff[ diff <= 5 ]
    
    sameGenres = getSameGenres(ID,g)
    sameGenres = sameGenres[ sameGenres['score'] > sameGenres['score'].quantile(.9) ].index.tolist()
    y = y[ (y.index.isin(sameGenres)) ]
    if t == True:
        y = y[ (y.index.isin(diff.index)) ]
    y = y[ y > y.quantile(.9) ]
    
    z = data[ (data['imdbId'].isin(y.index)) & (data['imdbId'] != ID) ].sort_values('rating',ascending = False)
    
    return z['title'].head(n)

def getSameGenres(ID, g):
    out = pd.DataFrame({'imdbId':g.index,'score':np.nan})
    out['score'] = 1- out['imdbId'].apply( lambda x: spatial.distance.cosine(g.loc[x,:], g.loc[ID,:]))
    out.set_index('imdbId', inplace = True)
    return out

def getYear(x):
    try:
        out = int(x.rsplit('(',1)[1].rsplit(')',1)[0].strip())
    except:
        out = None
    return out

def main():
    
    d = pd.read_csv(f'{inDir}ml-25m/links.csv',dtype={'imdbId':str,'movieId':str})
    m = pd.read_csv(f'{inDir}ml-25m/movies.csv',dtype={'imdbId':str,'movieId':str})
    r = pd.read_csv(f'{inDir}ml-25m/ratings.csv',dtype={'movieId':str})
    rN = r.groupby('movieId')['rating'].count().reset_index()
    rN.rename(columns = {'rating':'nRatings'}, inplace = True)
    r = r.groupby('movieId')['rating'].mean().reset_index()
    
    d = m.merge(d, how = 'inner', on = ['movieId']).merge(r, how = 'left', on = ['movieId']).merge(rN, how = 'left', on = ['movieId'])
    
    data = pd.read_sql_query("SELECT * FROM synopses", conn)
    data.rename(columns = {'synopsis':'text'}, inplace = True)
    data = data.merge(d, how = 'left', on = ['imdbId'])
    data = data[ data['rating'].notnull() ]
    data['text'] = data['text'].apply(process_text)
    data.loc[data.genres.str.contains(' '),'genres'] = 'none'
    data['genres'] = data['genres'].str.lower()
    data['genres'] = data['genres'].str.replace('sci-fi','scifi')
    data['year'] = data['title'].apply( lambda x: getYear(x))
    
    vectorizer = CountVectorizer()
    g = vectorizer.fit_transform(data['genres'])
    g = pd.DataFrame(g.toarray(), index = data['imdbId'], columns = vectorizer.get_feature_names())
    g.drop('imax', axis =1, inplace = True)
        
    module_url = "https://tfhub.dev/tensorflow/bert_en_wwm_uncased_L-24_H-1024_A-16/2"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    
    max_len = 300
    
    vocabFile = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    doLowerCase = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocabFile, doLowerCase)
    
    trainInput = bert_encode(data['text'].values, tokenizer, max_len=max_len)
    
    X = trainInput[0]
    s = cosine_similarity(X)
    s = pd.DataFrame(s, index = data['imdbId'], columns = data['imdbId'])
    
    name = process.extract("Once Upon a Time in America", data['title'], limit=1)[0][0]
    getMostSimilar(name, s, g, data, n = 5, t = True)

if __name__ == '__main__':
    
    main()
