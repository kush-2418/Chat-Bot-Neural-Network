#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 23:53:41 2020

@author: kush
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 20:08:26 2020

@author: kush
"""

import pickle
import os
import tensorflow as tf
import nltk
import tflearn
import json
import numpy as np
import random
from keras.models import load_model
from flask import Flask, render_template, flash, request, url_for, redirect, session

app = Flask(__name__)
app.static_folder = 'static'

# restoring all data 
def init():
    global data,words,tags,train_x,train_y,graph,questions,model
    # load the pre-trained Keras model
    data = pickle.load(open('/Users/kush/Downloads/ChatBot/training_data','rb'))
    words = data['words']
    tags = data['tags']
    train_x = data['train_x']
    train_y = data['train_y']
    graph = tf.get_default_graph()
    with open('/Users/kush/Downloads/ChatBot/DataScienceBot.json') as json_data:
        questions = json.load(json_data)

    model = load_model('/Users/kush/Downloads/ChatBot/chatbot_model.h5')

@app.route("/")
def home():
    return render_template("index.html")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words
    
    
def BOW(sentence,words):
     sentence_words = clean_up_sentence(sentence)
     bag = [0] * len(words)
     for s in sentence_words:
         for i,w in enumerate(words):
             if w==s:
                 bag[i] = 1
     return (np.array(bag))
  
ERROR_THRESHOLD = 0.30

def classify(sentence):
    # generate probabilities from model
    results = BOW(sentence, words)
    with graph.as_default():
        results = model.predict(np.array([results]))[0]
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]],r[1]))
    return return_list

def response(sentence):
    results = classify(sentence)
    if results:
        while results:
            for i in questions['questions']:
                if i['tag'] == results[0][0]:
                    res = random.choice(i['responses'])
            results.pop(0)
        return res
                
@app.route("/get", methods=['GET', 'POST'])
def get_response():
    userText = request.args.get('msg')
    resp = response(userText)
    return str(resp)


if __name__ == "__main__":
    init()
    app.run()
