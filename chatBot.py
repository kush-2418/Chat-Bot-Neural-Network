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
import numpy as np
import random
import json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# restoring all data 

data = pickle.load(open('training_data','rb'))
words = data['words']
tags = data['tags']
train_x = data['train_x']
train_y = data['train_y']

print(tags)

with open('DataScienceBot.json') as json_data:
    questions = json.load(json_data)
    
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")


from keras.models import load_model
model = load_model('chatbot_model.h5')


with open('DataScienceBot.json') as json_data:
    questions = json.load(json_data)
    
def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [word.lower() for word in sentence_words]

  return sentence_words

# BOW array

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



res = response('what is tf-idf')
print(res)
