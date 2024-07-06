from flask import Flask, request, render_template
import numpy
import re
import pandas as pd
import numpy as np
import keras
import pickle
import os
import tensorflow as tf
from tensorflow import keras



from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer


app = Flask(__name__)



model = tf.keras.models.load_model('Humourous Phrases Generator.keras')

dataX = np.load('dataX.npy')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def next_tokens(input_str, n):
    final_string = ''
    for i in range(n):
        token = tokenizer.texts_to_sequences([input_str])[0]

        prediction = model.predict([token], verbose=0)
        lista = list(range(0,9312))
        word = reverse_word_map[numpy.random.choice(a = lista, p = prediction[0])]
        final_string = final_string + word + ' '
        input_str = input_str + ' ' + word
        input_str = ' '.join(input_str.split(' ')[1:])
    return final_string




@app.route('/')
def main():
    return render_template('html.html', humor='')

@app.route('/', methods=['POST'])
def generate_clickbait():
    if request.method == 'POST':
        start = numpy.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        input_str = ' '.join([reverse_word_map[value] for value in pattern])
        output = next_tokens(input_str, 50)
         
        return render_template('html.html', humor=output)



