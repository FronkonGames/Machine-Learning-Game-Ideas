########################################################################################################################
# Copyright (c) Martin Bustos @FronkonGames <fronkongames@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
########################################################################################################################
__author__ = "Martin Bustos <fronkongames@gmail.com>"
__copyright__ = "Copyright 2022, Martin Bustos"
__license__ = "MIT"
__version__ = "0.9.0"
__email__ = "fronkongames@gmail.com"

import io
import sys
import argparse
import re
import os
from os.path import exists
import gzip
import json
from progress.bar import Bar
import random
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# '80000 Steam Games DataSet' by Deepan.N (https://www.kaggle.com/datasets/deepann/80000-steam-games-dataset)
DATASET_FILE        = 'final_data_new.json.gz'
PROCESSED_FILE      = 'processed.pickle'
WEIGHTS_FILE        = 'weights.hdf5'
MODEL_LOSS          = 'categorical_crossentropy'
MODEL_OPTIMIZER     = 'adam'

def LoadDescriptions(filename, max):
  '''
  Loads game descriptions, filters characters and returns a string of text with all the characters.
  '''
  filename = filename.replace("'", "")

  text = ''
  try:
    if '.gz' in filename:
      with gzip.open(filename, 'r') as fin:
        text = fin.read().decode('utf-8')
    else:
      with open(filename, 'r') as fin:
        text = fin.read()
  except IOError:
    print(f'[!] File \'{filename}\' not found.')  
    sys.exit(1)

  dataset = json.loads(text)
  max = len(dataset) if max == 0 else min(len(dataset), max)

  filter = []
  bar = Bar('[i] Importing game descriptions', max=max)
  for entry in dataset[:max]:
    if 'full_desc' in entry:
      desc = entry['full_desc']['desc'].lower()
      desc = re.sub('[^abcdefghijklmnopqrstuwxyz0123456789,.:?!() ]', '', desc)
      filter.append(desc)
    bar.next()
  bar.finish()

  return ' '.join(filter)

def ProcessText(descriptions):
  '''
  Loads a file with the processed descriptions or processes them and creates a new file.
  '''
  processedText = None

  if exists(f'{PROCESSED_FILE}.gz'):
    print('[i] Importing preprocessed text.')
    with gzip.open(f'{PROCESSED_FILE}.gz', 'rb') as fin:
      processedText = pickle.load(fin)
  else:
    chars = sorted(list(set(descriptions)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(descriptions)

    print(f'[i] {n_chars} total chars, {len(chars)} unique chars.')
  
    dataX = []
    dataY = []
    bar = Bar('[i] Processing text', max=n_chars - text_length)
    for i in range(0, n_chars - text_length, 1):
      seq_in = descriptions[i:i + text_length]
      seq_out = descriptions[i + text_length]
      dataX.append([char_to_int[char] for char in seq_in])
      dataY.append(char_to_int[seq_out])
      bar.next()
    bar.finish()
    n_patterns = len(dataX)
    print(f'[i] {n_patterns} patterns.')

    X = np.reshape(dataX, (n_patterns, text_length, 1))
    X = X / float(len(chars))
    y = np_utils.to_categorical(dataY)

    processedText = {
      'chars': chars,
      'char_to_int': char_to_int,
      'int_to_char': int_to_char,
      'X': X,
      'y': y,
      'dataX': dataX    
    }

    print('[i] Saving preprocessed text.')
    with gzip.open(f'{PROCESSED_FILE}.gz', 'wb') as fout:
      pickle.dump(processedText, fout)

  return processedText

def BuildModel(units, x, y, dropout, dense):
  '''
  Build a model with two LSTM layers.
  '''
  model = Sequential()
  model.add(LSTM(units, input_shape=(x, y), return_sequences=True))
  model.add(Dropout(dropout))
  model.add(LSTM(units))
  model.add(Dropout(dropout))
  model.add(Dense(dense, activation='softmax'))

  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Game idea generation using neural networks.')
  parser.add_argument('-games', type=int, default=0, help='Maximum number of set descriptions to use, 0 to use them all.')
  parser.add_argument('-length', type=int, default=100, help='length of description to create.')
  parser.add_argument('-units', type=int, default=256, help='LSTM nodes.')
  parser.add_argument('-epochs', type=int, default=20, help='Number of epochs.')
  parser.add_argument('-batch', type=int, default=64, help='Number of training samples per iteration.')
  parser.add_argument('-dropout', type=float, default=0.2, help='.')
  parser.add_argument('-dataset', type=ascii, default=DATASET_FILE, help='Dataset file.')
  parser.add_argument('-weights', type=ascii, default=WEIGHTS_FILE, help='Output file.')
  parser.add_argument('-train', action='store_true', help='Use to train the network, otherwise an idea will be generated.')
  args = parser.parse_args()

  if 'h' in args or 'help' in args:
    parser.print_help()
    sys.exit(1)

  text_length = max(32, args.length)
  lstm_units = max(32, args.units)
  max_epochs = max(1, args.epochs)
  batch_size = max(32, args.batch)
  dropout = args.dropout
  weights_file = args.weights.replace("'", "")

  descriptions = LoadDescriptions(args.dataset, args.games)
  
  processedText = ProcessText(descriptions)

  model = BuildModel(lstm_units, processedText['X'].shape[1], processedText['X'].shape[2], dropout, processedText['y'].shape[1])

  if args.train == True:
    print('[i] Training model.')
    model.compile(loss=MODEL_LOSS, optimizer=MODEL_OPTIMIZER)
    checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(processedText['X'], processedText['y'], epochs=max_epochs, batch_size=batch_size, callbacks=callbacks_list)
    model.summary()
    
    print('[i] Done.')
  else:
    if not exists(weights_file):
      print(f'[!] {weights_file} not found, you must train the network first.')
      sys.exit(1)
  
    print('[i] Generating game idea.')
    model.load_weights(weights_file)
    model.compile(loss=MODEL_LOSS, optimizer=MODEL_OPTIMIZER)
    start = np.random.randint(0, len(processedText['dataX'])-1)
    pattern = processedText['dataX'][start]
    
    idea = ''
    for i in range(text_length):
      x = np.reshape(pattern, (1, len(pattern), 1))
      x = x / float(len(processedText['chars']))
      prediction = model.predict(x, verbose=0)
      index = np.argmax(prediction)
      result = processedText['int_to_char'][index]
      seq_in = [processedText['int_to_char'][value] for value in pattern]
      idea += result
      pattern.append(index)
      pattern = pattern[1:len(pattern)]

    print(f'[i] Idea: \'{idea}\'.')
