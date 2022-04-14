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
__version__ = "0.0.2"
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
DATASET_FILE = 'final_data_new.json.gz'
WEIGHTS_FILE = 'weights.hdf5'

def BuildModel(units, x, y, dense):
  model = Sequential()
  model.add(LSTM(units, input_shape=(x, y)))
  model.add(Dropout(0.2))
  model.add(Dense(dense, activation='softmax'))
  
  return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Game idea generation using neural networks.')
  parser.add_argument('-max-games', type=int, default=0, help='Maximum number of set descriptions to use, 0 to use them all')
  parser.add_argument('-text-length', type=int, default=100, help='length of description to create')
  parser.add_argument('-lstm-units', type=int, default=256, help='LSTM nodes')
  parser.add_argument('-max-epochs', type=int, default=20, help='Number of epochs')
  parser.add_argument('-batch-size', type=int, default=64, help='Number of training samples per iteration')
  parser.add_argument('-dataset-file', type=ascii, default=DATASET_FILE, help='Dataset file')
  parser.add_argument('-weights-file', type=ascii, default=WEIGHTS_FILE, help='Output file')
  parser.add_argument('-train', action='store_true', help='Use to train the network, otherwise an idea will be generated')
  args = parser.parse_args()

  if 'h' in args or 'help' in args:
    parser.print_help()
    sys.exit(1)

  dataset_file = args.dataset_file.replace("'", "")

  text = ''
  try:
    if '.gz' in dataset_file:
      with gzip.open(dataset_file, 'r') as fin:
        text = fin.read().decode('utf-8')
    else:
      with open(dataset_file, 'r') as fin:
        text = fin.read()
  except IOError:
    print(f"[!] File {dataset_file} not found.")  
    sys.exit(1)

  dataset = json.loads(text)
  max_games = len(dataset) if args.max_games == 0 else min(len(dataset), args.max_games)
  text_length = max(32, args.text_length)
  lstm_units = max(32, args.lstm_units)
  max_epochs = max(1, args.max_epochs)
  batch_size = max(32, args.batch_size)
  weights_file = args.weights_file.replace("'", "")

  filter = []
  bar = Bar(f'[i] Importing {max_games} descriptions', max=max_games)
  for entry in dataset[:max_games]:
    if 'full_desc' in entry:
      desc = entry['full_desc']['desc'].lower()
      desc = re.sub('[^abcdefghijklmnopqrstuwxyz0123456789,.:?!() ]', '', desc)
      filter.append(desc)
    bar.next()
  bar.finish()

  descriptions = ' '.join(filter)

  chars = sorted(list(set(descriptions)))
  char_to_int = dict((c, i) for i, c in enumerate(chars))
  int_to_char = dict((i, c) for i, c in enumerate(chars))
  n_chars = len(descriptions)

  print(f'[i] {n_chars} total chars, {len(chars)} unique chars.')
  
  dataX = []
  dataY = []
  bar = Bar('[i] Patterns', max=n_chars - text_length)
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
  '''
  processedText = {
	'char_to_int': char_to_int,
	'int_to_char': int_to_char,
	'X': X,
	'y': y    
  }

  with open(f'processed{max_games}.pkl', 'wb') as fin:
    pickle.dump(processedText, fin)
  '''

  model = BuildModel(lstm_units, X.shape[1], X.shape[2], y.shape[1])

  if args.train == True:
    print('[i] Training model.')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    if exit(weights_file):
        checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint]
    model.fit(X, y, epochs=max_epochs, batch_size=batch_size, callbacks=callbacks_list)
  else:
    print('[i] Generating game idea.')
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    
    idea = ''
    for i in range(text_length):
      x = np.reshape(pattern, (1, len(pattern), 1))
      x = x / float(n_vocab)
      prediction = model.predict(x, verbose=0)
      index = np.argmax(prediction)
      result = int_to_char[index]
      seq_in = [int_to_char[value] for value in pattern]
      idea += result
      pattern.append(index)
      pattern = pattern[1:len(pattern)]

    print(f'[i] Idea: \'{idea}\'.')
