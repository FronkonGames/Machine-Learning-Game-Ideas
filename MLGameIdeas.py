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
__version__ = "0.0.1"
__email__ = "fronkongames@gmail.com"

import io
import sys
import argparse
import os
import gzip
import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# '80000 Steam Games DataSet' by Deepan.N (https://www.kaggle.com/datasets/deepann/80000-steam-games-dataset)
DATASET_FILE = 'final_data_new.json.gz'

DEFAULT_WEIGHTS_FILE = 'weights.hdf5'

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Game idea generation using neural networks.')
  parser.add_argument('-max-games', type=int, default=0)
  parser.add_argument('-text-length', type=int, default=140, help='length of description to create')
  parser.add_argument('-lstm-units', type=int, default=256)
  parser.add_argument('-max-epochs', type=int, default=20)
  parser.add_argument('-batch-size', type=int, default=128)
  parser.add_argument('-dataset-file', type=ascii, default=DATASET_FILE)
  parser.add_argument('-weights-file', type=ascii, default=DEFAULT_WEIGHTS_FILE)
  parser.add_argument('-train', type=bool, default=False)
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

  print(f'[i] {len(dataset)} descriptions, using {max_games}.')

  descriptions = ''
  for entry in dataset[:max_games]:
    if 'full_desc' in entry:
      descriptions += entry['full_desc']['desc'].lower() + ' '

  chars = sorted(list(set(descriptions)))
  char_to_int = dict((c, i) for i, c in enumerate(chars))
  int_to_char = dict((i, c) for i, c in enumerate(chars))
  n_chars = len(descriptions)
  n_vocab = len(chars)

  print(f'[i] {n_chars} total chars, {n_vocab} unique chars.')
  
  seq_length = 140
  dataX = []
  dataY = []
  for i in range(0, n_chars - seq_length, 1):
    seq_in = descriptions[i:i + seq_length]
    seq_out = descriptions[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
  n_patterns = len(dataX)
  print(f'[i] {n_patterns} patterns')

  X = np.reshape(dataX, (n_patterns, seq_length, 1))
  X = X / float(n_vocab)
  y = np_utils.to_categorical(dataY)
  
  model = Sequential()
  model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(y.shape[1], activation='softmax'))
  model.load_weights(DEFAULT_WEIGHTS_FILE)
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  
  start = np.random.randint(0, len(dataX)-1)
  pattern = dataX[start]

  for i in range(140):
  	x = np.reshape(pattern, (1, len(pattern), 1))
  	x = x / float(n_vocab)
  	prediction = model.predict(x, verbose=0)
  	index = np.argmax(prediction)
  	result = int_to_char[index]
  	seq_in = [int_to_char[value] for value in pattern]
  	sys.stdout.write(result)
  	pattern.append(index)
  	pattern = pattern[1:len(pattern)]
  
  sys.exit(1)

  model = Sequential()
#   model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
#   model.add(Dropout(0.2))
  model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dense(y.shape[1], activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  
  filepath=DEFAULT_WEIGHTS_FILE
  checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list = [checkpoint]
  
  model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
