# -*- coding: utf-8 -*-
"""
Created on Thu May 27 23:02:28 2021

@author: Ori Feldman
"""

import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Concatenate, Input
   
#functions
def one_hot_encode_rna(seq):
    mapping = dict(zip("ACGU", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def pearsonr(x, y):
  # Assume len(x) == len(y)
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(xi*xi for xi in x)
  sum_y_sq = sum(yi*yi for yi in y)
  psum = sum(xi*yi for xi, yi in zip(x, y))
  num = psum - (sum_x * sum_y/n)
  den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if den == 0: return 0
  return num / den

#Parameters
seq_len = 41
train_part = 0.9

#Read RNAcompete file 
print('Reading RNA-compete file')
df_rnacomp = pd.read_csv('Data/RNAcompete.clamp', sep=' ', names=['label','sequence'])

#Read structure data from RNAcompete annotations
print('Reading predicted rnacompete icSHAPE values')
df_SHAPE_rnacompete = pd.read_csv('Data/ShapeScoresRNAcompete.csv', sep=',', names=list(range(seq_len)))
SHAPE_rnacompete_np = df_SHAPE_rnacompete.to_numpy()
SHAPE_rnacompete_np = SHAPE_rnacompete_np.reshape(SHAPE_rnacompete_np.shape[0],SHAPE_rnacompete_np.shape[1],1)


print('1-Hot-Encoding rnacompete sequences')
rnacompete_ohe_matrix = np.zeros((len(df_rnacomp),41*4)) #Set onehotmatrix for RNA compete sequences
for i in range(len(df_rnacomp)):
    oneHotVec = one_hot_encode_rna(df_rnacomp['sequence'][i]).flatten()
    rnacompete_ohe_matrix[i,:len(oneHotVec)] = oneHotVec
    if i % 10000 == 0:
        print('Processed rnacompete sequence No. {}/{}'.format(i,len(df_rnacomp)))
rnacompete_ohe_matrix = rnacompete_ohe_matrix.reshape(len(df_rnacomp),41,4)

#Concatenate SHAPE values and sequences
rnacompete_ohe_matrix = np.concatenate((rnacompete_ohe_matrix, SHAPE_rnacompete_np), axis=2)

print('Partitioning data into training and validation')
train_part = int(0.9*len(df_rnacomp))
x_train = rnacompete_ohe_matrix[:train_part]
x_valid = rnacompete_ohe_matrix[train_part::]
y_train = df_rnacomp['label'][:train_part]
y_valid = df_rnacomp['label'][train_part::]

#Building DLPRB-seq Model
print('Building DLPRB-Seq Model')
A1 = Input(shape=(41,5),name='A1')
A2 = Conv1D(256, kernel_size = 11, activation='relu', input_shape=(41,5), use_bias=True, name='A2')(A1)
A3 = MaxPooling1D(pool_size=5, name='A3')(A2)
A4 = Flatten()(A3)

B2 = Conv1D(256, kernel_size = 5, activation='relu', input_shape=(41,5), use_bias=True, name='B2')(A1)
B3 = MaxPooling1D(pool_size=5, name='B3')(B2)
B4 = Flatten()(B3)

model_concat = Concatenate(axis=1)([A4, B4])
M1 = Dense(256, activation='relu', name = 'M2')(model_concat)
M2 = Dense(1, activation='linear', name = 'M3')(M1)

finalModel = Model(inputs=A1,outputs=M2)
finalModel.summary()

#Compile
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.001)
finalModel.compile(loss = 'mse', optimizer=opt)

print('Fitting model')
finalModel.fit(x_train, y_train, epochs = 5, batch_size = 256)

print('Predicting Protein Binding Intensities Validation Set')
y_pred = finalModel.predict(x_valid)

pearsonResult = pearsonr(y_valid,y_pred)
print('Pearson Score: {}'.format(pearsonResult))

finalModel.save('Models/DLPRB_Model')