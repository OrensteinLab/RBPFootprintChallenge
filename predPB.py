# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:22:39 2021

@author: Ori Feldman
"""

import tensorflow as tf
import numpy as np
import time
import pickle

def one_hot_encode_dna(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


#Load predicted SHAPE
print('Load predicted icSHAPE')
with open('Data/Transcripts_Predicted_SHAPE', 'rb') as f:
    predictedSHAPE= pickle.load(f)
predictedSHAPE = predictedSHAPE[0]

#Load Processed Transcripts
print('Load Processed transcripts')
with open('Data/transcripts', 'rb') as f:
    df_Transcripts = pickle.load(f)
df_Transcripts = df_Transcripts[0]

#Load Model Trained on RNAcompete
RNACMPT_Model = tf.keras.models.load_model('Models/DLPRB_Model')

#Pre Processing
Predicted_Protein_Binding = []
start = time.time()
for n in range(len(df_Transcripts)):
    #One hot encoding current transcript
    transcript = df_Transcripts['Sequence'][n]
    transcript_OHE = np.zeros((40+len(transcript),4))
    transcript_SHAPE = np.zeros((40+len(transcript),1))
    
    #Pad with zeros from right and left each transcript to enable prediction near the edges of the transcript
    transcript_OHE_inner = one_hot_encode_dna(transcript)
    transcript_OHE[20:20+len(transcript_OHE_inner)] = transcript_OHE_inner
    transcript_SHAPE[20:20+len(transcript_OHE_inner)] = predictedSHAPE[n]
    
    #Extend matrix using sliding window and concatenate SHAPE
    transcript_OHE_ext = np.zeros((len(transcript),41,5))
    for i in range(len(transcript)):
        transcript_OHE_ext[i,:,:4] = transcript_OHE[i:i+41,:]
        transcript_OHE_ext[i,:,4] = transcript_SHAPE[i:i+41,0]
    
    #Predict Protein binding for transcript[n]
    Predicted_Protein_Binding.append(RNACMPT_Model.predict(transcript_OHE_ext, batch_size=10000))
    
    if (n % 1000 == 0 or n == len(df_Transcripts)-1) and n!=0:
        end = time.time()
        print('Transcript No.:{}/{} [{:.2f}%]'.format(n,len(df_Transcripts),(n*100/len(df_Transcripts))))  
        ElapsedTime = end - start
        TimeLeft = (1/(n/len(df_Transcripts)))*ElapsedTime - ElapsedTime
        if TimeLeft > 3600:
            print('Elapsed Time: {}h and {}m. Time left approx.: {}h and {}m'.format(int(ElapsedTime//3600),int((ElapsedTime%3600)//60),int(TimeLeft//3600),int((TimeLeft%3600)//60)))
        else:
            print('Elapsed Time: {}h and {}m. Time left approx.: {} minutes'.format(int(ElapsedTime//3600),int((ElapsedTime%3600)//60),int(TimeLeft//60)))
with open('Data/Predicted_Protein_Binding'.format(n),'wb') as f:
    pickle.dump([Predicted_Protein_Binding],f)


