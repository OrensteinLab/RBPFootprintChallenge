# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow
import time
import pickle

def one_hot_encode_DNA(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2].astype(int)

#Read gencode v39 transcripts file
df = pd.read_csv('Data/gencode.v39.transcripts.fa', sep='\t', names=['Transcript_Name'])

#Create new dataframe 'df_Transcripts' containing: Transcript Name, Start line, end line and sequence columns
df_Transcripts = df[df['Transcript_Name'].str.contains('>ENST')]
df_Transcripts['Transcript_Name'] = df_Transcripts['Transcript_Name'].str.split('|').str[0] #Name of transcript
df_Transcripts['Line_Start'] = df_Transcripts.index                      #Index of transcript (in lines)
df_Transcripts['Line_End'] = ''                                          #Create 'Line end' placeholder
df_Transcripts['Sequence'] = ''                                          #Create 'Sequence' placeholder

#Define 'Line End'
df_Transcripts = df_Transcripts.reset_index(drop=True)
df_Transcripts['Line_End'][0:-1] = df_Transcripts['Line_Start'][1:]
df_Transcripts['Line_End'][-1:] = len(df)

#on each transcript stack corresponding lines into one sequence and assign them in the corresponding place in df_Transcripts
start = time.time()
for transcript_no in range(len(df_Transcripts)):
    from_Line = df_Transcripts['Line_Start'][transcript_no]+1
    to_Line   = df_Transcripts['Line_End'][transcript_no]
    transcript0 = df[from_Line:to_Line]
    transcript0_series = transcript0.stack()
    transcript0_str  = transcript0_series.str.cat()
    df_Transcripts['Sequence'][transcript_no] = transcript0_str
    if transcript_no % 2000 == 0 and transcript_no!=0:
        end = time.time()
        print('Transcript No.:{}/{} [{:.2f}%]'.format(transcript_no,len(df_Transcripts),(transcript_no*100/len(df_Transcripts))))  
        ElapsedTime = end - start
        TimeLeft = (1/(transcript_no/len(df_Transcripts)))*ElapsedTime - ElapsedTime
        print('Elapsed Time: {:.2f} Secs. Time left approx.: {:.2f} Secs'.format(ElapsedTime,TimeLeft))

#Delete sequences with missing data (nucleotides assigned with 'N')
df_Transcripts_filtered = df_Transcripts[~df_Transcripts['Sequence'].str.contains('N')].reset_index(drop=True)

#Save processed transcripts
with open('Data/transcripts','wb') as f:
    pickle.dump([df_Transcripts_filtered],f)
    
#Predict SHAPE for every transcript
#Load Model
Models/icSHAPE_InVivo_K562_PrismNetSHAPE_model = tensorflow.keras.models.load_model('Models/icSHAPE_InVivo_K562_PrismNet')

#Process transcripts
Predicted_SHAPE = []
start = time.time()
for n in range(len(df_Transcripts_filtered)):
    #One hot encode transcript, zero pad from left and right
    transcript = df_Transcripts_filtered['Sequence'][n]
    transcript_OHE = np.zeros((40+len(transcript),4))
    transcript_OHE_inner = one_hot_encode_DNA(transcript)
    transcript_OHE[20:20+len(transcript_OHE_inner)]=transcript_OHE_inner
    
    #Divide the transcripts into sequences with a length of 41 to fit the input size of SHAPE model than predict their SHAPE value using the loaded SHAPE model
    transcript_OHE_ext = np.zeros((len(transcript),41,4))
    for i in range(len(transcript)):
        transcript_OHE_ext[i,:,:] = transcript_OHE[i:i+41,:]
    transcript_OHE_ext = transcript_OHE_ext.reshape((transcript_OHE_ext.shape[0],transcript_OHE_ext.shape[1]*transcript_OHE_ext.shape[2]))
    Predicted_SHAPE.append(SHAPE_model.predict(transcript_OHE_ext, batch_size=5000))
    
    #Calculate time remaining and dump predicted SHAPE every 10,000 processed transcripts
    if (n % 1000 == 0 or n==len(df_Transcripts_filtered)-1) and n!=0:
        end = time.time()
        print('Transcript No.:{}/{} [{:.2f}%]'.format(n,len(df_Transcripts_filtered),(n*100/len(df_Transcripts_filtered))))  
        ElapsedTime = end - start
        TimeLeft = (1/(n/len(df_Transcripts_filtered)))*ElapsedTime - ElapsedTime
        print('Elapsed Time: {:.2f} Secs. Time left approx.: {:.2f} Secs'.format(ElapsedTime,TimeLeft))
print('Finished predicting SHAPE for transcripts')
with open('Data/Transcripts_Predicted_SHAPE'.format(n),'wb') as f:
    pickle.dump([Predicted_SHAPE],f)

