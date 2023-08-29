# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 09:46:43 2021

@author: Ori Feldman
"""
import pickle
import numpy as np
import time 
import pandas as pd

#Load predicted SHAPE
with open('Data/Predicted_Protein_Binding', 'rb') as f:
    predictedPB= pickle.load(f)
predictedPB = predictedPB[0]

maxValues = np.zeros(len(predictedPB)*10)

#Process each transcript and retrieve top 10 binding intensity scores from each one of  them
start = time.time()
for transcript in range(len(predictedPB)):
    if len(predictedPB[transcript])>10:
        maxValues[transcript*10:(transcript+1)*10] = sorted(predictedPB[transcript])[-10:]
        if transcript%2000 == 0 and transcript!=0:
            end = time.time()
            print('Transcript No.:{}/{} [{:.2f}%]'.format(transcript,len(predictedPB),(transcript*100/len(predictedPB))))  
            ElapsedTime = end - start
            TimeLeft = (1/(transcript/len(predictedPB)))*ElapsedTime - ElapsedTime
            print('Elapsed Time: {:.2f} Secs. Time left approx.: {:.2f} Secs'.format(ElapsedTime,TimeLeft))
            
with open('Data/MaxValues','wb') as f:
    pickle.dump([maxValues],f)
    
 #Load Maxmimum values if were already processed
# with open('Data/MaxValues', 'rb') as f:
#     maxValues= pickle.load(f)
# maxValues = maxValues[0]   

#Keep only unique scores
uniqueMaxValues = np.unique(maxValues)

#Sort Max Values    
sortedMaxValues = -np.sort(-uniqueMaxValues)

#Load Transcripts
with open('Data/transcripts', 'rb') as f:
    df_Transcripts= pickle.load(f)
df_Transcripts = df_Transcripts[0]

#Remove '>' and version number '.' from transcripts
df_Transcripts['Transcript_Name'] = df_Transcripts['Transcript_Name'].str.split('>').str[1].str.split('.').str[0]

#finding matching transcripts to every score (repeat only for top 5000 scores)
transcriptList = []
transcriptNameList = []
nucList = []
seqList = []
scoreList = []
for i in range(5000):
    #Find number of Transcript
    transcript_no_idx = np.where(maxValues == sortedMaxValues[i])
    
    for j in range(len(transcript_no_idx[0])):
        transcript_no = int(transcript_no_idx[0][j]/10)
        #Find index inside transcript
        nuc_no_idx = np.where(predictedPB[transcript_no] == sortedMaxValues[i])
        for k in range(len(nuc_no_idx[0])):
            nuc_no = nuc_no_idx[0][k]
            nucList.append(nuc_no)
            #Attach the sequence
            if nuc_no+21 <= len(df_Transcripts['Sequence'][transcript_no]) and nuc_no-20>=0:
                seq = df_Transcripts['Sequence'][transcript_no][nuc_no-20:nuc_no+21]
            #Handle cases where max values are at the edges (start or end of transcript)
            elif nuc_no+21 > len(df_Transcripts['Sequence'][transcript_no]) and nuc_no-20>=0:
                seq = df_Transcripts['Sequence'][transcript_no][nuc_no-20:len(df_Transcripts['Sequence'][transcript_no])]
            else:
                seq = df_Transcripts['Sequence'][transcript_no][0:nuc_no+21]
                
            seqList.append(seq)
            #append to transcript number list
            transcriptList.append(transcript_no)
            #Attach Transcript name
            transcriptNameList.append(df_Transcripts['Transcript_Name'][transcript_no])
            #Append relevant score
            scoreList.append(sortedMaxValues[i])

#Create dataframe with max values
dict = {'transcript_identifier': transcriptNameList, 'Score': scoreList, 'nuc_index_in_transcript':nucList, 'Sequence': seqList}  
df_max_values = pd.DataFrame(dict)

#Assign start and end indices for each sequence in the transcript
df_max_values['Start_index'] =  df_max_values['nuc_index_in_transcript'] - 20
df_max_values['Start_index'][df_max_values['Start_index']<0] =  0

df_max_values['End_index'] = ''
for i in range(len(df_max_values)):
    df_max_values['End_index'][i] = df_max_values['Start_index'][i] + len(df_max_values['Sequence'][i])

df_max_values = df_max_values.drop(columns=['nuc_index_in_transcript'])

#Read Transcript Quantification File
names = ['transcript_id','gene_id','length','effective_length','expected_count','TPM','FPKM','IsoPct','posterior_mean_count','posterior_standard_deviation_of_count',
         'pme_TPM','pme_FPKM','IsoPct_from_pme_TPM','TPM_ci_lower_bound','TPM_ci_upper_bound','FPKM_ci_lower_bound','FPKM_ci_upper_bound']
df_TransQuant = pd.read_csv('Data/GSM2400178_ENCFF741ZML_transcript_quantifications_hg19_K562.tsv',names=names,sep='\t')
#Remove version number (after '.')
df_TransQuant['transcript_id'] = df_TransQuant['transcript_id'].str.split('.').str[0]

#Find FPKM values for corresponding transcripts 
start = time.time()
FPKM_List = []
for i in range(len(transcriptNameList)):
    FPKM_Row = np.where(df_TransQuant['transcript_id'] == transcriptNameList[i])
    if len(FPKM_Row[0]) == 0: #If FPKM doesn't exist for the specific sequence
        FPKM = ''
    else:
        FPKM = df_TransQuant['FPKM'][FPKM_Row[0][0]]
    FPKM_List.append(FPKM)
    if i%2000 == 0 and i!=0:
        end = time.time()
        print('FPKM No.:{}/{} [{:.2f}%]'.format(i,len(transcriptNameList),(i*100/len(transcriptNameList))))  
        ElapsedTime = end - start
        TimeLeft = (1/(i/len(transcriptNameList)))*ElapsedTime - ElapsedTime
        print('Elapsed Time: {:.2f} Secs. Time left approx.: {:.2f} Secs'.format(ElapsedTime,TimeLeft))
        
df_max_values['FPKM'] = FPKM_List

df_max_values.to_csv('Top_Protein_Binding_Intensities_Prediction.tsv', sep = '\t',index=False)
