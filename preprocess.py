
######################################################################################
# NOTE PREPROCESSING                                                                 #
######################################################################################

# This document contains functions that perform different preprocessing aspects of EHR free-text notes. 
# Uninformative characters and line breaks are removed.

import re
import string
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
from imblearn.under_sampling import RandomUnderSampler
from sklearn import utils

def remove_uninformative_characters(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('date of discharge:','',y)
    y=re.sub('--|__|==','',y)
    
    # remove, digits, spaces
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())

    # MIMIC-specific preprocessing
    y=re.sub('date of birth:','',y)
    y=re.sub('sex: m','',y)
    y=re.sub('sex: f','',y)
    
    # Some more preprocessing I'm not entirely sure about
    y=re.sub('incomplete dictation','',y)
    y=re.sub('dictator hung up','',y)
    y=re.sub('dictated by: medquist','',y)
    y=re.sub('completed by:','',y)
    
    return y

def preprocess_notes(notes): 
    notes['text']=notes['text'].fillna(' ')
    notes['text']=notes['text'].str.replace('\n',' ')
    notes['text']=notes['text'].str.replace('\r',' ')
    notes['text']=notes['text'].apply(str.strip)
    notes['text']=notes['text'].str.lower()

    notes['text']=notes['text'].apply(lambda x: remove_uninformative_characters(x))
    
    return notes

def merge_notes(df):
    new_df = pd.DataFrame(columns=list(df.columns))
    for id in df['id'].unique():
        data=df[df['id']==id]
        data=data.reset_index()
        new_row=data.loc[0]
        text=''
        for idx in data.index:
            text=text+data.at[idx,'text']+' '
        new_row['text']=text
        new_df=new_df.append(new_row)
    return new_df

def split_notes(df,split_size):
    df=df.reset_index()
    new_df = pd.DataFrame(columns=list(df.columns))
    for idx in tqdm(df.index):
        x = df.at[idx,'text'].split()
        n = int(len(x)/split_size)
        for j in range(n):
            new_row=df.loc[idx]
            new_row['text']=' '.join(x[j * split_size:(j + 1) * split_size])
            new_df = new_df.append(new_row)
        if len(x) % split_size> 10:
            new_row=df.loc[idx]
            new_row['text']=' '.join(x[-(len(x) % split_size):])
            new_df = new_df.append(new_row)    
            
    return new_df.reset_index(drop=True)

def shuffle(df,col):
    for idx in df.index:
        text=df.at[idx,col]
        words=text.split()
        random.shuffle(words)
        new_text= ' '.join(words)
        df.at[idx,col]=new_text
    return df  

def balance(df,args):
    
    # Divide dataframe into positives and negatives
    df_pos=df[df['label']==1]
    df_neg=df[df['label']==0]
    
    # Calculate amount of desired negative notes
    positive_note_count=len(df_pos.index)
    desired_negative_notes=int(positive_note_count/args.class_balance)
    
    print(desired_negative_notes)
    
    # Shuffle the negative ids and count until the number of minimum notes are desired
    negative_ids=df_neg['id'].unique().tolist()
    negative_ids=utils.shuffle(negative_ids,random_state=42)
    selected_negative_ids=[]
    negative_note_count=0
    id_count=0
    while negative_note_count<desired_negative_notes:
        id_notes=df_neg[df_neg['id']==negative_ids[id_count]]
        selected_negative_ids.append(negative_ids[id_count])
        negative_note_count+=len(id_notes.index)
        id_count+=1        
    
    df_neg_selected=df_neg[df_neg['id'].isin(selected_negative_ids)]
    df=pd.concat([df_pos,df_neg_selected])
    df=df.reset_index(drop=True)
    
    return df