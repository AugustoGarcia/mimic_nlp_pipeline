######################################################################################
# DATASET CREATOR                                                                    #
######################################################################################

# This document creates an NLP dataset with n-fold CV from MIMIC-III data

import argparse
import os
import numpy as np
import pandas as pd
from sklearn import model_selection

######################################################################################
# Splitting

def create_folds(dataset,args):
    positives=[]
    negatives=[]
    for subject_id in dataset['id'].unique():
        subject_data=dataset[dataset['id']==subject_id]
        subject_data=subject_data.reset_index()
        if subject_data['label'].sum()>0:
            positives.append(subject_data.at[0,'id'])
        else:
            negatives.append(subject_data.at[0,'id'])
    
    included_admissions=positives+negatives

    X=included_admissions
    y=np.append(np.ones(len(positives)),np.zeros(len(negatives)))
    
    X=np.array(X)
    y=np.array(y)
    
    sss=model_selection.StratifiedShuffleSplit(n_splits=args.folds,test_size=args.test_size,random_state=42)

    folds=['fold_'+str(n) for n in range(args.folds)]
    dataset[folds]=''
    
    fold=0
    for train_index,test_index in sss.split(X,y):
        X_train_val=X[train_index]
        y_train_val=y[train_index]
        X_test=X[test_index]
        
        X_train,X_val,y_train,y_val = model_selection.train_test_split(X_train_val, y_train_val, train_size=1-(1.1*args.test_size), random_state=42)

        for idx in dataset.index:
            if dataset.at[idx,'id'] in list(X_train):
                dataset.at[idx,'fold_'+str(fold)]='train'
            if dataset.at[idx,'id'] in list(X_val):
                dataset.at[idx,'fold_'+str(fold)]='val'
            if dataset.at[idx,'id'] in list(X_test):
                dataset.at[idx,'fold_'+str(fold)]='test'
                
        fold=fold+1

    return dataset

######################################################################################
# Task-specific dataset

def create_specific_dataset(df,args):
    
    # For the readmission task, the output label is readmission in less than 30 days from discharge
    if args.task=='readmission':
        df['label'] = (df.days_next_admit < 30).astype('int')

    # For the mortality task, the output label is expiration within the hospital stay
    if args.task=='mortality':
        df['label'] = df['hospital_expire_flag'] 
        
    # For the both task, the output label is either readmission or expiration
    if args.task=='both':
        df['label']=0
        for idx in df.index:
            if df.at[idx,'hospital_expire_flag']==1:
                df.at[idx,'label']=1
            if df.at[idx,'days_next_admit']<30:
                df.at[idx,'label']=1
            
    # For the early df, we include notes from the first 3 days of admissions with LOS>4
    if args.dataset=='early':
        df=df[df['length_of_stay']>=4]
        df=df[df['note_day']<=3]

    # For the discharge df, we include discharge summaries only. The mortality task should be trivial in this dataset. Sometimes there will be more than
    # one summary for a reason (e.g. dictation was interrupted). For this reason all discharge summaries with the latest chart date
    # for the same admission id are merged into one
    if args.dataset=='discharge':
        df=df[df['category']=='Discharge summary']
        
        if args.task=='readmission':
            df=df[df['hospital_expire_flag']==0]
        
        reduced_df=pd.DataFrame(columns=list(df.columns))
        for id in df['hadm_id'].unique():
            id_df=df[df['hadm_id']==id]

            id_df=id_df.sort_values(by=['chartdate'],ascending=False)
            id_df=id_df.reset_index(drop=True)
            id_df=id_df[id_df['chartdate']==id_df.at[0,'chartdate']]
            id_df=id_df.reset_index(drop=True)

            text=''
            for idx in id_df.index:
                text+=id_df.at[idx,'text']+' '    
            new_row=id_df.loc[0]
            new_row['text']=text

            reduced_df=reduced_df.append(new_row)
            
        df=reduced_df
    
    df=df.rename(columns={"hadm_id":'id'})
    df=df[['id','text','label']]
    df=df.reset_index()
    return df

######################################################################################
# Retrieval

def create_dataset(args):
    admission_file=os.path.join(os.getcwd(),'mimic', 'admissions.csv')
    df_adm=pd.read_csv(admission_file)

    df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_adm = df_adm.sort_values(['subject_id','admittime'])
    df_adm = df_adm.reset_index(drop = True)
    
    # Exclude newborn admissions
    df_adm = df_adm[df_adm['admission_type']!='NEWBORN']

    # For each admission, get all subsequent admissions for same id that are emergencies. Since they are already sorted by chartdate, retrieve smallest time difference in days. Also get length of stay
    for idx in df_adm.index:
        dataset=df_adm[df_adm['subject_id']==df_adm.at[idx,'subject_id']]
        dataset=dataset.drop(idx)
        dataset=dataset[dataset['admittime']>df_adm.at[idx,'dischtime']]
        dataset=dataset[dataset['admission_type'] != 'ELECTIVE']
        if not dataset.empty:
            dataset=dataset.sort_values(['admittime'])
            dataset=dataset.reset_index()
            df_adm.at[idx,'days_next_admit']=(dataset.at[0,'admittime']-df_adm.at[idx,'dischtime']).total_seconds()/(24*60*60)

        df_adm.at[idx,'length_of_stay']=(df_adm.at[idx,'dischtime']-df_adm.at[idx,'admittime']).total_seconds()/(24*60*60)

    # Retrieve notes
    notes_file=os.path.join(os.getcwd(),'mimic', 'noteevents.csv')
    df_notes = pd.read_csv(notes_file,low_memory=False)
    df_notes = df_notes.sort_values(by=['subject_id','hadm_id','chartdate'])
    
    # Some notes are exact duplicates and they should be dropped
    df_notes=df_notes.drop_duplicates(subset=['hadm_id','text'],keep='last')

    # Merge notes
    df_adm_notes = pd.merge(df_adm[['subject_id','hadm_id','days_next_admit','length_of_stay','admittime','hospital_expire_flag']],
                        df_notes[['subject_id','hadm_id','chartdate','text','category']], 
                        on = ['subject_id','hadm_id'],
                        how = 'left')

    df_adm_notes=df_adm_notes.reset_index(drop=True)
    df_adm_notes.chartdate = pd.to_datetime(df_adm_notes.chartdate, format = '%Y-%m-%d', errors = 'coerce')

    for idx in df_adm_notes.index:
        df_adm_notes.at[idx,'note_day']=(df_adm_notes.at[idx,'chartdate'].to_pydatetime()-df_adm_notes.at[idx,'admittime']).total_seconds()/(24*60*60)

    # This is the dataset that now gets subdivided by tasks
    dataset=create_specific_dataset(df_adm_notes,args)

    dataset=create_folds(dataset,args)
    
    dataset_file=os.path.join(os.getcwd(),'datasets', args.dataset+'_'+args.task+'.csv')
    dataset.to_csv(dataset_file)

######################################################################################
