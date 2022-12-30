from transformers import Trainer, TrainingArguments, Trainer
from datasets import Dataset
import wandb
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import auc, precision_recall_curve, roc_curve,precision_score,recall_score, f1_score
from pynvml import *

import preprocess
import create_dataset

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def optimize_vote_score(model,tokenizer,model_args,df):
    
    for idx in df.index:
        input=tokenizer(df.at[idx,'text'],return_tensors='pt',max_length=model_args['max_length'],padding='max_length',truncation=True)
        input.to("cuda")
        outputs=model(**input).logits.cpu().detach().numpy()
        outputs=np.squeeze(outputs)

        df.at[idx,'pred_0']=outputs[0]
        df.at[idx,'pred_1']=outputs[1]
    
    df['softmax']=0.0

    for idx in df.index:
        x=[df.at[idx,'pred_0'],df.at[idx,'pred_1']]
        x=softmax(x)
        df.at[idx,'softmax']=x[1]
    
    best_c=0
    best_cutoff=0.0
    best_score=len(df['id'].unique())
    
    for c in range(1,10,1):
        for cutoff in np.arange(0.00,1.00,0.05):
            score=0
            for id in df['id'].unique():
                sub_df=df[df['id']==id]
                sub_df=sub_df.reset_index(drop=True)
                model_args['c']=c
                id_score=vote_score(sub_df,'pred_1',model_args)
                pred=id_score>cutoff
                if pred and sub_df.at[0,'label']==0:
                    score+=1
                if not pred and sub_df.at[0,'label']==1:
                    score+=1
                    
            if score<best_score:
                best_c=c
                best_cutoff=cutoff 
    
    model_args['c']=best_c
    model_args['cutoff']=best_cutoff
    return model_args

def vote_score(df,col,model_args):
    c=model_args['c']
    n=len(df.index)
    pmax=df[col].max()
    pmean=df[col].mean()
    score =(pmax+pmean*(n/c)) / (1+n/c)
    return score

def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def run_name(args,model_args):
        
    task_string=model_args['model_name']+'_'
    
    ignore_args=['wandb','clinicalbert','longformer_512','longformer_4096','longformer_4096_merge']
    
    for arg in vars(args):
        if str(arg) not in ignore_args:
            task_string+=str(arg)+'_'+str(getattr(args,arg))+'_'
  
    return task_string

def evaluate(tokenizer,model,df,args,model_args):
        
    task_string=run_name(args,model_args)
    
    task_string=task_string+model_args['mode']
    
    df['pred_0']=0.0
    df['pred_1']=0.0

    for idx in df.index:
        input=tokenizer(df.at[idx,'text'],return_tensors='pt',max_length=model_args['max_length'],padding='max_length',truncation=True)
        input.to("cuda")
        outputs=model(**input).logits.cpu().detach().numpy()
        outputs=np.squeeze(outputs)

        #outputs=softmax_stable(outputs)
        df.at[idx,'pred_0']=outputs[0]
        df.at[idx,'pred_1']=outputs[1]

    gpu_mem=gpu_utilization()

    if args.save_logits:        
        test_results_file=os.path.join(os.getcwd(),'models',model_args['model_name'],model_args['mode']+'_outputs.csv')
        df.to_csv(test_results_file)

    df['softmax_0']=0.0
    df['softmax_1']=0.0

    for idx in df.index:
        x=[df.at[idx,'pred_0'],df.at[idx,'pred_1']]
        x=softmax(x)
        df.at[idx,'softmax_0']=x[0]
        df.at[idx,'softmax_1']=x[1]

    vote_df=pd.DataFrame(columns=['id','score','label','pred'])
    vote_df.set_index('id')
    for id in df['id'].unique():
        data=df[df['id']==id]
        data=data.reset_index()
        vote_df.at[id,'score']=vote_score(data,'pred_1',model_args)
        vote_df.at[id,'label']=data.at[0,'label']
        if vote_df.at[id,'score']>model_args['cutoff']:
            vote_df.at[id,'pred']=1
        else:
            vote_df.at[id,'pred']=0

    if args.save_logits:
        vote_results_file=os.path.join(os.getcwd(),'models',model_args['model_name'],model_args['mode']+'_votes.csv')
        vote_df.to_csv(vote_results_file)

    metrics_file=os.path.join(os.getcwd(),'metrics.csv')
    if not os.path.isfile(metrics_file):
        metrics_df=pd.DataFrame(columns=['model','precision_0','precision_1',
            'recall_0','recall_1','f1_0','f1_1','auprc','auroc','train_time','memory'])
    else:
        metrics_df=pd.read_csv(metrics_file)
    
    metrics_df=metrics_df.set_index('model')
    
    labels=vote_df['label'].tolist()
    preds=vote_df['pred'].tolist()
    scores=vote_df['score'].tolist()

    precision,recall,_=precision_recall_curve(labels, scores)
    fpr,tpr,_=roc_curve(labels,scores)
    prc_score=precision_score(labels,preds,average=None)
    rc_score=recall_score(labels,preds,average=None)
    f1=f1_score(labels,preds,average=None)

    metrics_df.at[task_string,'precision_0']=prc_score[0]
    metrics_df.at[task_string,'precision_1']=prc_score[1]
    metrics_df.at[task_string,'recall_0']=rc_score[0]
    metrics_df.at[task_string,'recall_1']=rc_score[1]
    metrics_df.at[task_string,'f1_0'] =f1[0]
    metrics_df.at[task_string,'f1_1'] =f1[1]
    metrics_df.at[task_string,'auprc']=auc(recall,precision)
    metrics_df.at[task_string,'auroc']=auc(fpr,tpr)
    metrics_df.at[task_string,'train_time']=model_args['train_time']
    metrics_df.at[task_string,'memory']=gpu_mem

    metrics_df.to_csv(metrics_file)


def train(tokenizer,model,args,model_args,train,val,test):

    if args.wandb:
        wandb.init(project="nlp-pipeline", entity="clinical-nlp")

    start_time=time.time()

    model.to("cuda")

    #model.train()

    def tokenize_function(data):
        return tokenizer(data["text"], max_length=512, padding="max_length", truncation=True)

    train_ds=Dataset.from_pandas(train)
    val_ds=Dataset.from_pandas(val)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function,batched=True)
    
    def compute_metrics(pred):
        logits, labels = pred
        predictions = np.argmax(logits, axis=-1)
        outputs=np.apply_along_axis(softmax,axis=1,arr=logits)

        scores=outputs[:,1]
                               
        precision,recall,_=precision_recall_curve(labels, scores)
        fpr,tpr,_=roc_curve(labels,scores)
        auprc=auc(recall,precision)
        auroc=auc(fpr,tpr)
                
        return {"auprc": auprc, "auroc": auroc}
 
    batch_size = model_args["batch_size"]

    output_dir=os.path.join(os.getcwd(),'models',model_args['model_name'])

    logging_steps = len(train_ds) // batch_size
    #training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=args.epochs, learning_rate=args.lr, 
    #                                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
    #                                    load_best_model_at_end=True, metric_for_best_model="auprc", evaluation_strategy="steps", 
    #                                    save_strategy="steps", save_total_limit=1, disable_tqdm=False,
    #                                    report_to="wandb",run_name=run_name(args,model_args))
    
    training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=args.epochs, learning_rate=args.lr, 
                                      per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                        save_strategy='no',save_total_limit=1,disable_tqdm=False,
                                        report_to="wandb",run_name=run_name(args,model_args))

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,train_dataset=train_ds, eval_dataset=val_ds)

    trainer.train()

    model.eval()
    
    model_args=optimize_vote_score(model,tokenizer,model_args,train)

    model_args['train_time']=time.time()-start_time
    
    
    model_args['mode']='unshuffled'
    evaluate(tokenizer,model,test,args,model_args)
    
    model_args['mode']='shuffled'
    shuffled_test_df=preprocess.shuffle(test,'text')
    evaluate(tokenizer,model,shuffled_test_df,args,model_args)

def prepare_dataset(args,dataset_args):
    if args.dataset=='presplit':
        train_file=os.path.join(os.getcwd(),'datasets','train.csv')
        val_file=os.path.join(os.getcwd(),'datasets','val.csv')
        test_file=os.path.join(os.getcwd(),'datasets','test.csv')
        
        train=pd.read_csv(train_file)
        val=pd.read_csv(val_file)
        test=pd.read_csv(test_file)

        # Sometimes sets from other sources will save ids and labels as a float, turn it to int instead
        for column in ['id','label']:
            train[column]=train[column].astype(int)
            val[column]=val[column].astype(int)
            test[column]=test[column].astype(int)
        
        train=train.reset_index(drop=True)
        val=val.reset_index(drop=True)
        test=test.reset_index(drop=True)
    
        train=train[['id','text','label']]
        val=val[['id','text','label']]
        test=test[['id','text','label']]
        
        return train,val,test
    
    if not args.dataset=='custom':
        dataset_file=os.path.join(os.getcwd(),'datasets', args.dataset+'_'+args.task+'.csv')
    else:
        dataset_file=os.path.join(os.getcwd(),'datasets', args.dataset_name+'.csv')

    # If dataset does not exist, create it
    if not os.path.isfile(dataset_file):
        create_dataset.create_dataset(args)
    
    dataset=pd.read_csv(dataset_file)

    # If folds do not exist, or requested fold does not correspond to number of folds, recreate folds and replace dataset
    if not 'fold_'+str(args.fold) in dataset.columns:
            dataset=create_dataset.create_folds(dataset,args)
            dataset.to_csv(dataset_file)

    train=dataset[dataset['fold_'+str(args.fold)]=='train']
    val=dataset[dataset['fold_'+str(args.fold)]=='val']
    test=dataset[dataset['fold_'+str(args.fold)]=='test']

    if args.debug_mode:
        train=train.head(1000)
        val=val.head(1000)
        test=test.head(1000)

    if args.class_balance != 0:
        train=preprocess.balance(train,args)

    if args.balance_all_datasets:
        val=preprocess.balance(val,args)
        test=preprocess.balance(test,args)
              
    train=preprocess.preprocess_notes(train)
    val=preprocess.preprocess_notes(val)
    test=preprocess.preprocess_notes(test)

    if dataset_args['merge_notes']:
        train=preprocess.merge_notes(train)
        val=preprocess.merge_notes(val)
        test=preprocess.merge_notes(test)

    if dataset_args['sentence_splits'] !=0:
        train=preprocess.split_notes(train,args.sentence_splits)
        val=preprocess.split_notes(val,args.sentence_splits)
        test=preprocess.split_notes(test,args.sentence_splits)
    
    # Shuffle datasets, keep only columns of interest
    train=train.sample(frac=1,random_state=42)
    val=val.sample(frac=1,random_state=42)
    test=test.sample(frac=1,random_state=42)

    # Sometimes sets from other sources will save ids and labels as a float, turn it to int instead
    for column in ['id','label']:
        train[column]=train[column].astype(int)
        val[column]=val[column].astype(int)
        test[column]=test[column].astype(int)
    
    train=train.reset_index(drop=True)
    val=val.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    train=train[['id','text','label']]
    val=val[['id','text','label']]
    test=test[['id','text','label']]
    
    return train,val,test
