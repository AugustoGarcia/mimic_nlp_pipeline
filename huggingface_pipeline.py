from transformers import Trainer, TrainingArguments, Trainer
from datasets import Dataset
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

def vote_score(df,col):
    c=2
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

def evaluate(tokenizer,model,df,args,model_args):
    
    task_string=model_args['model_name']+'_'+model_args['mode']+'_'+args.task+'_'+str(args.fold)

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

    test_results_file=os.path.join(os.getcwd(),'models',model_args['model_name'],'outputs_'+task_string+'.csv')
    df.to_csv(test_results_file)

    df['softmax_0']=0.0
    df['softmax_1']=0.0

    for idx in df.index:
        x=[df.at[idx,'pred_0'],df.at[idx,'pred_1']]
        x=softmax(x)
        df.at[idx,'softmax_0']=x[0]
        df.at[idx,'softmax_1']=x[1]

    vote_df=pd.DataFrame(columns=['id','score_0','score_1','label','pred'])
    vote_df.set_index('id')
    for id in df['id'].unique():
        data=df[df['id']==id]
        data=data.reset_index()
        vote_df.at[id,'score_0']=vote_score(data,'softmax_0')
        vote_df.at[id,'score_1']=vote_score(data,'softmax_1')
        vote_df.at[id,'label']=data.at[0,'label']
        if vote_df.at[id,'score_0']>vote_df.at[id,'score_1']:
            vote_df.at[id,'pred']=0
        else:
            vote_df.at[id,'pred']=1

    vote_results_file=os.path.join(os.getcwd(),'models',model_args['model_name'],'votes_'+task_string+'.csv')
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
    scores=vote_df['score_1'].tolist()

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

    start_time=time.time()

    model.to("cuda")

    model.train()

    def tokenize_function(data):
        return tokenizer(data["text"], max_length=512, padding="max_length", truncation=True)

    train_ds=Dataset.from_pandas(train)
    val_ds=Dataset.from_pandas(val)

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function,batched=True)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision,recall,_=precision_recall_curve(labels,preds)
        auprc=auc(recall,precision)
        return {"auprc": auprc}
 
    batch_size = model_args["batch_size"]

    output_dir=os.path.join(os.getcwd(),'models',model_args['model_name'])

    logging_steps = len(train_ds) // batch_size
    training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=3, learning_rate=2e-5, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True, metric_for_best_model="auprc", weight_decay=0.01, evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1, disable_tqdm=False, logging_steps=logging_steps, seed=42)

    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics,train_dataset=train_ds, eval_dataset=val_ds)

    trainer.train()

    model.eval()

    model_args['train_time']=time.time()-start_time
    
    shuffled_test_df=preprocess.shuffle(test,'text')
    
    model_args['mode']='unshuffled'
    evaluate(tokenizer,model,test,args,model_args)
    model_args['mode']='shuffled'
    shuffled_test_df=preprocess.shuffle(test,'text')
    evaluate(tokenizer,model,shuffled_test_df,args,model_args)

def prepare_dataset(args,dataset_args):
    dataset_file=os.path.join(os.getcwd(),'datasets', args.dataset+'_'+args.task+'.csv')

    if not os.path.isfile(dataset_file):
        create_dataset.create_dataset(args)
    
    dataset=pd.read_csv(dataset_file)

    train=dataset[dataset['fold_'+str(args.fold)]=='train']
    val=dataset[dataset['fold_'+str(args.fold)]=='val']
    test=dataset[dataset['fold_'+str(args.fold)]=='test']

    if args.debug_mode:
        train=train.head(50)
        val=val.head(50)
        test=test.head(50)

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
    
    train=train.reset_index(drop=True)
    val=val.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    train=train[['text','label']]
    val=val[['text','label']]
    test=test[['id','text','label']]
    
    return train,val,test