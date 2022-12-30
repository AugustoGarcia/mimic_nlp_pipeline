######################################################################################
# NLP Pipeline Experiment                                                            #
######################################################################################

# This document launches huggingface_pipeline.py to fine-tune and evaluate language models
# from Huggingface with a previously created dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import huggingface_pipeline

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--sentence_splits",
                        default=318,
                        type=int,
                        required=False,
                        help="Whether text should be split in n-word sentences, default 318 for clinicalbert. Large-input longformer will use 8*this value")

parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        required=False,
                        help="Train epochs")

parser.add_argument("--lr",
                        default=2e-5,
                        type=float,
                        required=False,
                        help="Train learning rate")

parser.add_argument("--class_balance", 
                        default=1, 
                        type=float, 
                        required=False,
                        help="Whether train and validation sets should be balanced, expressed as minority class ratio. Default 1 for 1:1")

parser.add_argument("--balance_all_datasets",
                        default=False,
                        type=bool,
                        required=False,
                        help="Whether to balance the train, validation and test sets instead of just the train set")

parser.add_argument("--balance_id",
                        default='subject_id',
                        choices=['subject_id','hadm_id','id'],
                        type=str,
                        required=False,
                        help="Choice of id for splitting, between 'subject_id' and 'hadm_id' (admission id). Default is subject id to prevent label leakage")

parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        required=False,
                        help="Batch size, default 20. Adjust to available VRAM")
    
parser.add_argument("--fold", 
                        default=0, 
                        type=int, 
                        required=False,
                        help="Fold number to retrieve from dataset. This allows batch submission of experiment folds")

parser.add_argument("--folds",
                        default=5,
                        type=int,
                        required=False,
                        help="Number of folds for cross-validation in case dataset is to be created")

parser.add_argument("--test_size", 
                        default=0.1, 
                        type=float, 
                        required=False,
                        help="Test and validation set size, in %. Validation will be an approximation")                        

parser.add_argument("--task",
                        default='',
                        choices=['readmission','mortality','both'],
                        type=str,
                        required=False,
                        help="Clinical task: readmission prediction (readmission), mortality prediction (mortality), both as a set of negative outcomes (both)")

parser.add_argument("--dataset",
                        default='early',
                        choices=['early','discharge','custom','presplit'],
                        type=str,
                        required=True,
                        help="Task for dataset, 72 hours of ICU notes (early), discharge summaries (discharge), or custom/pre-split dataset")  

parser.add_argument("--dataset_name",
                        default='custom',
                        type=str,
                        required=False,
                        help="Custom dataset name")  

parser.add_argument("--custom_task_name",
                        default='',
                        type=str,
                        required=False,
                        help="Custom task name")                        

parser.add_argument("--debug_mode",
                        default=False,
                        type=bool,
                        required=False,
                        help="Debug mode will downsize the train, test and validation sets to 50 rows")

parser.add_argument("--wandb",
                        default=True,
                        type=bool,
                        required=False,
                        help="Upload training results to wandb")

parser.add_argument("--save_logits",
                        default=False,
                        type=bool,
                        required=False,
                        help="Save model output logits and vote scores for test sets")
   
models=['longformer_512','clinicalbert','longformer_4096','longformer_4096_merge']

for model in models:
    parser.add_argument("--"+model,
                        default=False,
                        type=bool,
                        required=False,
                        help="Include "+model+" in pipeline")

args = parser.parse_args()

if args.dataset=='custom':
    args.task=args.custom_task_name

if args.longformer_512 or args.clinicalbert:    
    dataset_args = {
        "sentence_splits": args.sentence_splits,
        "merge_notes": False
    }
    train,val,test=huggingface_pipeline.prepare_dataset(args,dataset_args)
    if args.longformer_512:
        tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')
        model = AutoModelForSequenceClassification.from_pretrained('yikuan8/Clinical-Longformer', return_dict=True)
        model_args = {
            "model_name": "longformer_512",
            "max_length": 512,
            "batch_size": args.batch_size
        }
        huggingface_pipeline.train(tokenizer,model,args,model_args,train,val,test)

    if args.clinicalbert:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model_args = {
            "model_name": "clinicalbert_512",
            "max_length": 512,
            "batch_size": args.batch_size
        }
        huggingface_pipeline.train(tokenizer,model,args,model_args,train,val,test)

if args.longformer_4096:
    dataset_args = {
        "sentence_splits": int(8*args.sentence_splits),
        "merge_notes": False
    }
    train,val,test=huggingface_pipeline.prepare_dataset(args,dataset_args)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')
    model = AutoModelForSequenceClassification.from_pretrained('yikuan8/Clinical-Longformer', return_dict=True)
    model_args = {
        "model_name": "longformer_4096",
        "max_length": 4096,
        "batch_size":int(args.batch_size/8)
    }
    huggingface_pipeline.train(tokenizer,model,args,model_args,train,val,test)

if args.longformer_4096_merge:
    dataset_args = {
        "sentence_splits": int(8*args.sentence_splits),
        "merge_notes": True
    }
    train,val,test=huggingface_pipeline.prepare_dataset(args,dataset_args)
    tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer')
    model = AutoModelForSequenceClassification.from_pretrained('yikuan8/Clinical-Longformer', return_dict=True)
    model_args = {
        "model_name": "longformer_4096_merge",
        "max_length": 4096,
        "batch_size":int(args.batch_size/8)
    }
    huggingface_pipeline.train(tokenizer,model,args,model_args,train,val,test)
