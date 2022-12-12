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
                        help="Whether text should be split in n-word sentences, default 318")

parser.add_argument("--class_balance", 
                        default=1, 
                        type=float, 
                        required=False,
                        help="Whether train and validation sets should be balanced, default 0.33")

parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        required=False,
                        help="Batch size, default 16")
    
parser.add_argument("--fold", 
                        default=0, 
                        type=int, 
                        required=False,
                        help="Fold number to retrieve from dataset")

parser.add_argument("--folds",
                        default=5,
                        type=int,
                        required=False,
                        help="Number of folds for Cross-validation")

parser.add_argument("--test_size", 
                        default=0.1, 
                        type=float, 
                        required=False,
                        help="Test set size")                        

parser.add_argument("--task",
                        default='readmission',
                        choices=['readmission','mortality','both'],
                        type=str,
                        required=True,
                        help="Clinical task: readmission prediction (readmission) or mortality prediction (mortality)")

parser.add_argument("--dataset",
                        default='early',
                        choices=['early','discharge'],
                        type=str,
                        required=True,
                        help="Task for dataset, 72 hours of ICU notes (early) or discharge summaries (discharge)")                        

parser.add_argument("--debug_mode",
                        default=False,
                        type=bool,
                        required=False,
                        help="Debug mode will downsize the train, test and validation sets to 50 rows")
   
models=['longformer_512','clinicalbert','longformer_4096','longformer_4096_merge']

for model in models:
    parser.add_argument("--"+model,
                        default=False,
                        type=bool,
                        required=False,
                        help="Include "+model+" in pipeline")

args = parser.parse_args()

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