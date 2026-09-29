"""Main script for UDAEE."""

import param
from train import pretrain, adapt, evaluate, evaluate_disc
from model import (BertEncoder, DistilBertEncoder, DistilRobertaEncoder,
                   BertClassifier, Discriminator, RobertaEncoder, RobertaClassifier, EarlyBertEncoder, EarlyBertClassifier, EarlyRoBertaEncoder, EarlyRoBertaClassifier)
from utils import XML2Array, CSV2Array, CSV2Array_yelp, convert_examples_to_features, \
    roberta_convert_examples_to_features, get_data_loader, init_model,load_squad_data, extract_src_x_y
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, RobertaTokenizer
import torch
import os
import random
import argparse
import pandas as pd
import pickle
from collections import Counter
import csv
import json

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb", "SST2","full_imdb","RTE","QNLI","QQP","SQuAD","mrpc","scitail", 'yelp'],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd",
                        choices=["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb","SST2","full_imdb","RTE","QNLI","QQP","SQuAD","mrpc","scitail", 'yelp'],
                        help="Specify tgt dataset")

    parser.add_argument('--pretrain', default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--dctrain', default=False, action='store_true',
                        help='Force to dctrain target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="exit_bert",
                        choices=["bert", "distilbert", "roberta", "distilroberta", "exit_bert", "early_roberta"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=0.7,
                        help="Specify adversarial weight")

    parser.add_argument('--beta', type=float, default=0.1,
                        help="Specify KD loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=20,
                        help="Specify batch size")

    parser.add_argument('--pre_epochs', type=int, default = 2,
                        help="Specify the number of epochs for pretrain")

    parser.add_argument('--pre_log_step', type=int, default=1,
                        help="Specify log step size for pretrain")

    parser.add_argument('--num_epochs', type=int, default=1,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=2,
                        help="Specify log step size for adaptation")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_arguments()
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("pre_epochs: " + str(args.pre_epochs))
    print("num_epochs: " + str(args.num_epochs))
    print("alpha: " + str(args.alpha))
    print("beta: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)

    if args.model in ['roberta', 'distilroberta', 'early_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    # preprocess data
    print("=== Processing datasets ===")
    if args.src in ['RTE']:
        rte_path = os.path.join('data', args.src, 'train.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        nan_count = rte_data['label'].isna().sum()

        rte_data = rte_data.dropna(subset=['label'])
        dropped_rows = nan_count

        src_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        ################################################################################

        rte_path = os.path.join('data', args.src, 'dev.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        nan_count = rte_data['label'].isna().sum()

        rte_data = rte_data.dropna(subset=['label'])
        dropped_rows = nan_count

        src_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_test_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

    elif args.src in ['QNLI']:
        rte_path = os.path.join('data', args.src, 'train.tsv')

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        src_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        src_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        #################################################################################

        rte_path = os.path.join('data', args.src, 'dev.tsv')

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        src_test_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        src_test_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

    elif args.src in ['QQP']:
        rte_path = os.path.join('data', args.src, 'train.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        src_x = (rte_data['question1'] + " [SEP] " + rte_data['question2']).tolist()
        src_y = rte_data['is_duplicate'].tolist()

        #############################################################################

        rte_path = os.path.join('data', args.src, 'dev.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        src_test_x = (rte_data['question1'] + " [SEP] " + rte_data['question2']).tolist()
        src_test_y = rte_data['is_duplicate'].tolist()

    elif args.src in ["SQuAD"]:
        train_path = os.path.join('data', args.src,'train-v2.0.json')
        dev_path = os.path.join('data', args.src,'dev-v2.0.json')

        train_data = load_squad_data(train_path)
        dev_data = load_squad_data(dev_path)

        src_x, src_y = extract_src_x_y(train_data)
        src_test_x, src_test_y = extract_src_x_y(dev_data)

    elif args.src in ['mrpc']:
        rte_path = os.path.join('data', args.src, 'mrpc_train.csv')
        rte_data = pd.read_csv(rte_path)

        src_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_y = rte_data['label'].tolist()

        ################################################################################

        rte_path = os.path.join('data', args.src, 'mrpc_validation.csv')
        rte_data = pd.read_csv(rte_path)

        src_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_test_y = rte_data['label'].tolist()

    elif args.src in ['scitail']:
        rte_path = os.path.join('data', args.src, 'scitail_train.csv')
        rte_data = pd.read_csv(rte_path)

        src_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_y = rte_data['gold_label'].map({'entailment': 1, 'neutral': 0}).tolist()

        ################################################################################

        rte_path = os.path.join('data', args.src, 'scitail_validation.csv')
        rte_data = pd.read_csv(rte_path)

        src_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        src_test_y = rte_data['gold_label'].map({'entailment': 1, 'neutral': 0}).tolist()


    elif args.src in ['blog', 'airline', 'imdb',"SST2","full_imdb"]:
        src_x, src_y = CSV2Array(os.path.join('data', args.src, args.src 
                                              + '.csv'))
    elif args.src in ['yelp']:
        src_x, src_y = CSV2Array_yelp(os.path.join('data', args.src, args.src 
                                              + '.csv'))
    else:
        src_x, src_y = XML2Array(os.path.join('data', args.src, 'negative.review'),
                               os.path.join('data', args.src, 'positive.review'))
    
    if args.src in ["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb", "SST2","full_imdb", 'yelp']:
        src_x, src_test_x, src_y, src_test_y = train_test_split(src_x, src_y,
                                                            test_size=0.25,
                                                            stratify=src_y,
                                                            random_state=args.seed)
    
    if args.tgt in ['RTE']:
        rte_path = os.path.join('data', args.tgt, 'train.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')

        nan_count = rte_data['label'].isna().sum()

        rte_data = rte_data.dropna(subset=['label'])

        dropped_rows = nan_count

        tgt_train_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()

        tgt_train_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        ####################################################################################

        rte_path = os.path.join('data', args.tgt, 'dev.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')

        nan_count = rte_data['label'].isna().sum()

        rte_data = rte_data.dropna(subset=['label'])

        dropped_rows = nan_count

        tgt_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()

        tgt_test_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y

    elif args.tgt in ['QNLI']:
        rte_path = os.path.join('data', args.tgt, 'train.tsv')

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        tgt_train_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        tgt_train_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        ###################################################################################

        rte_path = os.path.join('data', args.tgt, 'dev.tsv')

        try:
            rte_data = pd.read_csv(rte_path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            rte_data = pd.read_csv(rte_path, sep='\t', on_bad_lines='skip')

        tgt_test_x = (rte_data['question'] + " [SEP] " + rte_data['sentence']).tolist()
        tgt_test_y = rte_data['label'].map({'not_entailment': 0, 'entailment': 1}).tolist()

        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y

    elif args.tgt in ['QQP']:
        rte_path = os.path.join('data', args.tgt, 'train.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        tgt_train_x = (rte_data['question1'] + " [SEP] " + rte_data['question2']).tolist()
        tgt_train_y = rte_data['is_duplicate'].tolist()

        #########################################################################333

        rte_path = os.path.join('data', args.tgt, 'dev.tsv')
        rte_data = pd.read_csv(rte_path, sep='\t')
        tgt_test_x = (rte_data['question1'] + " [SEP] " + rte_data['question2']).tolist()
        tgt_test_y = rte_data['is_duplicate'].tolist()

        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y
    
    elif args.tgt in ["SQuAD"]:
        train_path = os.path.join('data', args.tgt,'train-v2.0.json')
        dev_path = os.path.join('data', args.tgt,'dev-v2.0.json')

        train_data = load_squad_data(train_path)
        dev_data = load_squad_data(dev_path)

        tgt_train_x, tgt_train_y = extract_src_x_y(train_data)
        tgt_test_x, tgt_test_y = extract_src_x_y(dev_data)
        
        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y

    elif args.tgt in ['mrpc']:
        rte_path = os.path.join('data', args.tgt, 'mrpc_train.csv')
        rte_data = pd.read_csv(rte_path)

        tgt_train_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        tgt_train_y = rte_data['label'].tolist()

        ################################################################################

        rte_path = os.path.join('data', args.tgt, 'mrpc_validation.csv')
        rte_data = pd.read_csv(rte_path)

        tgt_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        tgt_test_y = rte_data['label'].tolist()

        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y

    elif args.tgt in ['scitail']:
        rte_path = os.path.join('data', args.tgt, 'scitail_train.csv')
        rte_data = pd.read_csv(rte_path)

        tgt_train_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        tgt_train_y = rte_data['gold_label'].map({'entailment': 1, 'neutral': 0}).tolist()

        ################################################################################

        rte_path = os.path.join('data', args.tgt, 'scitail_validation.csv')
        rte_data = pd.read_csv(rte_path)

        tgt_test_x = (rte_data['sentence1'] + " [SEP] " + rte_data['sentence2']).tolist()
        tgt_test_y = rte_data['gold_label'].map({'entailment': 1, 'neutral': 0}).tolist()

        tgt_x = tgt_train_x + tgt_test_x
        tgt_y = tgt_train_y + tgt_test_y

    elif args.tgt in ['blog', 'airline', 'imdb',"SST2", "full_imdb"]:
        tgt_x, tgt_y = CSV2Array(os.path.join('data', args.tgt, args.tgt + '.csv'))
    elif args.src and args.tgt in ['yelp']:
        tgt_x, tgt_y = CSV2Array_yelp(os.path.join('data', args.tgt 
                                              + '/test.csv'))
    elif args.tgt in ['yelp']:
        tgt_x, tgt_y = CSV2Array_yelp(os.path.join('data', args.tgt, args.tgt 
                                              + '.csv'))
                                        
    else:
        tgt_x, tgt_y = XML2Array(os.path.join('data', args.tgt, 'negative.review'),
                                 os.path.join('data', args.tgt, 'positive.review'))


    if args.tgt in ["books", "dvd", "electronics", "kitchen", "blog", "airline", "imdb", "SST2","full_imdb", 'yelp']:
        tgt_train_x, tgt_test_x, tgt_train_y, tgt_test_y = train_test_split(tgt_x, tgt_y,
                                                                        test_size=0.2,
                                                                        stratify=tgt_y,
                                                                        random_state=args.seed)


    if args.model in ['roberta', 'distilroberta', 'early_roberta']:
        src_features = roberta_convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer) 
        src_test_features = roberta_convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = roberta_convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        tgt_train_features = roberta_convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
        tgt_test_features = convert_examples_to_features(tgt_test_x, tgt_test_y, args.max_seq_length, tokenizer)
    else:
        src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
        src_test_features = convert_examples_to_features(src_test_x, src_test_y, args.max_seq_length, tokenizer)
        tgt_features = convert_examples_to_features(tgt_x, tgt_y, args.max_seq_length, tokenizer)
        tgt_train_features = convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
        tgt_test_features = convert_examples_to_features(tgt_test_x, tgt_test_y, args.max_seq_length, tokenizer) 

    src_data_loader = get_data_loader(src_features, args.batch_size)   
    deferal_label_data_loader = get_data_loader(src_features, 1)
    src_val_data_loader = get_data_loader(src_features, 1)
    src_data_eval_loader = get_data_loader(src_test_features, 1)    
    tgt_data_train_loader = get_data_loader(tgt_train_features, args.batch_size)

    tgt_data_train_val_loader = get_data_loader(tgt_train_features, 1)
    tgt_data_all_v_loader = get_data_loader(tgt_features, args.batch_size)
    tgt_data_all_loader = get_data_loader(tgt_features, 1)
    tgt_data_test_loader = get_data_loader(tgt_test_features, 1) 

    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'distilbert':
        src_encoder = DistilBertEncoder()
        src_classifier = BertClassifier()
    elif args.model == 'roberta':
        src_encoder = RobertaEncoder()
        src_classifier = RobertaClassifier()
    elif args.model == 'exit_bert':
        src_encoder = EarlyBertEncoder()
        src_classifier = EarlyBertClassifier()
    elif args.model == 'early_roberta':
        src_encoder = EarlyRoBertaEncoder()
        src_classifier = EarlyRoBertaClassifier()
    else:
        src_encoder = DistilRobertaEncoder()
        src_classifier = RobertaClassifier()
    discriminator = Discriminator()

    if args.load:
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path)
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path)
        discriminator = init_model(args, discriminator, restore=param.d_model_path)
    else:
        src_encoder = init_model(args, src_encoder)
        src_classifier = init_model(args, src_classifier)
        discriminator = init_model(args, discriminator)

    # train source model
    print("=== Training classifier for source domain ===")
    if args.pretrain:
        src_encoder, src_classifier = pretrain(
            args, src_encoder, src_classifier, src_data_loader)

    print("source train dataset making deferal label")
    evaluate(args,src_encoder, src_classifier, deferal_label_data_loader)
         
    deferal_labels_df = pd.read_csv('deferal_label.csv')

    deferal_labels_df['src_x'] = src_x

    deferal_labels_df = deferal_labels_df.sort_values(by=['mean_of_trueclass_prob(confidence)', 'variance_of_prob'], ascending=[True,True ])
    
    # Save the updated DataFrame to a new CSV file 
    deferal_labels_df.to_csv('deferal_label_updated.csv', index=False)


    deferal_labels = deferal_labels_df['deferal_label'].values

    if args.model in ['roberta', 'distilroberta', 'early_roberta']:
        src_features_def = convert_examples_to_features(src_x, deferal_labels, args.max_seq_length, tokenizer)
    else:
        src_features_def = convert_examples_to_features(src_x, deferal_labels, args.max_seq_length, tokenizer)
    
    src_data_loader_def = get_data_loader(src_features_def, args.batch_size)
    
    for params in src_encoder.parameters():
        params.requires_grad = False

    for params in src_classifier.parameters():
        params.requires_grad = False

    print("=== Training encoder for target domain ===")
    if args.dctrain:
        discriminator = adapt(args, src_encoder, discriminator,
                            src_classifier, src_data_loader_def, tgt_data_train_loader, tgt_data_all_loader)

    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    result_def=[]
    p=[]
    for i in [2.0]:
        for j in [2.0]:
            print("defervalue_threshold,confidence_threshold",[i,j])           
            if args.src != args.tgt:  
                acc,def_rate, speedup, risk, coverage = evaluate_disc(src_encoder, src_classifier, tgt_data_all_loader,discriminator,h=i,k=j)
            else:
                acc,def_rate, speedup, risk, coverage = evaluate_disc(src_encoder, src_classifier, tgt_data_test_loader,discriminator,h=i,k=j)

            p.append((acc, def_rate, speedup, risk, coverage))
            result_def.append({
                'alpha' : args.alpha,
                'beta' :  args.beta,
                'accuracy' :acc,
                'defer_sample' : def_rate,
                'dis_value' : i,
                'confidence' : j
            })

    result_def_df= pd.DataFrame(result_def)
    result_def_df.to_csv('result_def_.csv', index = False)

    print("list of acc:",p)

if __name__ == '__main__':
    main()
