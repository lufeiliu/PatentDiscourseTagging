from typing import Dict, List
import pandas as pd
import glob
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import KFold


def check_empty_label(labels_dict: Dict[str, List[str]]):
    return '' not in labels_dict.keys() and '' not in labels_dict.values()

def convert_to_labels_dict(Path_data):
    labels_dict = {}
    for file_ in glob.glob(Path_data) :
        labels = [sent.split("\t")[1].strip() for sent in open(file_).readlines()]
        fileName = file_.split("/")[3].split(".")[0]
        labels_dict[fileName] = labels
    if check_empty_label(labels_dict):
        return labels_dict

def flatten_labels_per_patent(labels_per_patent: Dict[str, List[str]]):
    patents_names = list(labels_per_patent.keys())
    patents_names.sort()
    
    labels_flattened = []
    for patent_name in patents_names:
        labels_flattened += labels_per_patent[patent_name]
    return labels_flattened

def compute_l2i_and_i2l(labels):
    l2i = {}
    i2l = {}
    for label in labels:
        if label not in l2i:
            idx = len(l2i.keys())
            l2i[label] = idx
            i2l[idx] = label
    return l2i, i2l

def load_input_data(path_document,encoded_label):
    df = pd.DataFrame({'text':str(), 'label':int()}, index = [])
    with open(path_document) as file_:
        for line in file_.readlines():
            text = line.split('\t')[0]
            label = line.split('\t')[1].strip()
            row_series = pd.Series((text,encoded_label[label]), index=["text","label"])
            df = pd.concat([df,row_series.to_frame().T],
                            ignore_index = True)
    return df

def check_sentence_length(Path_data,tokenizer,encoded_label):
    import statistics
    sentence_nb_length = {}
    sentence_long = []
    for path_document in glob.glob(Path_data):
        df = load_input_data(path_document,encoded_label)
        sentence_length = []
        for idx, row in df.iterrows() : 
            tokenizedSentence = tokenizer.tokenize(row['text'])
            if len(tokenizedSentence) > 512 :
                sentence_long.append((path_document,idx))
            sentence_length.append(len(tokenizedSentence))
        sentence_nb_length[path_document.split("/")[-1].split(".")[0]] = (len(df),
                                                                          round(statistics.mean(sentence_length),0),
                                                                          max(sentence_length))
    return sentence_nb_length,sentence_long


def split_for_cross_validation(Path_data,nb_fold):
    """
    split data into train and validation for k fold validation
    """
    folds = {}
    files = glob.glob(Path_data)
    kf = KFold(n_splits=nb_fold, shuffle=True, random_state=42)
    fold_id = 1
    for train_index, val_index in kf.split(files):
        train_docs = [files[i] for i in train_index]
        val_docs = [files[i] for i in val_index]
        folds[fold_id] = (train_docs,val_docs)
        fold_id += 1
    return folds


if __name__ == "__main__":
    Path_data = "../data/train/*.txt"
    labels_dict = convert_to_labels_dict(Path_data)
    l2i, i2l = compute_l2i_and_i2l(flatten_labels_per_patent(labels_dict))


    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentence_nb_length, long = check_sentence_length(Path_data,tokenizer,l2i)
    sorted = {key: sentence_nb_length[key] for key in sorted(sentence_nb_length)}
    print(sorted)
    print(long)