import glob
import torch
from torch import nn
from torch.utils.data import DataLoader

import transformers
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer 
from transformers import BertPreTrainedModel, BertModel

import os
import time
import numpy as np
import re
from tqdm.auto import tqdm

from util import *

class Patent_BERT(BertPreTrainedModel):
    """ Transformer model class with custom output layer for fine-tuning.
    """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.projection = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(config.hidden_size, 16))            
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #print("last :" , outputs.last_hidden_state.shape)
        logits = self.projection(outputs.last_hidden_state) # torch.Size([batch_size, sequence_length, label_numbers])
        #print("output shape :", logits.shape)
        #print("output example :", logits[0])
        return logits



class Trainer_Patent(Trainer):
    """ Class inheriting from Trainer to configure loss function used.
    """

    def __init__(self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        global_target_sentence_index = None,
        max_paragraph_length = None
        ):
        super().__init__(model, args, data_collator, train_dataset, 
                         eval_dataset, tokenizer, model_init, compute_metrics, 
                         callbacks, optimizers)
        self.global_target_sentence_index = global_target_sentence_index
        self.max_paragraph_length = max_paragraph_length
        

    def compute_loss(self, model, inputs,return_outputs=False):
        #print("input labels :",inputs["labels"]) 
        labels = inputs.pop('labels') # input shape : [8,3]
        #print("labels :",labels)
        sep_positions = inputs.pop('sep_positions') # take all sep positions
        #print("sep_position :", sep_positions)
        outputs = model(**inputs) 
        batch_preds = []
        #print("outputshape :",outputs.shape)
        for i, sentence_sep_positions in zip(range(outputs.shape[0]), sep_positions): # 8 phrases, 8 list de position SEP 
          sentence_preds = []
          for j in sentence_sep_positions: # parcour chacune des 8 phrases 
            sentence_preds.append(outputs[i,j]) # logit de chaque SEP token
          batch_preds.append(torch.cat(sentence_preds[:-1])) #ignore last sep token as we don't predict from it (a list of tensor of size 48)
        
        logging.info(f"batch prediction size : {batch_preds[0].shape}")
        logging.info(f"labels shape : {labels.shape}")
        #preds = torch.cat(batch_preds).reshape((-1, self.max_paragraph_length, 16))
        preds = torch.cat(batch_preds).view(-1,16)
        labels = labels.view(-1)
        logging.info(f"resize batch prediction size : {preds.shape}")
        logging.info(f"resize labels shape : {labels.shape}")

        # compute loss 
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(preds, labels)

        logging.info(f"loss : {loss} loss shape : {loss.shape}")

        if return_outputs: 
            return (loss, (loss, outputs)) 
        else:
            return loss


class PatentDataset(torch.utils.data.Dataset):
    """Dataset class inheriting from pytorch to be used by dataloaders.
    """
    def __init__(self, tokenizer, input_set, input_context, max_paragraph_length, global_target_sentence_index):
        self.tokenizer = tokenizer
        self.texts = input_set['texts']
        self.contexts = input_context
        self.max_paragraph_length = max_paragraph_length
        self.global_target_sentence_index = global_target_sentence_index
        
        
    def collate_fn(self, batch):
        texts = [b['text'] for b in batch]
        contexts = [b['context'] for b in batch]
        encodings, sep_positions = self.custom_tokenizer(batch = contexts) 
        encodings['sep_positions'] = sep_positions
        return encodings
    
    def custom_collate_fn(self, batch):
      texts = [b['text'] for b in batch]
      contexts = [b['context'] for b in batch]
      return {'texts':texts, 'contexts':contexts}
    

    def custom_tokenizer(self, batch):
      """ Utility functions to tokenize a list of sentences using [SEP] at the beginning of each sentence with fixed positions.
      """
      batch_sequences = []
      batch_sep_positions = []
      batch_token_type_ids = []
      for sequence_list in batch:
        augmented_sequence = ''
        for sentence in sequence_list:
            augmented_sequence += '[SEP]' + sentence
        augmented_sequence.strip()
        batch_sequences.append(augmented_sequence)
      encoded_batch = self.tokenizer(batch_sequences, padding='longest', truncation=True, max_length=512, return_tensors='pt')
      for encoded_sequence in encoded_batch['input_ids']:
        sep_positions = [index for index in range(len(encoded_sequence)) if encoded_sequence[index]==3]
        while len(sep_positions) < self.max_paragraph_length + 1: # repeat last sep position to get full sequence 
            sep_positions.append(sep_positions[-1])
        batch_sep_positions.append(sep_positions)
        if self.global_target_sentence_index in range(len(sep_positions)-1): # check to see that the target sentence is actually part of the context
            custom_token_type_ids = [ 1 if index in range(sep_positions[self.global_target_sentence_index], sep_positions[self.global_target_sentence_index+1]+1) else 0 for index in range(len(encoded_sequence))]
        else:
            custom_token_type_ids = [1 for index in range(len(encoded_sequence))]
        batch_token_type_ids.append(custom_token_type_ids)
       
      encoded_batch['token_type_ids'] = torch.tensor(batch_token_type_ids)
      return encoded_batch, batch_sep_positions
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'text': self.texts[idx],
                'context': self.contexts[idx],
                }
        return item


def model_predict(model, tokenizer, dataloader, device, global_target_sentence_index, max_paragraph_length):
    """ Utility function to set the model to GPU and infer of given dataloader.
    """
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch_sequences = []
            batch_sep_positions = []
            batch_token_type_ids = []
            for sequence_list in batch['contexts']:
              augmented_sequence = ''
              for sentence in sequence_list:
                  augmented_sequence += '[SEP]' + sentence
              augmented_sequence.strip()
              batch_sequences.append(augmented_sequence)
            encoded_batch = tokenizer(batch_sequences, padding='longest', truncation=True, max_length=512, return_tensors='pt').to(device)
            for encoded_sequence in encoded_batch['input_ids']:
                sep_positions = [index for index in range(len(encoded_sequence)) if encoded_sequence[index]==3]
                while len(sep_positions) < max_paragraph_length + 1: # repeat last sep position to get full sequence 
                  sep_positions.append(sep_positions[-1])
                batch_sep_positions.append(sep_positions)
                if global_target_sentence_index in range(len(sep_positions)-1): # check to see that the target sentence is actually part of the context
                  custom_token_type_ids = [ 1 if index in range(sep_positions[global_target_sentence_index], sep_positions[global_target_sentence_index+1]+1) else 0 for index in range(len(encoded_sequence))]
                else:
                  custom_token_type_ids = [1 for index in range(len(encoded_sequence))]
                batch_token_type_ids.append(custom_token_type_ids)
            
            encoded_batch['token_type_ids'] = torch.tensor(batch_token_type_ids).to(device)
            output = model(**encoded_batch)             
            for i, sentence_sep_positions in zip(range(output.shape[0]), batch_sep_positions):
              sentence_preds = []
              for j in sentence_sep_positions:
                sentence_preds.append(output[i,j])
              preds = torch.cat(sentence_preds).reshape((-1, 16)).to('cpu').numpy()
              predicted_tags = np.argmax(preds[global_target_sentence_index]).flatten()
              predictions.append(predicted_tags.item())

            #print(f"TEST PREDICTION : {predictions}")

    return predictions


def createPredictionDataset(Path_data):
    """Convert documents to be annotated into one dataset 
        input : path to folder containing patents to be annotated (each document should be segmented into sentences)
        output : list of dict, each dict represent a list of sentences in one document
        @TODO : integrate sentence split function
    """
    dataset = []
    for file_ in glob.glob(Path_data) :
        fileName = file_.split("/")[-1].split(".")[0]
        patent_sentences = []
        for sent in open(file_).readlines():
            if sent.strip():
                sent_to_dict = {}
                sent_to_dict["text"] = sent.strip()
                patent_sentences.append(sent_to_dict)
        dataset.append({"tokenised_sentences_"+fileName:patent_sentences})
    return dataset


def filter_dataset(input_dataset):
    """ 
    Utility function to convert input dataset into custom data structure
    params: input_dataset: list of deserialised jsons, l2i : label_id dictionary
    returns: dictionary with the following structure:
            key:  sentence_global_idx value , value:list of dictionaries
                dict0: key: text, value: string
                dict1: key: sentence_length, value: int  
    """
    sentence_global_idx = 0
    re_punctuation_string = '[“”|()%&\s,_:;/\'!?-]'
    texts = []
    positions = []
    sentence_length = []

    #no_sentences_per_patent = len(input_dataset['tokenised_sentences'])
    allTexts = input_dataset.get(list(input_dataset.keys())[0], [])
    no_sentences_per_patent = len(allTexts)
    for sentence_no in range(no_sentences_per_patent):
        #tokenized_sentence = re.split(re_punctuation_string, input_dataset['tokenised_sentences'][sentence_no]['text']) # remove punct
        tokenized_sentence = re.split(re_punctuation_string, allTexts[sentence_no]['text']) # remove punct
        tokenized_sentence = list(filter(None, tokenized_sentence)) # remove punct
        if (len(tokenized_sentence) == 0): # if sentence is empty          
            continue
        else:
            texts.append(' '.join([elem for elem in tokenized_sentence]))
            positions.append(sentence_global_idx)
            sentence_length.append(len(tokenized_sentence))
            sentence_global_idx +=1

    actual_data_dict = {'texts':texts,'positions':positions,'sentence_length':sentence_length}

    return actual_data_dict


def context_builder(data_dict, left_context_size = 0, right_context_size = 0): #{'texts':texts,'positions':positions,'sentence_length':sentence_length}
    context = []
    for sentence_index in range(len(data_dict['texts'])): # iterate over sentences in the doc 
        sentence_context = []
        if (sentence_index - left_context_size >= 0) and (sentence_index + right_context_size < len(data_dict['texts'])):
            # test if target sentence is in the middle of the corpus
            for context_index in range(sentence_index - left_context_size, sentence_index + right_context_size + 1):
                sentence_context.append(data_dict['texts'][context_index])
        elif sentence_index - left_context_size >= 0: #if target sentence is at end of the corpus 
            for context_index in range(sentence_index - left_context_size, sentence_index + right_context_size + 1):
                if context_index < len(data_dict['texts']): # add in a smaller context window at end of the corpus
                    sentence_context.append(data_dict['texts'][context_index])
        elif sentence_index + right_context_size < len(data_dict['texts']): #if target sentence is at beginning of the corpus 
                for context_index in range(sentence_index - left_context_size, sentence_index + right_context_size + 1):
                    if context_index >= 0: # add in smaller context window at the beginning of the corpus
                        sentence_context.append(data_dict['texts'][context_index])
        context.append(sentence_context)
        while len(sentence_context) < (1 + left_context_size + right_context_size): # pad with 0 labels for senteces with a smaller context eg. beginning/end of docs
          sentence_context.append("")
    return context


Path_data = "../../data/data_16/train/*.txt"
text_label_dict = convert_to_labels(Path_data)
labels_dict = {k: v["labels"] for k, v in text_label_dict.items()}
l2i, i2l = compute_l2i_and_i2l(flatten_labels_per_patent(labels_dict))
l2i


# Setting random seed and device
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)
print(use_cuda)


# Prepare test data 

Path_prediction_data = "../../data/epo1000Splitted/*.txt"
testing_dataset = createPredictionDataset(Path_prediction_data)
print(testing_dataset)
print(f"TESTING DOCUMENTS : {len(testing_dataset)}")   



# Initialize tokenizer
from transformers import AutoTokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents" )
left_size = 1
right_size = 1
max_paragraph_length = left_size+right_size+1
global_target_sentence_index = 1 #target sentence index to be used throughout entire script 
patent_model = Patent_BERT.from_pretrained('../../models/bert_context_11_16_v4/') 

for doc in testing_dataset:
    test_data = filter_dataset(doc)
    print(f"###################################")
    print(f"PROCESSING THE {testing_dataset.index(doc)} PATENT OF DATASET")
    print(f"DOCUMENT CONTAINS {len(test_data['texts'])} SENTENCES")
    assert(len(test_data['texts']) == len(test_data['positions']) == len(test_data['sentence_length']) )
    print(f"EXAMPLE : {test_data['texts'][0]},{test_data['positions'][0]},{test_data['sentence_length'][0]}")

    test_context = context_builder(test_data, left_context_size = left_size, right_context_size = right_size)
    test_dataset = PatentDataset(tokenizer, test_data, test_context, max_paragraph_length = max_paragraph_length, global_target_sentence_index = global_target_sentence_index)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn = test_dataset.custom_collate_fn)

    # Perform predictions on test dataset
    start_time = time.time()
    test_predictions_list = model_predict(patent_model, tokenizer, test_loader, device,  global_target_sentence_index = global_target_sentence_index, max_paragraph_length = max_paragraph_length)
    end_time = time.time()
    print(f'PREDICTION OF DOC {testing_dataset.index(doc)} TOOK {end_time-start_time} SECONDS')
    print(f"Predictions : {test_predictions_list}")
    assert(len(test_predictions_list) == len(test_data['texts']))
    print(f"FINISHING PREDICTION OF {len(test_predictions_list) } SENTENCES")
    outName = list(doc.keys())[0].split("-")[-1]
    print(f"WRITING TO FILE {outName}")
    with open("../../data/epo1000Annotated/"+outName+".txt","w") as fileOut:
        for text,pred in zip(test_data['texts'],test_predictions_list):
            predictedLabel = i2l[pred]
            fileOut.write(text+"\t"+predictedLabel.strip()+"\n")
    print(f"END OF DOCUMENT {testing_dataset.index(doc)}")
    print(f"###################################")

