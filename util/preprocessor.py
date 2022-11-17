import pytorch_lightning as pl
import torch
import os
import re
import pandas as pd

from transformers import BertTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class Preprocessor(pl.LightningDataModule):

    def __init__(self, max_length=100, batch_size=100, dataset='dataset/indo_tweet_sentiment.csv'):
        super(Preprocessor, self).__init__()        
        self.id_label = {
            -1 : 0,
            0  : 1,
            1  : 2
        }

        dataset = pd.read_csv(dataset)
        dataset["sentimen"] = dataset["sentimen"].map(self.id_label)

        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.stemmer = StemmerFactory().create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def setup(self, stage=None):
        train_data, valid_data, test_data = self.preprocessor()
        if stage == "fit":
            self.train_data = train_data
            self.valid_data = valid_data
        elif stage == "predict":
            self.test_data = test_data

    def train_dataloader(self):
        sampler = RandomSampler(self.train_data)
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2
        )

    def val_dataloader(self):
        sampler = RandomSampler(self.valid_data)
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2
        )

    def predict_dataloader(self):
        sampler = SequentialSampler(self.test_data)
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2
        )

    def preprocessor(self):
        if os.path.exists("dataset/train.pt") and os.path.exists("dataset/valid.pt") and os.path.exists("dataset/test.pt"):
            print("\nLoading Data...")
            train_data = torch.load("dataset/train.pt")
            valid_data = torch.load("dataset/valid.pt")
            test_data = torch.load("dataset/test.pt")
            print('[ Loading Completed ]\n')
        
        else:
            print("\nPreprocessing Data...")
            train_data, valid_data, test_data = self.preprocessing_data(self.dataset)
            print('[ Preprocessing Completed ]\n')

        return train_data, valid_data, test_data

    def preprocessing_data(self, dataset):
        x_input_ids, x_token_type_ids, x_attention_mask, y = [], [], [], []

        for data in dataset.values.tolist():
            tweet = self.data_cleaning(str(data[1])) 
            label = data[0]

            binary_label = [0] * len(self.id_label)
            binary_label[label] = 1

            token = self.tokenizer(text=tweet,  
                                max_length=self.max_length, 
                                padding="max_length", 
                                truncation=True)  

            x_input_ids.append(token['input_ids'])
            x_token_type_ids.append(token['token_type_ids'])
            x_attention_mask.append(token['attention_mask'])
            y.append(binary_label)

        x_input_ids = torch.tensor(x_input_ids)
        x_token_type_ids = torch.tensor(x_token_type_ids)
        x_attention_mask = torch.tensor(x_attention_mask)
        y = torch.tensor(y)

        tensor_dataset = TensorDataset(x_input_ids, x_token_type_ids, x_attention_mask, y)

        train_valid_size = round(len(tensor_dataset) * 0.8)
        test_size = len(tensor_dataset) - train_valid_size

        train_valid_data, test_data = torch.utils.data.random_split(
            tensor_dataset, [
                train_valid_size, test_size
            ]
        )

        train_size = round(len(train_valid_data) * 0.9)
        valid_size = len(train_valid_data) - train_size

        train_data, valid_data = torch.utils.data.random_split(
            train_valid_data, [
                train_size, valid_size
            ]
        )

        torch.save(train_data, "dataset/train.pt")
        torch.save(valid_data, "dataset/valid.pt")
        torch.save(test_data, "dataset/test.pt")

        return train_data, valid_data, test_data

    def data_cleaning(self, string):
        string = string.lower()
        string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\n", "", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.strip()

        return self.stemmer.stem(string) 