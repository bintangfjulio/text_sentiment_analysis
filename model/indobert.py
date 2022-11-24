import torch.nn as nn
import torch
import pytorch_lightning as pl
import random

from sklearn.metrics import classification_report
from transformers import BertModel

class IndoBERT(pl.LightningModule):
    def __init__(self, 
                n_out=2, 
                dropout=0.5, # 0.3
                lr=2e-5):

        super(IndoBERT, self).__init__()

        torch.manual_seed(1)
        random.seed(43)

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_out)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        bert_last_hiddenstate = bert_out[0]
        pooler = bert_last_hiddenstate[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.tanh(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = train_batch

        out = self(input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                token_type_ids=x_token_type_ids)

        loss = self.criterion(out.cpu(), target=y.float().cpu())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.log_dict({'train_loss': loss, 'train_acc': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                token_type_ids=x_token_type_ids)
        
        loss = self.criterion(out.cpu(), target=y.float().cpu())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.log_dict({'val_loss': loss, 'val_acc': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch

        out = self(input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                token_type_ids=x_token_type_ids)

        loss = self.criterion(out.cpu(), target=y.float().cpu())
        
        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        report = classification_report(true, pred, output_dict = True, zero_division = 0)
        self.log_dict({'test_loss': loss, 'test_acc': report["accuracy"]}, prog_bar=True, on_epoch=True)

        return loss
