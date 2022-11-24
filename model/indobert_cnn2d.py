import torch.nn as nn
import torch
import pytorch_lightning as pl
import random
import torch.nn.functional as F

from sklearn.metrics import classification_report
from transformers import BertModel

class IndoBERT_CNN2D(pl.LightningModule):

    def __init__(self, 
                n_out=3, 
                dropout=0.5, 
                lr=2e-5, 
                embedding_dim=768, 
                in_channels=8, 
                out_channels=24):

        super(IndoBERT_CNN2D, self).__init__()
        torch.manual_seed(1)
        random.seed(43)

        conv = 3
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), padding=(4,0))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv * out_channels, n_out)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(input_ids = input_ids, 
                            token_type_ids = token_type_ids, 
                            attention_mask = attention_mask)

        bert_hidden_state = bert_out[2]
        bert_hidden_state = torch.stack(bert_hidden_state, dim=1)
        bert_hidden_state = bert_hidden_state[:, -8:]

        x = [
            F.relu(self.conv1(bert_hidden_state).squeeze(3)),
            F.relu(self.conv2(bert_hidden_state).squeeze(3)),
            F.relu(self.conv3(bert_hidden_state).squeeze(3))
        ]

        x = [ F.max_pool1d(i, i.size(2)).squeeze(2) for i in x ]
        x = torch.cat(x, dim = 1) 
        x = self.dropout(x)
        logits = self.classifier(x)
        logits = self.sigmoid(logits)
        
        return logits

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

        return 