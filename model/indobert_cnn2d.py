import torch.nn as nn
import random
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics import AUROC
from transformers import BertModel

class IndoBERT_CNN2D(pl.LightningModule):

    def __init__(self, labels=['Neutral', 'Positive', 'Negative'], n_out=3, dropout=0.3, lr=1e-5, embedding_dim=768, in_channels=8, out_channels=24):
        super(IndoBERT_CNN2D, self).__init__()

        torch.manual_seed(1)
        random.seed(43)

        conv = 3
        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased', output_hidden_states = True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), groups=4)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), groups=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), groups=4)

        self.labels = labels
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.classifier = nn.Linear(conv * out_channels, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        hidden_state = bert_out[2]
        hidden_state = torch.stack(hidden_state, dim = 1)
        hidden_state = hidden_state[:, -8:]

        x = [
            F.relu(self.conv1(hidden_state).squeeze(3)),
            F.relu(self.conv2(hidden_state).squeeze(3)),
            F.relu(self.conv3(hidden_state).squeeze(3))
        ]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        
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

        loss = self.criterion(out.cpu(), y.float().cpu())
        self.log("train_loss", loss)

        return {"loss": loss, "predictions": out, "labels": y}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = valid_batch

        out = self(input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                token_type_ids=x_token_type_ids)
        
        loss = self.criterion(out.cpu(), y.float().cpu())
        self.log("val_loss", loss)

        return {"val_loss": loss}

    def predict_step(self, test_batch, batch_idx):
        x_input_ids, x_token_type_ids, x_attention_mask, y = test_batch

        out = self(input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                token_type_ids=x_token_type_ids)

        return out

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for outputs in outputs:
            for out_lbl in outputs["labels"].detach().cpu():
                labels.append(out_lbl)
            for out_pred in outputs["predictions"].detach().cpu():
                predictions.append(out_pred)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(self.labels):
            auroc = AUROC(num_classes = len(self.labels))
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            print(f"{name} \t : {class_roc_auc}")
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)