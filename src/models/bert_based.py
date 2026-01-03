import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import config


class BertCoupletModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["model_name"])
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits