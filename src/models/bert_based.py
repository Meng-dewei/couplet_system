import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import config as global_config

class BertCoupletModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_config["model_name"])
        self.tokenizer = BertTokenizer.from_pretrained(model_config["model_name"])
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.device = global_config.DEVICE

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

    # 重构 generate 方法，兼容参数（忽略不支持的参数）
    def generate(self, input_ids, max_length=32, eos_token_id=None, pad_token_id=None,
                 do_sample=True, top_k=30, no_repeat_ngram_size=2, repetition_penalty=1.5):
        # 增加随机性，降低重复惩罚的强度
        if eos_token_id is None:
            eos_token_id = self.tokenizer.sep_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        generated = input_ids.clone()
        attention_mask = (generated != pad_token_id).to(self.device)
        input_len = input_ids.size(1)
        min_length = input_len + 2

        # 记录已生成的ngram，避免重复
        generated_ngrams = {}

        for _ in range(max_length - input_len):
            outputs = self.forward(generated, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]

            # 降低特殊符号概率（适度调整）
            special_tokens = [self.tokenizer.pad_token_id, self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
            next_token_logits[:, special_tokens] -= 50  # 减小惩罚力度

            # 实现ngram重复控制
            if no_repeat_ngram_size is not None and generated.size(1) >= no_repeat_ngram_size:
                ngram = tuple(generated[0, -no_repeat_ngram_size + 1:].tolist())
                if ngram in generated_ngrams:
                    for token_id in generated_ngrams[ngram]:
                        next_token_logits[0, token_id] = -float('inf')

            # 采样逻辑（增加随机性）
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                # 使用温度调整概率分布
                temperature = 0.8
                probs = probs ** (1 / temperature)
                probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 更新ngram记录
            if no_repeat_ngram_size is not None and generated.size(1) >= no_repeat_ngram_size - 1:
                current_ngram = tuple(generated[0, -no_repeat_ngram_size + 2:].tolist() + [next_token.item()])
                if current_ngram not in generated_ngrams:
                    generated_ngrams[current_ngram] = set()
                generated_ngrams[current_ngram].add(next_token.item())

            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            if next_token.item() == eos_token_id and generated.size(1) >= min_length:
                break

        return generated