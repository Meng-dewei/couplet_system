import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer  # 关键：改用AutoTokenizer
import config as global_config


class GPT2CoupletModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # 加载中文GPT2模型（自动适配tokenizer）
        self.model_name = model_config["model_name"]
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            ignore_mismatched_sizes=True  # 忽略tokenizer尺寸不匹配问题
        )
        # 关键修复：用AutoTokenizer自动识别tokenizer类型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",  # GPT2从左padding
            truncation_side="right",
            pad_token="[PAD]"  # 手动指定PAD token
        )
        # 强制设置eos/pad token（解决中文GPT2无pad token的问题）
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id
        self.gpt2.config.eos_token_id = self.tokenizer.eos_token_id
        self.device = global_config.DEVICE

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 适配GPT2的forward逻辑
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)

        # GPT2的labels需要和input_ids维度一致，且-100为忽略值
        if labels is not None:
            labels = torch.where(labels == 0, -100, labels)  # 将PAD token(0)改为-100

        outputs = self.gpt2(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None
        )
        # 返回损失和logits
        return outputs.loss, outputs.logits

    def generate(self, input_ids, max_length=32, eos_token_id=None, pad_token_id=None, do_sample=True, top_k=50):
        # 生成逻辑适配
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id

        return self.gpt2.generate(
            input_ids=input_ids,
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            top_k=top_k,
            attention_mask=(input_ids != pad_token_id),
            num_return_sequences=1
        )