import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer
import config as global_config


class GPT2CoupletModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_name = model_config["model_name"]
        self.gpt2 = GPT2LMHeadModel.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", truncation_side="right", pad_token="[PAD]"
        )
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id
        self.gpt2.config.eos_token_id = self.tokenizer.eos_token_id
        self.device = global_config.DEVICE

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)
        if labels is not None:
            labels = torch.where(labels == 0, -100, labels)

        outputs = self.gpt2(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None
        )
        return outputs.loss, outputs.logits

    # 直接调用原生 generate，自动兼容所有参数
    def generate(self, *args, **kwargs):
        # 从输入参数中提取input_ids计算有效长度
        input_ids = args[0] if args else kwargs.get('input_ids')
        if input_ids is not None:
            # 计算输入的有效长度（非PAD token的数量）
            valid_input_length = torch.sum(input_ids != self.tokenizer.pad_token_id).item()  # 修复重复torch的问题
            # 调整最大长度，确保生成内容长度合理
            if "max_length" in kwargs and kwargs["max_length"] <= input_ids.size(1):
                # 最大长度设为输入长度 + 有效长度（保证生成足够内容）
                kwargs["max_length"] = input_ids.size(1) + valid_input_length
        else:
            # 若无输入，使用默认长度
            if "max_length" not in kwargs:
                kwargs["max_length"] = 64

        bad_tokens = [
            token for token in self.tokenizer.vocab
            if token.startswith('[unused') or
               not any('\u4e00' <= c <= '\u9fff' for c in token)  # 过滤非中文字符
        ]
        kwargs["max_length"] = input_ids.size(1) + valid_input_length + 5

        # 强制添加去重参数
        kwargs.update({
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2,  # 降低惩罚力度
            "do_sample": True,
            "top_k": 30,  # 减少候选词数量
            "temperature": 0.6,  # 降低随机性
            "bad_words_ids": [
                [self.tokenizer.convert_tokens_to_ids(token)]
                for token in self.tokenizer.vocab
                if token.startswith('[unused') or not any('\u4e00' <= c <= '\u9fff' for c in token)
            ],
        })

        # 生成结果
        output = self.gpt2.generate(*args, **kwargs)

        # 手动移除与输入相同的前缀
        if input_ids is not None and output.size(1) > input_ids.size(1):
            if torch.all(output[:, :input_ids.size(1)] == input_ids):
                return output[:, input_ids.size(1):]  # 只返回新增部分
        return output