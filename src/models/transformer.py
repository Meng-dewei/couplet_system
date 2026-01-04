import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入全局config模块（重命名为global_config避免冲突）
import config as global_config


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, model_config):
        super().__init__()
        self.d_model = model_config["d_model"]
        self.embedding = nn.Embedding(vocab_size, model_config["d_model"], padding_idx=0)
        # 使用全局配置的MAX_LEN，而非局部字典
        self.pos_encoder = PositionalEncoding(
            model_config["d_model"],
            model_config["dropout"],
            max_len=global_config.MAX_LEN  # 关键修复：用全局config
        )

        # 修正Transformer初始化（移除device参数，避免版本兼容问题）
        self.transformer = nn.Transformer(
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            batch_first=True  # 仅保留batch_first，移除device参数
        )

        self.fc_out = nn.Linear(model_config["d_model"], vocab_size)
        self.model_config = model_config  # 局部模型配置
        self.device = global_config.DEVICE  # 全局设备配置

    def forward(self, src, trg, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, device=self.device))
        src_emb = self.pos_encoder(src_emb)  # [batch, src_len, d_model]

        trg_emb = self.embedding(trg) * torch.sqrt(torch.tensor(self.d_model, device=self.device))
        trg_emb = self.pos_encoder(trg_emb)  # [batch, trg_len, d_model]

        if src_mask is None:
            src_mask = self._generate_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self._generate_tgt_mask(trg)

        src_pad_mask = self._generate_padding_mask(src)
        tgt_pad_mask = self._generate_padding_mask(trg)

        output = self.transformer(
            src_emb,
            trg_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )

        output = self.fc_out(output)
        src_len = torch.sum(src != 0).item()
        if trg.size(1) < src_len:
            pad_length = src_len - trg.size(1)
            trg = torch.cat([trg, torch.zeros(trg.size(0), pad_length, dtype=trg.dtype, device=self.device)], dim=1)
        return output

    def _generate_src_mask(self, src):
        src_len = src.size(1)
        return torch.zeros((src_len, src_len), device=self.device).type(torch.bool)

    def _generate_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        mask = torch.triu(torch.ones((tgt_len, tgt_len), device=self.device), diagonal=1).type(torch.bool)
        return mask

    def _generate_padding_mask(self, seq):
        return (seq == 0).to(self.device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(1)]  # [seq_len, 1, d_model]
        x = x + pe.expand(-1, x.size(0), -1).transpose(0, 1)  # 适配batch_first
        return self.dropout(x)