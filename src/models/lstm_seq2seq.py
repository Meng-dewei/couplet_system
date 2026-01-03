import torch
import torch.nn as nn
import config


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0  # 单层时禁用dropout
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embed = self.embedding(x)
        out, (hidden, cell) = self.lstm(embed)

        # 调整双向LSTM的隐藏层形状：[num_layers*2, batch, hidden] → [num_layers, batch, hidden*2]
        batch_size = hidden.size(1)
        # 拆分双向维度并重组为num_layers层
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)  # [num_layers, batch, hidden*2]

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)  # [num_layers, batch, hidden*2]

        return out, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 解码器输入维度 = 词嵌入维度 + 编码器输出维度（hidden*2）
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim * 2,
            hidden_size=hidden_dim * 2,  # 匹配编码器输出维度
            num_layers=num_layers,  # 与编码器层数一致
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, context):
        x = x.unsqueeze(1)  # [batch] → [batch, 1]
        embed = self.dropout(self.embedding(x))  # [batch, 1, embedding_dim]

        # 拼接上下文向量（广播到序列长度）
        context = context.unsqueeze(1).repeat(1, embed.size(1), 1)  # [batch, 1, hidden*2]
        input_combined = torch.cat([embed, context], dim=2)  # [batch, 1, embedding_dim + hidden*2]

        # LSTM前向传播（hidden/cell已匹配层数）
        out, (hidden, cell) = self.lstm(input_combined, (hidden, cell))
        out = self.dropout(out)
        pred = self.fc(out)  # [batch, 1, vocab_size]

        return pred, hidden, cell


class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.encoder = LSTMEncoder(
            vocab_size,
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"]
        )
        self.decoder = LSTMDecoder(
            vocab_size,
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"]
        )
        self.vocab_size = vocab_size
        self.config = config
        self.num_layers = config["num_layers"]
        self.hidden_dim = config["hidden_dim"]

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)

        # 初始化输出张量 [batch, trg_len, vocab_size]
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(config.DEVICE)

        # 编码器前向传播
        enc_out, hidden, cell = self.encoder(src)

        # 上下文向量（取编码器最后一层输出）
        context = hidden[-1]  # [batch, hidden_dim*2]

        # 解码器初始输入：<SOS> token
        input = trg[:, 0]  # [batch]

        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell, context)
            outputs[:, t, :] = output.squeeze(1)  # 移除序列维度

            # 教师强制：随机使用真实标签或预测结果作为下一个输入
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)  # 取概率最大的token
            input = trg[:, t] if teacher_force else top1

        return outputs