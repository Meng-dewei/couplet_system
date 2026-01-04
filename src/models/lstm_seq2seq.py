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

        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)  # [num_layers, batch, hidden*2]

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)  # [num_layers, batch, hidden*2]

        return out, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim * 2,
            hidden_size=hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell, context):
        x = x.unsqueeze(1)  # [batch] → [batch, 1]
        embed = self.dropout(self.embedding(x))  # [batch, 1, embedding_dim]

        context = context.unsqueeze(1).repeat(1, embed.size(1), 1)  # [batch, 1, hidden*2]
        input_combined = torch.cat([embed, context], dim=2)  # [batch, 1, embedding_dim + hidden*2]

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

        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(config.DEVICE)
        enc_out, hidden, cell = self.encoder(src)
        context = hidden[-1]  # [batch, hidden_dim*2]
        input = trg[:, 0]  # [batch]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, context)
            outputs[:, t, :] = output.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = trg[:, t] if teacher_force else top1

        return outputs