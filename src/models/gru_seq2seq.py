import torch
import torch.nn as nn
import config


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.gru(embed)
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)

        return out, hidden


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim + hidden_dim * 2,
            hidden_size=hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, context):
        x = x.unsqueeze(1)
        embed = self.dropout(self.embedding(x))

        context = context.unsqueeze(1).repeat(1, embed.size(1), 1)
        input_combined = torch.cat([embed, context], dim=2)

        out, hidden = self.gru(input_combined, hidden)
        out = self.dropout(out)
        pred = self.fc(out)

        return pred, hidden


class GRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.encoder = GRUEncoder(
            vocab_size,
            config["embedding_dim"],
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"]
        )
        self.decoder = GRUDecoder(
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
        enc_out, hidden = self.encoder(src)

        context = hidden[-1]
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, context)
            outputs[:, t, :] = output.squeeze(1)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = trg[:, t] if teacher_force else top1

        return outputs