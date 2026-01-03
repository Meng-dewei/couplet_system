import torch
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import CoupletDataset  # 修改导入路径


class Predictor:
    def __init__(self, model_type, vocab=None):
        self.model_type = model_type
        self.vocab = vocab if vocab else self._load_vocab()
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.model = self._load_model()

    def _load_vocab(self):
        train_dataset = CoupletDataset(config.TRAIN_IN_PATH, config.TRAIN_OUT_PATH)
        return train_dataset.vocab

    def _load_model(self):
        if self.model_type == "lstm":
            from src.models.lstm_seq2seq import LSTMSeq2Seq  # 修改导入路径
            model = LSTMSeq2Seq(len(self.vocab), config.LSTM_CONFIG)
            model.load_state_dict(torch.load(config.LSTM_CONFIG["save_path"], map_location=config.DEVICE))
        elif self.model_type == "gru":
            from src.models.gru_seq2seq import GRUSeq2Seq  # 修改导入路径
            model = GRUSeq2Seq(len(self.vocab), config.GRU_CONFIG)
            model.load_state_dict(torch.load(config.GRU_CONFIG["save_path"], map_location=config.DEVICE))
        elif self.model_type == "transformer":
            from src.models.transformer import TransformerModel  # 修改导入路径
            model = TransformerModel(len(self.vocab), config.TRANSFORMER_CONFIG)
            model.load_state_dict(torch.load(config.TRANSFORMER_CONFIG["save_path"], map_location=config.DEVICE))
        elif self.model_type == "bert":
            from src.models.bert_based import BertCoupletModel  # 修改导入路径
            model = BertCoupletModel(config.PRETRAINED_CONFIG["bert"])
            model.load_state_dict(torch.load(config.PRETRAINED_CONFIG["bert"]["save_path"], map_location=config.DEVICE))
        elif self.model_type == "gpt2":
            from src.models.gpt2_based import GPT2CoupletModel  # 修改导入路径
            model = GPT2CoupletModel(config.PRETRAINED_CONFIG["gpt2"])
            model.load_state_dict(torch.load(config.PRETRAINED_CONFIG["gpt2"]["save_path"], map_location=config.DEVICE))

        model.to(config.DEVICE)
        model.eval()
        return model

    def generate(self, up_couplet):
        up_clean = up_couplet.replace(' ', '')
        encoded = self._encode(up_clean)
        input_tensor = torch.tensor(encoded, device=config.DEVICE).unsqueeze(0).long()

        with torch.no_grad():
            if self.model_type in ["lstm", "gru"]:
                return self._generate_seq2seq(input_tensor)
            elif self.model_type == "transformer":
                return self._generate_transformer(input_tensor)
            elif self.model_type in ["bert", "gpt2"]:
                return self._generate_pretrained(input_tensor)

    def _encode(self, sentence):
        encoded = [self.word2idx['<SOS>']]
        for word in list(sentence):
            encoded.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        encoded.append(self.word2idx['<EOS>'])
        if len(encoded) < config.MAX_LEN:
            encoded += [self.word2idx['<PAD>']] * (config.MAX_LEN - len(encoded))
        return encoded[:config.MAX_LEN]

    # 找到 _generate_seq2seq 方法，修改如下
    def _generate_seq2seq(self, input_tensor):
        # LSTM/GRU生成逻辑
        batch_size = input_tensor.size(0)
        target_seq = torch.zeros(batch_size, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
        target_seq[:, 0] = self.word2idx['<SOS>']

        # 编码器输出（hidden已适配层数）
        enc_out, hidden, cell = self.model.encoder(input_tensor)
        context = hidden[-1]  # 取最后一层作为上下文
        input = target_seq[:, 0]

        for t in range(1, config.MAX_LEN):
            output, hidden, cell = self.model.decoder(input, hidden, cell, context)
            pred = output.argmax(2).squeeze(1)
            target_seq[:, t] = pred
            input = pred

            if pred.item() == self.word2idx['<EOS>']:
                break

        return self._decode(target_seq.squeeze(0).cpu().numpy())

    def _generate_transformer(self, input_tensor):
        # Transformer生成逻辑（适配修正后的模型）
        target_seq = torch.zeros(1, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
        target_seq[:, 0] = self.word2idx['<SOS>']

        for t in range(1, config.MAX_LEN):
            # 模型前向传播（无需手动生成掩码）
            output = self.model(input_tensor, target_seq[:, :t])
            # 取最后一个token的预测结果
            pred = output[:, -1, :].argmax(1)
            target_seq[:, t] = pred

            if pred.item() == self.word2idx['<EOS>']:
                break

        return self._decode(target_seq.squeeze(0).cpu().numpy())

    def _generate_pretrained(self, input_tensor):
        # 预训练模型生成逻辑
        output = self.model.generate(
            input_tensor,
            max_length=config.MAX_LEN,
            eos_token_id=self.word2idx['<EOS>'],
            pad_token_id=self.word2idx['<PAD>'],
            do_sample=True,
            top_k=50
        )
        return self._decode(output.squeeze(0).cpu().numpy())

    def _decode(self, indices):
        result = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word == '<EOS>':
                break
            if word not in ['<SOS>', '<PAD>', '<UNK>']:
                result.append(word)
        return ''.join(result)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask