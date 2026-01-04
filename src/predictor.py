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
        self.last_input_tensor = input_tensor  # 保存输入张量用于长度计算

        with torch.no_grad():
            if self.model_type in ["lstm", "gru"]:
                return self._generate_seq2seq(input_tensor)
            elif self.model_type == "transformer":
                return self._generate_transformer(input_tensor)
            elif self.model_type in ["bert", "gpt2"]:
                return self._generate_pretrained(input_tensor)

    def _encode(self, sentence):
        punctuation = '，。,.'
        sentence = ''.join([c for c in sentence if c not in punctuation])
        encoded = [self.word2idx['<SOS>']]
        for word in list(sentence):
            encoded.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        encoded.append(self.word2idx['<EOS>'])

        encoded = [self.word2idx['<SOS>']]
        for word in list(sentence):
            encoded.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        encoded.append(self.word2idx['<EOS>'])
        if len(encoded) < config.MAX_LEN:
            encoded += [self.word2idx['<PAD>']] * (config.MAX_LEN - len(encoded))
        return encoded[:config.MAX_LEN]

    # 找到 _generate_seq2seq 方法，修改如下
    def _generate_seq2seq(self, input_tensor):
        batch_size = input_tensor.size(0)
        target_seq = torch.zeros(batch_size, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
        target_seq[:, 0] = self.word2idx['<SOS>']

        # 计算上联有效长度（去除特殊符号）
        input_indices = input_tensor.squeeze(0).cpu().numpy()
        valid_input_indices = [idx for idx in input_indices
                               if self.idx2word.get(idx, '') not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
        input_length = len(valid_input_indices)  # 上联有效长度

        if self.model_type == "lstm":
            enc_out, hidden, cell = self.model.encoder(input_tensor)
        else:  # gru
            enc_out, hidden = self.model.encoder(input_tensor)
            cell = None

        context = hidden[-1]
        input_token = target_seq[:, 0]
        generated_length = 0  # 已生成有效字符数

        for t in range(1, config.MAX_LEN):
            if self.model_type == "lstm":
                output, hidden, cell = self.model.decoder(input_token, hidden, cell, context)
            else:  # gru
                output, hidden = self.model.decoder(input_token, hidden, context)

            pred = output.argmax(2).squeeze(1)
            target_seq[:, t] = pred
            input_token = pred

            # 计算有效字符（过滤特殊符号）
            pred_word = self.idx2word.get(pred.item(), '')
            if pred_word not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
                generated_length += 1

            # 当生成长达到上联长度或遇到结束符时停止
            if (generated_length == input_length) or (pred.item() == self.word2idx['<EOS>']):
                break

        # 强制截断到上联长度
        decoded = self._decode(target_seq.squeeze(0).cpu().numpy())
        return decoded[:input_length] if len(decoded) > input_length else decoded

    def _generate_transformer(self, input_tensor):
        target_seq = torch.zeros(1, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
        target_seq[:, 0] = self.word2idx['<SOS>']
        input_len = len(input_tensor[0].nonzero())  # 上联实际长度
        min_length = max(3, input_len)  # 确保生成足够长度

        for t in range(1, config.MAX_LEN):
            output = self.model(input_tensor, target_seq[:, :t])
            pred = output[:, -1, :].argmax(1)
            target_seq[:, t] = pred

            # 控制生成长度
            if pred.item() == self.word2idx['<EOS>'] and t >= min_length:
                break

        # 后处理：去除重复字符
        decoded = self._decode(target_seq.squeeze(0).cpu().numpy())
        # 简单去重逻辑
        result = []
        for char in decoded:
            if len(result) == 0 or char != result[-1]:
                result.append(char)
        return ''.join(result)

    def _generate_pretrained(self, input_tensor):
        input_len = torch.sum(input_tensor != self.word2idx['<PAD>']).item()
        # 计算上联实际有效长度（不含特殊符号）
        input_indices = input_tensor.squeeze(0).cpu().numpy()
        valid_input_length = len([idx for idx in input_indices
                                  if self.idx2word.get(idx, '') not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
        max_gen_length = input_len + valid_input_length + 5

        # 获取模型专用tokenizer
        tokenizer = self.model.tokenizer  # 统一获取tokenizer

        if self.model_type == "bert":
            output = self.model.generate(
                input_tensor,
                max_length=max_gen_length,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                top_k=30
            )
        else:  # gpt2
            output = self.model.generate(
                input_tensor,
                max_length=input_len + valid_input_length,
                min_length=input_len + valid_input_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=30,
                temperature=0.7,
                no_repeat_ngram_size=2,
                repetition_penalty=2.0,  # 降低重复惩罚力度
                num_return_sequences=1,
                length_penalty=1.5
            )

        # 使用模型专用tokenizer解码（只保留这一处解码）
        decoded_tokens = tokenizer.decode(
            output.squeeze(0).cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 移除可能包含的上联部分（使用模型tokenizer解码上联）
        up_clean = tokenizer.decode(
            input_tensor.squeeze(0).cpu().numpy(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        if up_clean in decoded_tokens:
            decoded_tokens = decoded_tokens.replace(up_clean, '', 1).strip()

        # 长度调整（确保与上联有效长度一致）
        if len(decoded_tokens) > valid_input_length:
            decoded_tokens = decoded_tokens[:valid_input_length]
        elif len(decoded_tokens) < valid_input_length:
            print("BERT原始生成结果:", decoded_tokens)
            couplet_chars = ['春', '夏', '秋', '冬', '风', '花', '雪', '月',
                             '山', '水', '云', '天', '地', '人', '心', '情']
            import random
            random.shuffle(couplet_chars)
            decoded_tokens += ''.join(couplet_chars[:valid_input_length - len(decoded_tokens)])

        return decoded_tokens if decoded_tokens else "（生成失败）"

    # 在predictor.py的_decode方法中
    def _decode(self, indices):
        result = []
        punctuation = '，。,.'
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word == '<EOS>':
                break
            if word not in ['<SOS>', '<PAD>', '<UNK>'] and word not in punctuation:
                result.append(word)

        # 计算上联有效长度
        input_indices = self.last_input_tensor.squeeze(0).cpu().numpy() if hasattr(self, 'last_input_tensor') else []
        input_length = len(
            [idx for idx in input_indices if self.idx2word.get(idx, '') not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])

        # 改进补充逻辑：使用更丰富的候选池并避免固定模式
        if len(result) < input_length:
            # 扩展补充字库，增加更多与对联相关的常用字
            complement = ['天', '地', '人', '日', '月', '山', '水', '云', '风', '雨',
                          '花', '鸟', '春', '夏', '秋', '冬', '晨', '昏', '朝', '暮']
            # 随机选择补充字，避免固定顺序
            import random
            random.shuffle(complement)
            result += complement[:input_length - len(result)]

        # 截断到上联长度
        return ''.join(result[:input_length])

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask