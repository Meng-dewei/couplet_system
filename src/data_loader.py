import os
import jieba
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import config


class CoupletDataset(Dataset):
    def __init__(self, in_path, out_path, vocab=None, max_len=32):
        self.in_sentences = self.load_data(in_path)
        self.out_sentences = self.load_data(out_path)
        self.max_len = max_len

        # 构建词汇表
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            sentences = [line.strip().replace(' ', '') for line in f.readlines()]
        return sentences

    def build_vocab(self):
        all_words = []
        for sent in self.in_sentences + self.out_sentences:
            all_words.extend(list(sent))

        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(config.VOCAB_SIZE - 4)]

        vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + top_words
        return vocab

    def encode(self, sentence):
        encoded = [self.word2idx['<SOS>']]  # 起始符号
        for word in list(sentence):
            if word in self.word2idx:
                encoded.append(self.word2idx[word])
            else:
                encoded.append(self.word2idx['<UNK>'])  # 未知词
        encoded.append(self.word2idx['<EOS>'])  # 结束符号

        # 填充或截断到最大长度
        if len(encoded) < self.max_len:
            encoded += [self.word2idx['<PAD>']] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return np.array(encoded)

    def __len__(self):
        return len(self.in_sentences)

    def __getitem__(self, idx):
        in_sent = self.in_sentences[idx]
        out_sent = self.out_sentences[idx]

        in_encoded = self.encode(in_sent)
        out_encoded = self.encode(out_sent)

        return {
            'input': in_encoded,
            'target': out_encoded
        }


def get_data_loaders(batch_size=config.BATCH_SIZE):
    """获取训练集和测试集的数据加载器"""
    # 先加载训练集构建词汇表
    train_dataset = CoupletDataset(
        config.TRAIN_IN_PATH,
        config.TRAIN_OUT_PATH
    )

    # 使用训练集的词汇表加载测试集
    test_dataset = CoupletDataset(
        config.TEST_IN_PATH,
        config.TEST_OUT_PATH,
        vocab=train_dataset.vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, train_dataset.vocab