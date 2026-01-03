import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders
from src.trainer import Trainer
import config


def main():
    parser = argparse.ArgumentParser(description="训练对联生成模型")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "gru", "transformer", "bert", "gpt2"],
                        help="选择要训练的模型")
    args = parser.parse_args()

    # 加载数据
    train_loader, test_loader, vocab = get_data_loaders()
    vocab_size = len(vocab)

    # 初始化模型
    if args.model == "lstm":
        from src.models.lstm_seq2seq import LSTMSeq2Seq
        model = LSTMSeq2Seq(vocab_size, config.LSTM_CONFIG)
    elif args.model == "gru":
        from src.models.gru_seq2seq import GRUSeq2Seq
        model = GRUSeq2Seq(vocab_size, config.GRU_CONFIG)
    elif args.model == "transformer":
        from src.models.transformer import TransformerModel
        # 关键：传入模型配置字典，而非全局config
        model = TransformerModel(vocab_size, config.TRANSFORMER_CONFIG)
    elif args.model == "bert":
        from src.models.bert_based import BertCoupletModel
        model = BertCoupletModel(config.PRETRAINED_CONFIG["bert"])
    elif args.model == "gpt2":
        from src.models.gpt2_based import GPT2CoupletModel
        model = GPT2CoupletModel(config.PRETRAINED_CONFIG["gpt2"])

    # 训练模型
    trainer = Trainer(model, train_loader, test_loader, args.model)
    trainer.train(config.EPOCHS)


if __name__ == "__main__":
    main()