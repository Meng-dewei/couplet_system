import os
import torch

# 数据路径配置
DATA_DIR = os.path.join("data", "couplets")
TRAIN_IN_PATH = os.path.join(DATA_DIR, "train", "in.txt")
TRAIN_OUT_PATH = os.path.join(DATA_DIR, "train", "out.txt")
TEST_IN_PATH = os.path.join(DATA_DIR, "test", "in.txt")
TEST_OUT_PATH = os.path.join(DATA_DIR, "test", "out.txt")

# 通用配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 8000
MAX_LEN = 32
BATCH_SIZE = 16  # 降低批次大小，避免OOM
EPOCHS = 20      # 减少训练轮数，加快验证
LEARNING_RATE = 0.001
LOG_INTERVAL = 10000
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 模型专用配置（统一层数为1，避免维度不匹配）
LSTM_CONFIG = {
    "embedding_dim": 64,    # 降低维度，适配CPU/低配GPU
    "hidden_dim": 128,      # 双向后变为256，解码器匹配
    "num_layers": 1,        # 关键：层数统一为1
    "dropout": 0.2,
    "save_path": os.path.join(MODEL_SAVE_DIR, "lstm_model.pt")
}

GRU_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_layers": 1,        # 层数统一为1
    "dropout": 0.2,
    "save_path": os.path.join(MODEL_SAVE_DIR, "gru_model.pt")
}

TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "save_path": os.path.join(MODEL_SAVE_DIR, "transformer_model.pt")
}

PRETRAINED_CONFIG = {
    "bert": {
        "model_name": "bert-base-chinese",
        "save_path": os.path.join(MODEL_SAVE_DIR, "bert_model.pt")
    },
    "gpt2": {
        "model_name": "uer/gpt2-chinese-cluecorpussmall",
        "save_path": os.path.join(MODEL_SAVE_DIR, "gpt2_model.pt")
    }
}