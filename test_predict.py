import torch
import config
from src.data_loader import CoupletDataset

# 加载共享词汇表（与训练时一致）
train_dataset = CoupletDataset(config.TRAIN_IN_PATH, config.TRAIN_OUT_PATH)
vocab = train_dataset.vocab
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)


# 加载所有模型
def load_all_models():
    models = {}

    # LSTM模型
    from src.models.lstm_seq2seq import LSTMSeq2Seq
    lstm_model = LSTMSeq2Seq(vocab_size, config.LSTM_CONFIG)
    lstm_model.load_state_dict(torch.load(config.LSTM_CONFIG["save_path"], map_location=config.DEVICE))
    lstm_model.to(config.DEVICE)
    lstm_model.eval()
    models["lstm"] = lstm_model

    # GRU模型
    from src.models.gru_seq2seq import GRUSeq2Seq
    gru_model = GRUSeq2Seq(vocab_size, config.GRU_CONFIG)
    gru_model.load_state_dict(torch.load(config.GRU_CONFIG["save_path"], map_location=config.DEVICE))
    gru_model.to(config.DEVICE)
    gru_model.eval()
    models["gru"] = gru_model

    # Transformer模型
    from src.models.transformer import TransformerModel
    transformer_model = TransformerModel(vocab_size, config.TRANSFORMER_CONFIG)
    transformer_model.load_state_dict(torch.load(config.TRANSFORMER_CONFIG["save_path"], map_location=config.DEVICE))
    transformer_model.to(config.DEVICE)
    transformer_model.eval()
    models["transformer"] = transformer_model

    # BERT模型
    from src.models.bert_based import BertCoupletModel
    bert_model = BertCoupletModel(config.PRETRAINED_CONFIG["bert"])
    bert_model.load_state_dict(torch.load(config.PRETRAINED_CONFIG["bert"]["save_path"], map_location=config.DEVICE))
    bert_model.to(config.DEVICE)
    bert_model.eval()
    models["bert"] = bert_model

    # GPT2模型
    from src.models.gpt2_based import GPT2CoupletModel
    gpt2_model = GPT2CoupletModel(config.PRETRAINED_CONFIG["gpt2"])
    gpt2_model.load_state_dict(torch.load(config.PRETRAINED_CONFIG["gpt2"]["save_path"], map_location=config.DEVICE))
    gpt2_model.to(config.DEVICE)
    gpt2_model.eval()
    models["gpt2"] = gpt2_model

    return models


# 统一编码函数
def encode(sentence):
    encoded = [word2idx['<SOS>']]
    for word in list(sentence):
        encoded.append(word2idx.get(word, word2idx['<UNK>']))
    encoded.append(word2idx['<EOS>'])
    encoded += [word2idx['<PAD>']] * (config.MAX_LEN - len(encoded))
    return torch.tensor(encoded[:config.MAX_LEN], device=config.DEVICE).unsqueeze(0).long()


# 针对不同模型的生成函数
def generate_for_model(model, model_type, input_tensor):
    with torch.no_grad():
        if model_type in ["lstm", "gru"]:
            # Seq2Seq模型生成逻辑
            batch_size = input_tensor.size(0)
            target_seq = torch.zeros(batch_size, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
            target_seq[:, 0] = word2idx['<SOS>']

            # 编码器前向传播
            if model_type == "lstm":
                _, hidden, cell = model.encoder(input_tensor)
            else:  # gru
                _, hidden = model.encoder(input_tensor)
                cell = None

            context = hidden[-1]
            input_token = target_seq[:, 0]

            # 解码器生成
            for t in range(1, config.MAX_LEN):
                if model_type == "lstm":
                    output, hidden, cell = model.decoder(input_token, hidden, cell, context)
                else:  # gru
                    output, hidden = model.decoder(input_token, hidden, context)

                pred = output.argmax(2).squeeze(1)
                target_seq[:, t] = pred
                input_token = pred

                if pred.item() == word2idx['<EOS>']:
                    break

            return target_seq.squeeze(0).cpu().numpy()

        elif model_type == "transformer":
            # Transformer生成逻辑
            target_seq = torch.zeros(1, config.MAX_LEN, dtype=torch.long).to(config.DEVICE)
            target_seq[:, 0] = word2idx['<SOS>']

            for t in range(1, config.MAX_LEN):
                output = model(input_tensor, target_seq[:, :t])
                pred = output[:, -1, :].argmax(1)
                target_seq[:, t] = pred

                if pred.item() == word2idx['<EOS>']:
                    break

            return target_seq.squeeze(0).cpu().numpy()

        elif model_type == "bert":
            # BERT模型生成逻辑
            output = model.generate(
                input_tensor,
                max_length=config.MAX_LEN,
                eos_token_id=word2idx['<EOS>'],
                pad_token_id=word2idx['<PAD>']
            )
            return output.squeeze(0).cpu().numpy()

        elif model_type == "gpt2":
            # GPT2模型生成逻辑
            output = model.generate(
                input_tensor,
                max_length=config.MAX_LEN,
                eos_token_id=word2idx['<EOS>'],
                pad_token_id=word2idx['<PAD>'],
                do_sample=True,
                top_k=50,
                repetition_penalty=1.0
            )
            return output.squeeze(0).cpu().numpy()


# 解码函数（无过滤，仅移除特殊符号）
def decode(indices):
    return ''.join([
        idx2word.get(idx.item(), '')
        for idx in indices
        if idx not in [word2idx['<SOS>'], word2idx['<EOS>'], word2idx['<PAD>']]
    ])


# 测试所有模型
def test_all_models(up_couplet):
    models = load_all_models()
    input_tensor = encode(up_couplet)

    print(f"上联: {up_couplet}")
    print("-" * 50)

    for model_type, model in models.items():
        try:
            output_indices = generate_for_model(model, model_type, input_tensor)
            down_couplet = decode(output_indices)
            print(f"{model_type.upper()} 下联: {down_couplet}")
        except Exception as e:
            print(f"{model_type.upper()} 生成失败: {str(e)[:50]}")
    print("-" * 50)


# 测试案例
if __name__ == "__main__":
    test_couplets = [
        "春风化雨润万物",
        "山清水秀风光好",
        "一元复始开新宇"
    ]
    for couplet in test_couplets:
        test_all_models(couplet)