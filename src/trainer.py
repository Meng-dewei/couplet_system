import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config  # 全局配置

class Trainer:
    def __init__(self, model, train_loader, test_loader, model_type):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_type = model_type
        # 修正损失函数（忽略PAD token）
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
            inputs = batch['input'].to(config.DEVICE).long()
            targets = batch['target'].to(config.DEVICE).long()

            self.optimizer.zero_grad()

            if self.model_type in ["lstm", "gru"]:
                outputs = self.model(inputs, targets)
                loss = self.criterion(outputs.transpose(1, 2), targets)
            elif self.model_type == "transformer":
                # 核心修复：Transformer输入输出维度对齐
                # 目标输入：去掉最后一个token（<EOS>）
                trg_input = targets[:, :-1]
                # 目标标签：去掉第一个token（<SOS>）
                trg_label = targets[:, 1:]

                # 前向传播
                outputs = self.model(inputs, trg_input)

                # 计算损失（展平维度）
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),
                    trg_label.reshape(-1)
                )
            elif self.model_type in ["bert", "gpt2"]:
                # 预训练模型处理
                attention_mask = (inputs != 0).to(config.DEVICE)
                loss, _ = self.model(input_ids=inputs, attention_mask=attention_mask, labels=targets)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # 打印日志（每LOG_INTERVAL批次）
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                avg_loss = total_loss / config.LOG_INTERVAL
                print(f"Batch {batch_idx + 1}, Loss: {avg_loss:.4f}")
                total_loss = 0.0

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch['input'].to(config.DEVICE).long()
                targets = batch['target'].to(config.DEVICE).long()

                if self.model_type in ["lstm", "gru"]:
                    outputs = self.model(inputs, targets, teacher_forcing_ratio=0)
                    loss = self.criterion(outputs.transpose(1, 2), targets)
                elif self.model_type == "transformer":
                    trg_input = targets[:, :-1]
                    trg_label = targets[:, 1:]
                    outputs = self.model(inputs, trg_input)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), trg_label.reshape(-1))
                elif self.model_type in ["bert", "gpt2"]:
                    attention_mask = (inputs != 0).to(config.DEVICE)
                    loss, _ = self.model(input_ids=inputs, attention_mask=attention_mask, labels=targets)

                total_loss += loss.item()

        return total_loss / len(self.test_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate()
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n")

        # 保存模型
        save_paths = {
            "lstm": config.LSTM_CONFIG["save_path"],
            "gru": config.GRU_CONFIG["save_path"],
            "transformer": config.TRANSFORMER_CONFIG["save_path"],
            "bert": config.PRETRAINED_CONFIG["bert"]["save_path"],
            "gpt2": config.PRETRAINED_CONFIG["gpt2"]["save_path"]
        }
        torch.save(self.model.state_dict(), save_paths[self.model_type])
        print(f"Model saved to {save_paths[self.model_type]}")


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask