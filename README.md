# 对联生成系统 (Couplet System)

一个基于深度学习的对联自动生成系统，能够根据输入的上联生成对应的下联，支持传统对联的平仄、对仗规则约束。

## 项目简介

本项目实现了一个智能对联生成系统，通过训练深度学习模型来学习对联的语言规律和对仗规则，实现给定上联自动生成合适下联的功能。系统支持多种模型配置，并提供了训练、测试和推理的完整流程。

## 文件结构

```
couplet_system/
├── .gitignore                  # Git忽略文件配置
├── config.py                   # 项目配置参数（模型、训练、数据等配置）
├── metric_evaluator.py         # 模型评估指标计算脚本
├── output.md                   # 模型输出结果记录
├── README.md                   # 项目说明文档
├── test_predict.py             # 模型测试与预测脚本
├── 指令.md                     # 项目相关指令说明
├── .idea/                      # IDE配置目录
│   ├── .gitignore
│   ├── couplet_system.iml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   ├── workspace.xml
│   └── inspectionProfiles/
│       └── profiles_settings.xml
├── data/                       # 数据集目录
│   └── couplets/
│       ├── test/               # 测试数据集
│       │   ├── in.txt          # 测试集上联
│       │   └── out.txt         # 测试集下联（参考答案）
│       └── train/              # 训练数据集
│           ├── in.txt          # 训练集上联
│           └── out.txt         # 训练集下联
├── saved_models/               # 保存的模型文件
│   ├── bert_model.pt           # BERT模型权重
│   ├── gpt2_model.pt           # GPT2模型权重
│   ├── gru_model.pt            # GRU模型权重
│   ├── lstm_model.pt           # LSTM模型权重
│   └── transformer_model.pt    # Transformer模型权重
├── src/                        # 核心源代码目录
│   ├── data_loader.py          # 数据加载与预处理模块
│   ├── gui.py                  # 图形用户界面（交互预测）
│   ├── predictor.py            # 对联生成预测模块
│   ├── train.py                # 训练入口脚本
│   ├── trainer.py              # 模型训练核心逻辑
│   ├── __init__.py
│   ├── models/                 # 模型定义目录
│   │   ├── bert_based.py       # 基于BERT的模型实现
│   │   ├── gpt2_based.py       # 基于GPT2的模型实现
│   │   ├── gru_seq2seq.py      # GRU序列到序列模型
│   │   ├── lstm_seq2seq.py     # LSTM序列到序列模型
│   │   ├── transformer.py      # Transformer模型实现
│   │   ├── __init__.py
│   │   └── __pycache__/        # 模型编译缓存
│   │       ├── bert_based.cpython-310.pyc
│   │       ├── gpt2_based.cpython-310.pyc
│   │       ├── gru_seq2seq.cpython-310.pyc
│   │       ├── lstm_seq2seq.cpython-310.pyc
│   │       ├── transformer.cpython-310.pyc
│   │       └── __init__.cpython-310.pyc
│   └── __pycache__/            # 源代码编译缓存
│       ├── data_loader.cpython-310.pyc
│       ├── predictor.cpython-310.pyc
│       ├── train.cpython-310.pyc
│       ├── trainer.cpython-310.pyc
│       └── __init__.cpython-310.pyc
└── __pycache__/                # 根目录模块编译缓存
    └── config.cpython-310.pyc
```

## 核心模块说明

1. **数据模块** (src/data_loader.py):
   - 负责加载`data/couplets`目录下的训练集和测试集
   - 实现文本预处理（如分词、编码等）
2. **模型模块** (`src/models/`):
   - 包含多种模型实现：
     - LSTM/GRU 序列到序列模型（基础序列生成）
     - Transformer 模型（自注意力机制）
     - BERT/GPT2 预训练模型（基于预训练微调）
3. **训练模块** (src/train.py、src/trainer.py):
   - 实现模型训练流程，支持多种训练参数配置（通过config.py）
   - 训练结果保存至`saved_models`目录
4. **预测模块** (src/predictor.py、test_predict.py):
   - 加载预训练模型进行下联生成
   - 支持批量测试和单句预测
5. **评估模块** (metric_evaluator.py):
   - 计算模型生成结果的评估指标（如 BLEU 等）
   - 评估结果可记录于output.md
6. **交互界面** (src/gui.py):
   - 提供图形化界面，支持用户输入上联并实时生成下联

## 使用流程

1. 配置参数：修改config.py设置模型类型、训练参数等
2. 模型训练：运行src/train.py训练指定模型
3. 模型测试：通过test_predict.py测试模型性能
4. 交互预测：运行src/gui.py打开图形界面进行对联生成

运行代码如下：

```
 python -m src.train --model lstm 
 python -m src.train --model gru        
 python -m src.train --model transformer
 python -m src.train --model bert   
 python -m src.train --model gpt2 
 python test_predict.py 
 python src/gui.py 
```

## 注意事项

- 预训练模型保存在`saved_models`目录，可直接用于预测
- 数据集格式需符合`data/couplets`中`in.txt`（上联）和`out.txt`（下联）的对应关系
- 不同模型的性能和生成效果可能存在差异，可根据需求选择合适模型