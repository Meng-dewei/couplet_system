import os
import sys
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import CoupletDataset
from src.predictor import Predictor


def load_test_data():
    """加载测试集的上联（输入）和下联（参考标准答案）"""
    test_dataset = CoupletDataset(config.TEST_IN_PATH, config.TEST_OUT_PATH)
    # 过滤空字符串和过短样本（避免计算误差）
    test_pairs = []
    for in_sent, out_sent in zip(test_dataset.in_sentences, test_dataset.out_sentences):
        if in_sent.strip() and out_sent.strip() and len(in_sent) >= 3 and len(out_sent) >= 3:
            test_pairs.append((in_sent.strip(), out_sent.strip()))
    return test_pairs


def generate_all_model_results(test_pairs):
    """生成所有模型的下联结果（与测试集一一对应）"""
    models = ["lstm", "gru", "transformer", "bert", "gpt2"]
    model_results = {model: [] for model in models}

    print("开始生成所有模型的下联结果（请耐心等待，视测试集大小而定）...")
    for idx, (up_couplet, ref_down) in enumerate(test_pairs):
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(test_pairs)} 条测试数据")

        for model_type in models:
            try:
                predictor = Predictor(model_type)
                gen_down = predictor.generate(up_couplet).strip()
                # 确保生成结果与参考下联长度一致（对联核心要求）
                gen_down = gen_down[:len(ref_down)] if len(gen_down) > len(ref_down) else gen_down.ljust(len(ref_down),
                                                                                                         ' ')
                model_results[model_type].append(gen_down)
            except Exception as e:
                print(f"模型 {model_type} 处理上联「{up_couplet}」失败：{str(e)[:30]}")
                model_results[model_type].append("")  # 失败样本填充空字符串，不影响整体计算

    return model_results


def calculate_metrics(model_results, test_pairs):
    """计算所有模型的 BLEU-1~4 与 ROUGE-L 得分"""
    # 整理参考文本（corpus_bleu 要求格式：[[ref1], [ref2], ...]）
    references = [[list(ref_down)] for _, ref_down in test_pairs]

    # 初始化评估工具
    smoothie = SmoothingFunction().method4  # BLEU 平滑（解决短文本得分偏低问题）
    rouge = Rouge()
    metrics = {}

    print("\n开始计算指标得分...")
    for model_type, hypotheses in model_results.items():
        # 过滤失败样本（空字符串）
        valid_idx = [i for i, hyp in enumerate(hypotheses) if hyp.strip()]
        valid_hypotheses = [list(hypotheses[i].strip()) for i in valid_idx]
        valid_references = [references[i] for i in valid_idx]

        if not valid_hypotheses:
            metrics[model_type] = {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0, "ROUGE-L": 0.0}
            print(f"\n{model_type.upper()}：无有效生成结果，所有指标得分为 0")
            continue

        # 计算 BLEU-1~4（对联以短文本为主，重点关注 1-gram 和 2-gram 匹配）
        bleu1 = corpus_bleu(valid_references, valid_hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = corpus_bleu(valid_references, valid_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = corpus_bleu(valid_references, valid_hypotheses, weights=(0.33, 0.33, 0.34, 0),
                            smoothing_function=smoothie)
        bleu4 = corpus_bleu(valid_references, valid_hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=smoothie)

        # 计算 ROUGE-L（关注语义连贯性，适合对联的长距离匹配）
        # 转换为字符串格式（Rouge 库要求输入为句子字符串）
        hyp_strings = ["".join(hyp) for hyp in valid_hypotheses]
        ref_strings = ["".join(ref[0]) for ref in valid_references]
        rouge_scores = rouge.get_scores(hyp_strings, ref_strings, avg=True)
        rouge_l = rouge_scores["rouge-l"]["f"]  # 取 F1 分数（综合精确率和召回率）

        # 保存结果（保留4位小数，便于对比）
        metrics[model_type] = {
            "BLEU-1": round(bleu1 * 100, 4),
            "BLEU-2": round(bleu2 * 100, 4),
            "BLEU-3": round(bleu3 * 100, 4),
            "BLEU-4": round(bleu4 * 100, 4),
            "ROUGE-L": round(rouge_l * 100, 4)
        }

        # 打印当前模型得分
        print(f"\n{model_type.upper()} 指标得分：")
        print(f"  BLEU-1: {metrics[model_type]['BLEU-1']:.4f}")
        print(f"  BLEU-2: {metrics[model_type]['BLEU-2']:.4f}")
        print(f"  BLEU-3: {metrics[model_type]['BLEU-3']:.4f}")
        print(f"  BLEU-4: {metrics[model_type]['BLEU-4']:.4f}")
        print(f"  ROUGE-L: {metrics[model_type]['ROUGE-L']:.4f}")

    return metrics


def save_results(metrics, save_path="metric_results.txt"):
    """保存指标结果到文件（便于后续分析）"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("对联生成模型 BLEU + ROUGE-L 指标得分\n")
        f.write("=" * 60 + "\n\n")
        for model_type, scores in metrics.items():
            f.write(f"【{model_type.upper()}】\n")
            for metric, score in scores.items():
                f.write(f"{metric}: {score:.4f}\n")
            f.write("\n")
    print(f"\n指标结果已保存到：{os.path.abspath(save_path)}")


if __name__ == "__main__":
    # 步骤1：加载测试数据
    test_pairs = load_test_data()
    print(f"成功加载测试集：共 {len(test_pairs)} 条有效对联样本")

    # 步骤2：生成所有模型的下联结果
    model_results = generate_all_model_results(test_pairs)

    # 步骤3：计算指标得分
    metrics = calculate_metrics(model_results, test_pairs)

    # 步骤4：保存结果到文件
    save_results(metrics)