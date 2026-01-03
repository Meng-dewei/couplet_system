import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import Predictor
import config

# 定义测试用的上联列表（覆盖不同长度/风格）
TEST_COUPLETS = [
    "春风化雨润万物",
    "山清水秀风光好",
    "风弦未拨心先乱",
    "一元复始开新宇",
    "举杯邀月思千里",
    "梅开五福迎新春"
]


def test_single_model(model_type: str, up_couplets: list):
    """
    测试单个模型的对联生成效果
    :param model_type: 模型类型（lstm/gru/transformer/bert/gpt2）
    :param up_couplets: 测试用的上联列表
    """
    print("=" * 80)
    print(f"【测试模型】: {model_type.upper()}")
    print("=" * 80)

    try:
        # 初始化预测器
        predictor = Predictor(model_type)

        # 逐个测试上联
        for idx, up in enumerate(up_couplets, 1):
            print(f"\n[{idx}] 上联: {up}")
            try:
                # 生成下联
                down = predictor.generate(up)
                print(f"    下联: {down}")
            except Exception as e:
                print(f"    生成失败: {str(e)[:100]}")  # 截断过长的错误信息

    except Exception as e:
        print(f"\n模型初始化失败: {str(e)[:200]}")
        print("可能原因：1. 模型未训练 2. 模型文件缺失 3. 依赖未安装\n")


def test_all_models(up_couplets: list):
    """测试所有支持的模型"""
    # 支持的模型列表
    model_types = ["lstm", "gru", "transformer", "bert", "gpt2"]

    print("=" * 80)
    print("开始测试所有对联生成模型（仅测试已训练的模型）")
    print("=" * 80)

    # 逐个测试模型
    for model_type in model_types:
        test_single_model(model_type, up_couplets)
        print("\n" + "-" * 80 + "\n")


def batch_test_custom_couplets(model_type: str, custom_couplets: list):
    """批量测试自定义上联"""
    print(f"\n【批量测试 - {model_type.upper()}】")
    print("=" * 80)
    predictor = Predictor(model_type)
    for up in custom_couplets:
        down = predictor.generate(up)
        print(f"上联: {up:<10} → 下联: {down}")


if __name__ == "__main__":
    # 1. 基础模式：测试所有模型 + 预设上联
    test_all_models(TEST_COUPLETS)

    # 2. 可选：批量测试自定义上联（取消注释启用）
    # CUSTOM_COUPLETS = ["海阔天空任我行", "书山有路勤为径"]
    # batch_test_custom_couplets("lstm", CUSTOM_COUPLETS)

    # 3. 可选：单独测试某一个模型（取消注释启用）
    # test_single_model("transformer", ["明月松间照"])