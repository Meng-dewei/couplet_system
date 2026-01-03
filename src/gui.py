import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QPushButton, QLabel,
                             QCheckBox, QGroupBox, QFormLayout)
from PyQt5.QtGui import QFont  # 添加字体设置，解决中文乱码
from src.predictor import Predictor  # 修改导入路径


class CoupletGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("对联生成系统")
        self.setGeometry(100, 100, 800, 600)

        # 初始化预测器
        self.predictors = {
            "LSTM Seq2Seq": Predictor("lstm"),
            "GRU Seq2Seq": Predictor("gru"),
            "Transformer": Predictor("transformer"),
            "BERT-based": Predictor("bert"),
            "GPT2-based": Predictor("gpt2")
        }

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 输入区域
        input_layout = QFormLayout()
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("请输入上联...")
        input_layout.addRow("上联输入:", self.input_edit)

        # 模型选择
        model_group = QGroupBox("选择生成模型")
        model_layout = QVBoxLayout()
        self.model_checks = {}
        for model_name in self.predictors.keys():
            check = QCheckBox(model_name)
            check.setChecked(True)
            self.model_checks[model_name] = check
            model_layout.addWidget(check)
        model_group.setLayout(model_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("生成下联")
        self.generate_btn.clicked.connect(self.generate_couplet)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.generate_btn)
        btn_layout.addWidget(self.clear_btn)

        # 结果显示
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # 组装布局
        main_layout.addLayout(input_layout)
        main_layout.addWidget(model_group)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(QLabel("生成结果:"))
        main_layout.addWidget(self.result_display)

    def generate_couplet(self):
        up_couplet = self.input_edit.toPlainText().strip()
        if not up_couplet:
            self.result_display.setPlainText("请输入上联！")
            return

        self.result_display.clear()
        selected_models = [name for name, check in self.model_checks.items() if check.isChecked()]

        for model_name in selected_models:
            self.result_display.append(f"【{model_name}】")
            try:
                predictor = self.predictors[model_name]
                down_couplet = predictor.generate(up_couplet)
                self.result_display.append(f"上联: {up_couplet}")
                self.result_display.append(f"下联: {down_couplet}\n")
            except Exception as e:
                self.result_display.append(f"生成失败: {str(e)}\n")

    def clear_all(self):
        self.input_edit.clear()
        self.result_display.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置中文字体（解决乱码）
    font = QFont("SimHei", 9)
    app.setFont(font)
    window = CoupletGUI()
    window.show()
    sys.exit(app.exec_())