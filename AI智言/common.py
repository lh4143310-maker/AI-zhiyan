import os
import torch
import random
import numpy as np


class Config:
    """项目配置"""

    # 数据路径
    SENTIMENT_DATA_PATH = "data/sentiment.csv"
    SUMMARY_DATA_PATH = "data/summary.csv"
    QA_DATA_PATH = "data/qa.csv"

    # 预训练模型路径 (使用 fnlp/bart-base-chinese，首次运行会自动下载)
    BART_PATH = "fnlp/bart-base-chinese"

    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 情感分类标签（基于千言数据集，可扩展）
    SENTIMENT_LABELS = ["消极", "积极"]

    # 训练超参数
    MAX_EXAMPLES = 3000   # 使用全部生成数据
    BATCH_SIZE = 8
    EPOCHS = 1
    LEARNING_RATE = 5e-5
    TRAIN_RATIO = 0.8
    TEST_RATIO = 0.1
    CHECKPOINT_STEPS = 100

    # DeepSeek 内容审查配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    ENABLE_GLOBAL_REVIEW = os.getenv("ENABLE_GLOBAL_REVIEW", "false").lower() == "true"


def set_seed(seed=42):
    """设置随机数种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
