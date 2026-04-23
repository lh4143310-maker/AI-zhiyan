import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def process(
    task,
    data_path,
    max_examples,
    batch_size,
    tokenizer,
    train_ratio=0.8,
    test_ratio=0.1,
    label_list=None,
):
    """
    数据预处理，支持三种任务：
    - sentiment: 情感分类
    - summarize: 文本摘要
    - qa: 智能问答

    参数:
    - task: 任务类型，["sentiment", "summarize", "qa"]
    - data_path: 数据路径
    - train_ratio: 训练集比例
    - test_ratio: 测试集比例
    - max_examples: 最大样本数
    - batch_size: 批次大小
    - tokenizer: 分词器
    - label_list: 标签列表（分类任务需要）

    返回值:
    - dataloader: 数据加载器字典 {"train": ..., "valid": ..., "test": ...}
    """

    def _map_fn(batch):
        """对批次数据进行 tokenize"""
        if task == "sentiment":
            result = tokenizer(
                batch["text"],
                max_length=512,
                truncation=True,
            )
            result["labels"] = [label_map[label] for label in batch["label"]]
        elif task == "summarize":
            result = tokenizer(
                batch["text"],
                max_length=512,
                truncation=True,
            )
            # 摘要标签单独 tokenize
            summary_tokens = tokenizer(
                batch["summary"],
                max_length=128,
                truncation=True,
            )
            result["labels"] = summary_tokens["input_ids"]
        elif task == "qa":
            # 问答任务：将问题和上下文拼接
            inputs = [
                f"问题：{q} 上下文：{c}"
                for q, c in zip(batch["question"], batch["context"])
            ]
            result = tokenizer(
                inputs,
                max_length=512,
                truncation=True,
            )
            answer_tokens = tokenizer(
                batch["answer"],
                max_length=128,
                truncation=True,
            )
            result["labels"] = answer_tokens["input_ids"]
        return result

    def _collate_fn(batch):
        """自定义批次合并函数"""
        input_ids = [torch.tensor(x["input_ids"]) for x in batch]
        input_ids = pad_sequence(input_ids, True, tokenizer.pad_token_id)
        attention_mask = (input_ids != tokenizer.pad_token_id).int()

        if task in ("summarize", "qa"):
            labels = [torch.tensor(x["labels"]) for x in batch]
            labels = pad_sequence(labels, True, -100)
        elif task == "sentiment":
            labels = torch.tensor([x["labels"] for x in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # 确定读取的列
    if task == "sentiment":
        assert label_list is not None, "情感分类任务需要提供 label_list"
        label_map = {label: i for i, label in enumerate(label_list)}
        columns = ["text", "label"]
    elif task == "summarize":
        columns = ["text", "summary"]
    elif task == "qa":
        columns = ["question", "context", "answer"]
    else:
        raise ValueError("task 必须是 sentiment, summarize 或 qa 之一")

    # 读取数据
    df = pd.read_csv(data_path)[columns].dropna()
    df = df.sample(min(len(df), max_examples), random_state=42)
    dataset = Dataset.from_pandas(df)

    # tokenize 并划分数据集
    dataset = dataset.map(_map_fn, batched=True)
    dataset = dataset.train_test_split(test_size=test_ratio, seed=42)
    train_size = int(dataset["train"].num_rows * train_ratio)
    dataset["train"], dataset["valid"] = (
        dataset["train"].train_test_split(train_size=train_size, seed=42).values()
    )

    # 转换为 DataLoader
    return {
        phase: DataLoader(
            dataset=dataset[phase],
            batch_size=batch_size,
            collate_fn=_collate_fn,
            shuffle=(phase == "train"),
        )
        for phase in ["train", "valid", "test"]
    }


if __name__ == "__main__":
    from common import Config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Config.BART_PATH)
    dataloader = process(
        task="sentiment",
        data_path=Config.SENTIMENT_DATA_PATH,
        max_examples=200,
        batch_size=Config.BATCH_SIZE,
        tokenizer=tokenizer,
        label_list=Config.SENTIMENT_LABELS,
    )
    print("训练批次示例:")
    print(next(iter(dataloader["train"])))
