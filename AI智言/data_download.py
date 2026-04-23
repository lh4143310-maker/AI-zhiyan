"""
百度开源数据集下载与处理脚本

支持下载的数据集:
1. 情感分析 - 使用 lansinuote/chinese_sentiment (千言情感分析数据镜像)
2. 文本摘要 - 使用中文摘要数据集
3. 智能问答 - 使用 PaddlePaddle/DuReader (百度开源阅读理解数据集)

运行方式:
    python data_download.py
"""

import os
import pandas as pd
from datasets import load_dataset


def download_sentiment_data(output_path="data/sentiment.csv", max_samples=50000):
    """
    下载千言情感分析数据集（积极/消极二分类）
    数据来源: lansinuote/chinese_sentiment
    """
    print("=" * 50)
    print("【下载】情感分析数据集")
    print("=" * 50)

    try:
        # 尝试从 datasets 加载
        dataset = load_dataset("lansinuote/chinese_sentiment", split="train")
        df = dataset.to_pandas()

        # 重命名列以匹配项目格式
        if "text" in df.columns and "label" in df.columns:
            df = df[["text", "label"]]
            # 将数字标签转为中文标签
            label_map = {0: "消极", 1: "积极"}
            df["label"] = df["label"].map(label_map)
        else:
            # 适配不同格式的数据集
            df.columns = ["text", "label"]
            df["label"] = df["label"].map({0: "消极", 1: "积极"})

        df = df.dropna().sample(min(len(df), max_samples), random_state=42)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"成功保存到: {output_path}")
        print(f"样本数: {len(df)}")
        print(df.head(3))
        return True

    except Exception as e:
        print(f"自动下载失败: {e}")
        print("将创建示例数据...")
        return create_sentiment_sample(output_path)


def create_sentiment_sample(output_path="data/sentiment.csv"):
    """创建情感分析示例数据"""
    samples = [
        ("这部电影真的太棒了，演员演技在线，剧情紧凑！", "积极"),
        ("服务态度极差，等了一个小时还没上菜。", "消极"),
        ("今天天气不错，心情也挺好的。", "积极"),
        ("产品质量太差了，用了两天就坏了。", "消极"),
        ("物流很快，包装完好，非常满意！", "积极"),
        ("说明书写得太混乱了，完全看不懂。", "消极"),
        ("这家餐厅的火锅味道正宗，下次还会来。", "积极"),
        ("客服态度冷漠，问题始终没解决。", "消极"),
        ("孩子很喜欢这个玩具，玩得特别开心。", "积极"),
        ("软件闪退严重，根本无法正常使用。", "消极"),
    ]
    # 扩展到1000条
    extended = samples * 100
    df = pd.DataFrame(extended, columns=["text", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"示例数据已保存到: {output_path} (样本数: {len(df)})")
    return True


def download_summary_data(output_path="data/summary.csv", max_samples=20000):
    """
    下载中文摘要数据集
    使用 nlpcc_entrance_exam 或类似数据集
    """
    print("\n" + "=" * 50)
    print("【下载】文本摘要数据集")
    print("=" * 50)

    try:
        # 尝试加载中文摘要数据集
        dataset = load_dataset(" WangZeJun/LCSTS", split="train")
        df = dataset.to_pandas()
        df = df.rename(columns={"content": "text", "summary": "summary"})
        df = df[["text", "summary"]].dropna()
        df = df.sample(min(len(df), max_samples), random_state=42)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"成功保存到: {output_path}")
        print(f"样本数: {len(df)}")
        return True
    except Exception as e:
        print(f"自动下载失败: {e}")
        print("将创建示例数据...")
        return create_summary_sample(output_path)


def create_summary_sample(output_path="data/summary.csv"):
    """创建摘要示例数据"""
    samples = [
        (
            "北京时间4月22日，中国航天科技集团发布消息称，长征七号遥八运载火箭"
            "已完成出厂前所有研制工作，于近日安全运抵文昌航天发射场。",
            "长征七号遥八运载火箭运抵文昌发射场。",
        ),
        (
            "据日本鹿儿岛地方气象台消息，当地时间3日13时49分左右，位于鹿儿岛县"
            "和宫崎县交界地区雾岛山的新燃岳火山喷发，火山灰柱最大高度达5000米。",
            "日本新燃岳火山喷发，火山灰柱高达5000米。",
        ),
        (
            "自7月7日开始，在中国人民抗日战争纪念馆举办主题展览，"
            "展出照片1525张、文物3237件，将作为基本陈列长期展出。",
            "抗战纪念馆举办主题展览，展出大量珍贵文物。",
        ),
    ]
    extended = samples * 500
    df = pd.DataFrame(extended, columns=["text", "summary"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"示例数据已保存到: {output_path} (样本数: {len(df)})")
    return True


def download_qa_data(output_path="data/qa.csv", max_samples=20000):
    """
    下载 DuReader 阅读理解数据集
    数据来源: PaddlePaddle/DuReader
    """
    print("\n" + "=" * 50)
    print("【下载】智能问答数据集 (DuReader)")
    print("=" * 50)

    try:
        dataset = load_dataset("PaddlePaddle/DuReader", "robust", split="train")
        df = dataset.to_pandas()

        # DuReader 数据格式适配
        records = []
        for _, row in df.iterrows():
            question = row.get("question", "")
            context = " ".join(row.get("documents", [])) if "documents" in row else row.get("context", "")
            answers = row.get("answers", [])
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0] if isinstance(answers[0], str) else answers[0].get("text", "")
            else:
                answer = str(answers)
            if question and context and answer:
                records.append({"question": question, "context": context, "answer": answer})

        df = pd.DataFrame(records).dropna()
        df = df.sample(min(len(df), max_samples), random_state=42)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"成功保存到: {output_path}")
        print(f"样本数: {len(df)}")
        return True

    except Exception as e:
        print(f"自动下载失败: {e}")
        print("将创建示例数据...")
        return create_qa_sample(output_path)


def create_qa_sample(output_path="data/qa.csv"):
    """创建问答示例数据"""
    samples = [
        (
            "中国首艘国产航母叫什么名字？",
            "2019年12月17日，经中央军委批准，中国第一艘国产航母命名为"
            "'中国人民解放军海军山东舰'，舷号为'17'。",
            "山东舰",
        ),
        (
            "《红楼梦》的作者是谁？",
            "《红楼梦》，中国古代章回体长篇小说，中国古典四大名著之一，"
            "一般认为是清代作家曹雪芹所著。",
            "曹雪芹",
        ),
        (
            "世界上最高的山峰是什么？",
            "珠穆朗玛峰是喜马拉雅山脉的主峰，位于中国与尼泊尔边境线上，"
            "海拔8848.86米，是世界最高峰。",
            "珠穆朗玛峰",
        ),
        (
            "北京故宫建于哪个朝代？",
            "北京故宫是中国明清两代的皇家宫殿，旧称紫禁城，位于北京中轴线的中心，"
            "始建于明成祖永乐四年（1406年）。",
            "明朝",
        ),
    ]
    extended = samples * 500
    df = pd.DataFrame(extended, columns=["question", "context", "answer"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"示例数据已保存到: {output_path} (样本数: {len(df)})")
    return True


if __name__ == "__main__":
    print("开始下载百度开源数据集...")
    print("如果自动下载失败，将自动生成示例数据用于测试。\n")

    download_sentiment_data()
    download_summary_data()
    download_qa_data()

    print("\n" + "=" * 50)
    print("所有数据集准备完成！")
    print("=" * 50)
    print("\n接下来可以:")
    print("  1. 训练模型: 在 main.py 中取消注释对应任务代码，运行 python main.py")
    print("  2. 启动服务: 训练完成后，运行 python app.py")
