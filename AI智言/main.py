from datetime import datetime
from preprocess import process
from common import Config, set_seed
from train import SentimentTrainer, Seq2SeqTrainer
from torch.utils.tensorboard.writer import SummaryWriter
from models_def import SentimentModel, Seq2SeqModel

set_seed(42)

device = Config.DEVICE
model_name = Config.BART_PATH
label_list = Config.SENTIMENT_LABELS


def model_go(
    task,
    data_path,
    train=False,
    test=False,
    inference=False,
    texts=None,
    model_params_path=None,
):
    """
    模型训练、验证、测试、推理统一入口

    参数:
    - task: 任务类型 ["sentiment", "summarize", "qa"]
    - data_path: 数据文件路径
    - train: 是否训练
    - test: 是否测试
    - inference: 是否推理
    - texts: 推理文本列表
    - model_params_path: 模型参数路径
    """
    assert task in ["sentiment", "summarize", "qa"], "任务类型错误"

    if task == "sentiment":
        model = SentimentModel(model_name, label_list)
        trainer = SentimentTrainer(
            model, device, Config.EPOCHS, Config.LEARNING_RATE, Config.CHECKPOINT_STEPS
        )
        label_list_arg = label_list
    else:
        model = Seq2SeqModel(model_name)
        trainer = Seq2SeqTrainer(
            model, device, Config.EPOCHS, Config.LEARNING_RATE, Config.CHECKPOINT_STEPS
        )
        label_list_arg = None

    if train or test:
        dataloader = process(
            task=task,
            data_path=data_path,
            max_examples=Config.MAX_EXAMPLES,
            batch_size=Config.BATCH_SIZE,
            tokenizer=model.tokenizer,
            train_ratio=Config.TRAIN_RATIO,
            test_ratio=Config.TEST_RATIO,
            label_list=label_list_arg,
        )

    writer = None
    this_id = datetime.now().strftime("%Y%m%d%H%M%S")

    if model_params_path:
        model.load_params(model_params_path)

    if train:
        writer = SummaryWriter(f"logs/{task}-{this_id}")
        # 保存到标准路径，覆盖旧模型
        save_path = f"finetuned/{task}.pt"
        trainer(dataloader, save_path, writer)

    if test:
        trainer(dataloader, writer=writer, is_test=True)

    if writer:
        writer.close()

    if inference and texts:
        return model.predict(texts, device)


if __name__ == "__main__":
    # ========== 情感分类任务 ==========
    print("\n" + "=" * 40)
    print("【任务1】情感分类")
    print("=" * 40)

    sentiment_texts = [
        "这部电影真的太棒了，演员演技在线，剧情紧凑！",
        "服务态度极差，等了一个小时还没上菜，再也不会来了。",
        "今天天气不错，心情也挺好的。",
    ]

    # 训练 + 测试 + 推理示例
    model_go(
        task="sentiment",
        data_path=Config.SENTIMENT_DATA_PATH,
        train=True,
        test=True,
        inference=True,
        texts=sentiment_texts,
        model_params_path="finetuned/sentiment.pt",
    )

    # ========== 文本摘要任务 ==========
    print("\n" + "=" * 40)
    print("【任务2】文本摘要")
    print("=" * 40)

    summary_texts = [
        "北京时间4月22日，中国航天科技集团发布消息称，长征七号遥八运载火箭已完成出厂前所有研制工作，"
        "于近日安全运抵文昌航天发射场。后续，该火箭将与先期已运抵的天舟七号货运飞船一起，"
        "按计划开展发射场区总装和测试工作，择机发射。",
    ]

    # 训练 + 测试 + 推理示例
    model_go(
        task="summarize",
        data_path=Config.SUMMARY_DATA_PATH,
        train=True,
        test=True,
        inference=True,
        texts=summary_texts,
        model_params_path="finetuned/summarize.pt",
    )

    # ========== 智能问答任务 ==========
    print("\n" + "=" * 40)
    print("【任务3】智能问答")
    print("=" * 40)

    qa_texts = [
        "问题：中国首艘国产航母叫什么名字？ 上下文："
        "2019年12月17日，经中央军委批准，中国第一艘国产航母命名为"
        "'中国人民解放军海军山东舰'，舷号为'17'。",
    ]

    # 训练 + 测试 + 推理示例
    model_go(
        task="qa",
        data_path=Config.QA_DATA_PATH,
        train=True,
        test=True,
        inference=True,
        texts=qa_texts,
        model_params_path="finetuned/qa.pt",
    )

    print("\n提示: 如需重新生成训练数据，请运行 python data_generate.py")
