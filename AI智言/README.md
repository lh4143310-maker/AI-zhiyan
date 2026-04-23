# 智言中文 - 多智能体协同中文 NLP 平台

> 基于 BART + DeepSeek 的双 Agent 协同架构，实现情感分析、文本摘要、智能问答三大核心任务。

---

## 项目概述

智言中文是一个面向中文自然语言处理的多任务 AI 平台，采用**多智能体协同架构**：

- **BART Agent**：基于 `fnlp/bart-base-chinese` 预训练模型，负责核心 NLP 任务的生成与推理
- **DeepSeek Agent**：基于 DeepSeek 大语言模型，负责对 BART 输出结果进行质量复核与优化

两个 Agent 协同工作，用户只需一次请求，即可自动完成"生成 → 复核 → 输出"的完整流程。

---

## 核心特性

### 1. 多智能体自动协同
用户发起一次 API 请求，系统自动完成：
```
用户输入 → BART Agent 生成结果 → DeepSeek Agent 自动复核 → 返回最终结果 + 复核报告
```

### 2. 三大 NLP 任务
| 任务 | 模型架构 | 评估指标 | 测试表现 |
|------|---------|---------|---------|
| 情感分析 | BART Encoder + 分类器 | Accuracy / F1 | 100% |
| 文本摘要 | 完整 BART Seq2Seq | ROUGE-1/2/L | 1.0 |
| 智能问答 | 完整 BART Seq2Seq | ROUGE-1/2/L | 0.9857 |

### 3. 多样化训练数据
使用模板变量生成覆盖多领域的训练数据，每个任务 3000 条唯一样本：
- **情感分析**：电商购物、餐饮美食、影视娱乐、旅游出行、数码产品、服务态度
- **文本摘要**：科技、体育、财经、国际、社会、文化
- **智能问答**：历史、地理、科技、文学、艺术、生活常识

### 4. 完整的 REST API
基于 FastAPI 构建，提供交互式 Swagger 文档，支持外部系统集成。

---

## 技术架构

### 多智能体协同流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户请求                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│   BART Agent        │         │   DeepSeek Agent    │
│   (生成模型)         │         │   (复核模型)         │
├─────────────────────┤         ├─────────────────────┤
│ · 情感分类          │         │ · 准确性审查         │
│ · 文本摘要          │         │ · 完整性审查         │
│ · 智能问答          │         │ · 一致性审查         │
└──────────┬──────────┘         │ · 语义理解审查       │
           │                    │ · 质量评分 0-100     │
           │                    └──────────┬──────────┘
           │                               │
           └──────────────┬────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │      最终结果 + 复核报告        │
           │  · BART 原始输出               │
           │  · DeepSeek 质量评分            │
           │  · 复核结论与建议               │
           └──────────────────────────────┘
```

### 模型架构

#### 情感分类模型（SentimentModel）
```python
BART Encoder → [CLS] Token 隐藏状态 → Linear 分类器 →  Softmax
```
- 仅使用 BART 的 Encoder 部分
- 取第一个 token 的隐藏状态作为句子表示
- 输出 2 分类：积极 / 消极

#### 序列生成模型（Seq2SeqModel）
```python
输入文本 → BART Encoder → BART Decoder → LM Head → 生成文本
```
- 使用完整 BART 的 Encoder-Decoder 架构
- 束搜索（Beam Search）生成，num_beams=4
- 支持文本摘要和智能问答两个任务

---

## 项目结构

```
AI智言/
├── app.py                 # FastAPI 服务入口（多智能体协同 API）
├── common.py              # 全局配置（超参数、DeepSeek 配置等）
├── models_def.py          # 模型定义（SentimentModel、Seq2SeqModel）
├── train.py               # 训练器（Trainer、SentimentTrainer、Seq2SeqTrainer）
├── preprocess.py          # 数据预处理与 DataLoader 构建
├── main.py                # 训练脚本入口（三任务顺序训练）
├── content_review.py      # DeepSeek Agent 复核模块
├── data_generate.py       # 多样化训练数据生成脚本
├── data_download.py       # （已废弃）原数据下载脚本
├── requirements.txt       # Python 依赖
│
├── data/                  # 训练数据目录
│   ├── sentiment.csv      # 情感分析数据（3000条）
│   ├── summary.csv        # 文本摘要数据（3000条）
│   └── qa.csv             # 智能问答数据（3000条）
│
├── finetuned/             # 微调后的模型权重
│   ├── sentiment.pt       # 情感分类模型
│   ├── summarize.pt       # 文本摘要模型
│   └── qa.pt              # 智能问答模型
│
├── templates/             # 前端页面
│   └── index.html         # Web UI（三任务 + 复核展示）
│
├── logs/                  # TensorBoard 训练日志
└── pretrained/            # 预训练模型缓存（首次运行时自动下载）
```

---

## 快速开始

### 环境准备

```bash
# 1. 克隆项目
cd AI智言

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 生成训练数据

```bash
python data_generate.py
```

将自动生成 `data/sentiment.csv`、`data/summary.csv`、`data/qa.csv` 三个文件。

### 训练模型

```bash
# 设置 HuggingFace 镜像（国内用户推荐）
set HF_ENDPOINT=https://hf-mirror.com  # Windows
# 或 export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac

# 运行训练（自动依次训练三个任务）
python main.py
```

训练完成后，模型权重将保存到 `finetuned/` 目录。

### 启动服务

```bash
# 配置 DeepSeek API Key（可选，用于启用复核功能）
set DEEPSEEK_API_KEY=your_api_key_here  # Windows
# 或 export DEEPSEEK_API_KEY=your_api_key_here  # Linux/Mac

# 启动 FastAPI 服务
python app.py
```

服务启动后访问：
- Web UI：`http://localhost:8089`
- Swagger API 文档：`http://localhost:8089/docs`
- ReDoc 文档：`http://localhost:8089/redoc`

---

## API 接口文档

### 核心任务接口（自动协同）

#### 1. 情感分析

```http
POST /sentiment
Content-Type: application/json

{
  "text": "这部电影真的太棒了！"
}
```

**响应示例：**
```json
{
  "sentiment": "积极",
  "confidence": "0.9998",
  "review": {
    "passed": true,
    "score": 95,
    "issues": [],
    "optimized_output": "积极",
    "reason": "BART模型输出准确识别了原始输入中的正面情感...",
    "original_output": "积极"
  }
}
```

#### 2. 文本摘要

```http
POST /summarize
Content-Type: application/json

{
  "text": "需要摘要的长文本..."
}
```

**响应示例：**
```json
{
  "summary": "摘要结果",
  "review": {
    "passed": true,
    "score": 95,
    "issues": [],
    "reason": "输出准确、完整、一致...",
    "original_output": "摘要结果"
  }
}
```

#### 3. 智能问答

```http
POST /qa
Content-Type: application/json

{
  "question": "中国首艘国产航母叫什么名字？",
  "context": "2019年12月17日，中国第一艘国产航母命名为'中国人民解放军海军山东舰'..."
}
```

**响应示例：**
```json
{
  "answer": "山东舰",
  "review": {
    "passed": true,
    "score": 95,
    "issues": [],
    "reason": "模型输出准确、完整...",
    "original_output": "山东舰"
  }
}
```

### 批量处理接口

```http
POST /batch
Content-Type: application/json

{
  "task": "sentiment",
  "texts": ["文本1", "文本2", "文本3"]
}
```

### 状态查询接口

```http
GET /health
```

```http
GET /review/status
```

---

## 配置说明

所有配置集中在 `common.py` 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BART_PATH` | `fnlp/bart-base-chinese` | 预训练模型路径 |
| `MAX_EXAMPLES` | `3000` | 训练数据量 |
| `BATCH_SIZE` | `8` | 训练批次大小 |
| `EPOCHS` | `1` | 训练轮数 |
| `LEARNING_RATE` | `5e-5` | 学习率 |
| `DEEPSEEK_API_KEY` | 环境变量读取 | DeepSeek API 密钥 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名称 |

### 环境变量

| 变量名 | 说明 |
|--------|------|
| `HF_ENDPOINT` | HuggingFace 镜像地址，国内用户建议设为 `https://hf-mirror.com` |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥，用于启用复核功能 |
| `ENABLE_GLOBAL_REVIEW` | 是否启用全局输入审查，默认 `false` |

---

## 模型训练详情

### 训练参数

```python
MAX_EXAMPLES = 3000      # 每任务训练样本数
BATCH_SIZE = 8           # 批次大小
EPOCHS = 1               # 训练轮数
LEARNING_RATE = 5e-5     # 学习率
TRAIN_RATIO = 0.8        # 训练集比例
TEST_RATIO = 0.1         # 测试集比例
```

### 训练结果

| 任务 | 训练 Loss | 验证指标 | 测试指标 |
|------|----------|---------|---------|
| 情感分类 | 0.1485 | Accuracy: 1.0000 | Accuracy: 1.0 |
| 文本摘要 | 0.0579 | ROUGE-1: 1.0000 | ROUGE-1: 1.0 |
| 智能问答 | 0.0627 | ROUGE-1: 0.9888 | ROUGE-1: 0.9857 |

### 自定义训练

编辑 `main.py` 可以调整训练行为：
- 修改 `train=True/False` 控制是否训练
- 修改 `test=True/False` 控制是否测试
- 修改 `inference=True/False` 控制是否推理示例

---

## 多智能体协同原理

### DeepSeek 复核维度

当 BART Agent 生成结果后，DeepSeek Agent 从以下 5 个维度进行复核：

1. **准确性**：结果是否正确，有无事实错误或逻辑矛盾
2. **完整性**：是否遗漏了关键信息
3. **一致性**：结果是否与原始输入保持一致
4. **语义理解**：是否误解了用户意图（如反讽、隐喻等）
5. **优化建议**：如何改进输出质量

### 评分规则

| 评分区间 | 含义 | 处理策略 |
|---------|------|---------|
| 90-100 | 完全正确 | 直接通过 |
| 70-89 | 有小瑕疵 | 通过，附带优化建议 |
| < 70 | 明显错误 | 标记未通过，提供优化输出 |

### 降级策略

当 DeepSeek API 不可用时（网络故障、API Key 无效等），系统自动降级：
- 复核功能静默失效
- 直接返回 BART 原始输出
- 不影响主流程可用性

---

## 前端界面

项目包含一个现代化的 Web UI，支持：

- **三任务切换**：情感分析 / 文本摘要 / 智能问答
- **一键分析**：输入文本后自动完成生成 + 复核
- **实时展示**：BART 结果与 DeepSeek 复核信息同步呈现
- **快捷示例**：内置多个领域的示例文本
- **复制功能**：一键复制分析结果

---

## 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch | >= 2.0.0 |
| 预训练模型 | Transformers | >= 4.35.0 |
| 基础模型 | fnlp/bart-base-chinese | — |
| 复核模型 | DeepSeek API | deepseek-chat |
| Web 框架 | FastAPI | >= 0.104.0 |
| ASGI 服务器 | Uvicorn | >= 0.24.0 |
| 数据验证 | Pydantic | >= 2.5.0 |
| 训练监控 | TensorBoard | >= 2.14.0 |
| 评估指标 | scikit-learn / rouge-score | — |

---

## 部署建议

### 本地开发

```bash
python app.py
```

### 生产环境

```bash
# 使用多 worker 部署
uvicorn app:app --host 0.0.0.0 --port 8089 --workers 4
```

### Docker 部署（示例）

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8089

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8089"]
```

---

## 常见问题

### Q: 首次启动时模型下载慢？
A: 设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com      # Windows
```

### Q: DeepSeek 复核没有生效？
A: 检查环境变量 `DEEPSEEK_API_KEY` 是否正确设置。未配置时系统会静默跳过复核。

### Q: 模型文件不存在？
A: 需要先运行 `python main.py` 训练模型，或准备预训练权重放到 `finetuned/` 目录。

### Q: 如何在 GPU 上运行？
A: 确保 PyTorch 安装了 CUDA 版本，程序会自动检测 GPU 并使用。

---

## 许可证

本项目仅供学习和研究使用。

---

## 致谢

- [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese) - 中文 BART 预训练模型
- [DeepSeek](https://deepseek.com/) - 大语言模型 API
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - NLP 模型框架
