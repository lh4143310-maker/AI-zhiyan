import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from common import Config
from models_def import SentimentModel, Seq2SeqModel
from content_review import get_reviewer

# ========== 加载模型 ==========
print("正在加载模型...")

sentiment_model = SentimentModel(Config.BART_PATH, Config.SENTIMENT_LABELS)
try:
    sentiment_model.load_params("finetuned/sentiment.pt")
except Exception:
    print("[警告] 情感模型未找到，请先在 main.py 中训练")

summary_model = Seq2SeqModel(Config.BART_PATH)
try:
    summary_model.load_params("finetuned/summarize.pt")
except Exception:
    print("[警告] 摘要模型未找到，请先在 main.py 中训练")

qa_model = Seq2SeqModel(Config.BART_PATH)
try:
    qa_model.load_params("finetuned/qa.pt")
except Exception:
    print("[警告] 问答模型未找到，请先在 main.py 中训练")

print("模型加载完成")

# ========== FastAPI 应用 ==========
app = FastAPI(
    title="智言中文 - 智能文本分析与问答平台",
    description="基于 BART 的中文情感分类、文本摘要与智能问答系统",
    version="1.0.0",
)

# 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory="templates"), name="static")


# ========== 请求/响应模型 ==========
class SentimentRequest(BaseModel):
    text: str = Field(..., example="这部电影真的太棒了！")


class SummaryRequest(BaseModel):
    text: str = Field(..., example="需要摘要的长文本...")


class QARequest(BaseModel):
    question: str = Field(..., example="中国首艘国产航母叫什么名字？")
    context: str = Field(..., example="相关背景文章...")


class BatchRequest(BaseModel):
    task: str = Field(..., example="sentiment")
    texts: list[str] = Field(..., example=["文本1", "文本2"])


class ReviewRequest(BaseModel):
    text: str = Field(..., example="待审查的文本内容")


class ReviewResponse(BaseModel):
    passed: bool = Field(..., example=True)
    category: str = Field(..., example="")
    reason: str = Field(..., example="审查通过")
    risk_level: str = Field(..., example="low")


class OutputReviewRequest(BaseModel):
    task_type: str = Field(..., example="情感分析")
    input_text: str = Field(..., example="原始输入文本")
    model_output: str = Field(..., example="BART模型输出结果")


class OutputReviewResponse(BaseModel):
    passed: bool = Field(..., example=True)
    score: int = Field(..., example=95)
    issues: list[str] = Field(..., example=[])
    optimized_output: str = Field(..., example="优化后的结果")
    reason: str = Field(..., example="BART输出正确，无需修正")
    original_output: str = Field(..., example="BART原始输出")


# ========== 审查中间件 ==========
reviewer = get_reviewer()


# ========== API 路由 ==========
@app.get("/")
async def homepage():
    return FileResponse("templates/index.html")


def _auto_review(task_type: str, input_text: str, model_output: str):
    """自动调用 DeepSeek 复核，返回复核结果字典"""
    if not reviewer.enabled:
        return None
    try:
        review = reviewer.review_output(task_type, input_text, model_output, timeout=15.0)
        return review.to_dict()
    except Exception:
        return None


@app.post("/sentiment")
async def sentiment_analysis(request: SentimentRequest):
    """情感分析（BART Agent + DeepSeek Agent 自动协同）"""
    result = sentiment_model.predict(request.text, Config.DEVICE)
    sentiment_model.eval()
    inputs = sentiment_model.tokenizer(
        request.text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(Config.DEVICE)
    with torch.no_grad():
        logits = sentiment_model(inputs["input_ids"], inputs["attention_mask"])["logits"]
        probs = torch.softmax(logits, dim=1)
        confidence = probs.max().item()

    review = _auto_review("情感分析", request.text, result)
    return {
        "sentiment": result,
        "confidence": f"{confidence:.4f}",
        "review": review,
    }


@app.post("/summarize")
async def text_summarize(request: SummaryRequest):
    """文本摘要（BART Agent + DeepSeek Agent 自动协同）"""
    result = summary_model.predict(request.text, Config.DEVICE)
    review = _auto_review("文本摘要", request.text, result)
    return {
        "summary": result,
        "review": review,
    }


@app.post("/qa")
async def question_answer(request: QARequest):
    """智能问答（BART Agent + DeepSeek Agent 自动协同）"""
    input_text = f"问题：{request.question} 上下文：{request.context}"
    result = qa_model.predict(input_text, Config.DEVICE)
    review = _auto_review("智能问答", input_text, result)
    return {
        "answer": result,
        "review": review,
    }


@app.post("/batch")
async def batch_process(request: BatchRequest):
    """批量处理"""
    if request.task == "sentiment":
        results = sentiment_model.predict(request.texts, Config.DEVICE)
        return {"results": results}
    elif request.task == "summarize":
        results = summary_model.predict(request.texts, Config.DEVICE)
        return {"results": results}
    elif request.task == "qa":
        results = qa_model.predict(request.texts, Config.DEVICE)
        return {"results": results}
    return {"error": "未知任务类型"}


@app.post("/review", response_model=ReviewResponse)
async def content_review(request: ReviewRequest):
    """DeepSeek 内容审查接口"""
    result = reviewer.review(request.text)
    return ReviewResponse(
        passed=result.passed,
        category=result.category,
        reason=result.reason,
        risk_level=result.risk_level,
    )


@app.post("/review/output", response_model=OutputReviewResponse)
async def review_model_output(request: OutputReviewRequest):
    """DeepSeek 对 BART 输出结果进行复核（多智能体协同核心接口）"""
    result = reviewer.review_output(
        task_type=request.task_type,
        input_text=request.input_text,
        model_output=request.model_output,
    )
    return OutputReviewResponse(
        passed=result.passed,
        score=result.score,
        issues=result.issues,
        optimized_output=result.optimized_output,
        reason=result.reason,
        original_output=result.original_output,
    )


@app.post("/agent/collaborate")
async def agent_collaborate(request: dict):
    """
    双模型协作接口：BART Agent 生成 + DeepSeek Agent 复核

    请求体：
    {
        "task": "sentiment|summarize|qa",
        "text": "...",           // sentiment/summarize 用
        "question": "...",       // qa 用
        "context": "...",        // qa 用
        "enable_review": true    // 是否启用 DeepSeek 复核
    }
    """
    task = request.get("task", "sentiment")
    enable_review = request.get("enable_review", True)

    # Step 1: BART Agent 生成结果
    if task == "sentiment":
        text = request.get("text", "")
        if not text:
            return {"error": "缺少 text 参数"}
        bart_result = sentiment_model.predict(text, Config.DEVICE)
        sentiment_model.eval()
        inputs = sentiment_model.tokenizer(
            text, max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(Config.DEVICE)
        with torch.no_grad():
            logits = sentiment_model(inputs["input_ids"], inputs["attention_mask"])["logits"]
            probs = torch.softmax(logits, dim=1)
            confidence = probs.max().item()
        output = {
            "sentiment": bart_result,
            "confidence": f"{confidence:.4f}",
        }
        raw_output = bart_result

    elif task == "summarize":
        text = request.get("text", "")
        if not text:
            return {"error": "缺少 text 参数"}
        bart_result = summary_model.predict(text, Config.DEVICE)
        output = {"summary": bart_result}
        raw_output = bart_result

    elif task == "qa":
        question = request.get("question", "")
        context = request.get("context", "")
        if not question or not context:
            return {"error": "缺少 question 或 context 参数"}
        input_text = f"问题：{question} 上下文：{context}"
        bart_result = qa_model.predict(input_text, Config.DEVICE)
        output = {"answer": bart_result}
        raw_output = bart_result

    else:
        return {"error": "未知任务类型"}

    result = {
        "task": task,
        "bart_output": output,
        "deepseek_review": None,
        "final_output": output,
        "collaboration": "BART 单独输出（未启用复核）",
    }

    # Step 2: DeepSeek Agent 复核（如果启用）
    if enable_review and reviewer.enabled:
        input_for_review = request.get("text", "") or f"问题：{request.get('question','')} 上下文：{request.get('context','')}"
        task_name = {"sentiment": "情感分析", "summarize": "文本摘要", "qa": "智能问答"}.get(task, task)
        review = reviewer.review_output(task_name, input_for_review, raw_output)
        result["deepseek_review"] = review.to_dict()
        result["collaboration"] = "BART 生成 + DeepSeek 复核"

        # 如果 DeepSeek 认为需要优化，使用优化后的输出
        if not review.passed and review.optimized_output:
            if task == "sentiment":
                result["final_output"] = {"sentiment": review.optimized_output, "confidence": "复核修正"}
            elif task == "summarize":
                result["final_output"] = {"summary": review.optimized_output}
            elif task == "qa":
                result["final_output"] = {"answer": review.optimized_output}

    return result


@app.get("/review/status")
async def review_status():
    """审查服务状态"""
    return {
        "enabled": reviewer.enabled,
        "mode": "输出复核（多智能体协同）",
        "model": Config.DEEPSEEK_MODEL if reviewer.enabled else None,
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "device": str(Config.DEVICE)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
