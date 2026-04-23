"""
DeepSeek 内容审查模块

功能：
1. 对输入文本进行安全审查（涉黄、涉暴、涉政、辱骂等）
2. 对模型输出进行合规检查
3. 支持独立审查接口和全局审查中间件

使用方式：
    from content_review import ContentReviewer
    reviewer = ContentReviewer()
    result = reviewer.review("待审查文本")
"""

import os
import json
import httpx
from typing import Optional
from common import Config


class ReviewResult:
    """输入安全审查结果"""

    def __init__(self, passed: bool, category: str = "", reason: str = "", risk_level: str = "low"):
        self.passed = passed          # 是否通过审查
        self.category = category      # 违规类别（如色情、暴力、政治等）
        self.reason = reason          # 审查原因说明
        self.risk_level = risk_level  # 风险等级：low / medium / high

    def to_dict(self):
        return {
            "passed": self.passed,
            "category": self.category,
            "reason": self.reason,
            "risk_level": self.risk_level,
        }


class OutputReviewResult:
    """模型输出复核结果"""

    def __init__(self, passed: bool, score: int = 100, issues: list = None,
                 optimized_output: str = "", reason: str = "", original_output: str = ""):
        self.passed = passed              # 是否通过复核
        self.score = score                # 质量评分 0-100
        self.issues = issues or []        # 发现的问题列表
        self.optimized_output = optimized_output  # DeepSeek 优化后的输出
        self.reason = reason              # 复核结论说明
        self.original_output = original_output    # BART 原始输出

    def to_dict(self):
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "optimized_output": self.optimized_output,
            "reason": self.reason,
            "original_output": self.original_output,
        }


class ContentReviewer:
    """基于 DeepSeek LLM 的内容审查器"""

    SYSTEM_PROMPT = """你是一名严格的内容安全审查专家。请对用户提供的中文文本进行安全审查。

审查维度：
1. 色情内容（porn）- 包含露骨性描述、性行为暗示
2. 暴力恐怖（violence）- 包含血腥、虐待、恐怖主义内容
3. 政治敏感（political）- 包含敏感政治话题、颠覆性言论
4. 仇恨辱骂（hate）- 包含种族歧视、人身攻击、恶意辱骂
5. 违法犯罪（crime）- 包含教唆犯罪、毒品、赌博
6. 隐私泄露（privacy）- 包含身份证号、手机号、住址等敏感信息

输出格式（必须严格遵循JSON）：
{
    "passed": true/false,
    "category": "违规类别或空字符串",
    "reason": "具体原因说明",
    "risk_level": "low/medium/high"
}

规则：
- 轻微调侃、日常吐槽、正常批评 → passed: true, risk_level: low
- 明显违规内容 → passed: false, 给出具体 category 和 reason
- 无法确定时保守处理，标记为 medium 风险
- 只输出JSON，不要任何其他文字"""

    OUTPUT_REVIEW_PROMPT = """你是一名AI模型输出质量审查专家。请对BART模型生成的结果进行复核和优化。

当前任务类型：{task_type}
原始输入：
{input_text}

BART模型输出：
{model_output}

请从以下维度进行审查：
1. 准确性 - 结果是否正确，有无事实错误或逻辑矛盾
2. 完整性 - 是否遗漏了关键信息
3. 一致性 - 结果是否与原始输入保持一致
4. 语义理解 - 是否误解了用户意图（如反讽、隐喻等）
5. 优化建议 - 如何改进输出质量

输出格式（必须严格遵循JSON）：
{{
    "passed": true/false,
    "score": 0-100,
    "issues": ["问题1", "问题2"],
    "optimized_output": "优化后的结果（如无问题则填原输出）",
    "reason": "审查结论说明"
}}

规则：
- 结果完全正确 → passed: true, score: 90-100
- 有小瑕疵但不影响核心 → passed: true, score: 70-89, 给出优化建议
- 明显错误或遗漏 → passed: false, score: <70, 必须给出 optimized_output
- 反讽、隐喻等语言现象被误判时要特别指出
- 只输出JSON，不要任何其他文字"""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self.api_base = api_base or Config.DEEPSEEK_API_BASE
        self.model = model or Config.DEEPSEEK_MODEL
        self.enabled = bool(self.api_key)

    def review(self, text: str, timeout: float = 10.0) -> ReviewResult:
        """
        审查单条文本

        参数:
        - text: 待审查文本
        - timeout: 请求超时时间

        返回:
        - ReviewResult 审查结果对象
        """
        if not self.enabled:
            # 未配置 API Key，默认放行
            return ReviewResult(passed=True, reason="审查服务未配置，默认通过")

        if not text or not text.strip():
            return ReviewResult(passed=True, reason="空文本")

        try:
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": f"请审查以下文本：\n\n{text[:2000]}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 256,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # 提取 JSON
            result = self._parse_json(content)
            return ReviewResult(
                passed=result.get("passed", True),
                category=result.get("category", ""),
                reason=result.get("reason", ""),
                risk_level=result.get("risk_level", "low"),
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return ReviewResult(passed=True, reason="API Key 无效，审查服务不可用", risk_level="low")
            return ReviewResult(passed=True, reason=f"审查服务异常({e.response.status_code})，默认放行", risk_level="low")
        except Exception as e:
            return ReviewResult(passed=True, reason=f"审查服务异常({str(e)})，默认放行", risk_level="low")

    def review_output(self, task_type: str, input_text: str, model_output: str, timeout: float = 15.0) -> OutputReviewResult:
        """
        对 BART 模型输出结果进行复核和优化

        参数:
        - task_type: 任务类型（情感分析/文本摘要/智能问答）
        - input_text: 原始输入文本
        - model_output: BART 模型生成的输出
        - timeout: 请求超时时间

        返回:
        - OutputReviewResult 复核结果对象
        """
        if not self.enabled:
            return OutputReviewResult(
                passed=True,
                score=100,
                reason="复核服务未配置，默认通过",
                optimized_output=model_output,
                original_output=model_output,
            )

        try:
            prompt = self.OUTPUT_REVIEW_PROMPT.format(
                task_type=task_type,
                input_text=input_text[:1500],
                model_output=model_output[:1000],
            )
            response = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "你是一名AI模型输出质量审查专家。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            result = self._parse_json(content)
            return OutputReviewResult(
                passed=result.get("passed", True),
                score=result.get("score", 100),
                issues=result.get("issues", []),
                optimized_output=result.get("optimized_output", model_output),
                reason=result.get("reason", ""),
                original_output=model_output,
            )

        except Exception as e:
            return OutputReviewResult(
                passed=True,
                score=100,
                reason=f"复核服务异常({str(e)})，默认通过",
                optimized_output=model_output,
                original_output=model_output,
            )

    def _parse_json(self, text: str) -> dict:
        """从 LLM 输出中提取 JSON"""
        text = text.strip()
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取代码块中的 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 兜底：返回安全结果
            return {"passed": True, "reason": "解析异常，默认放行"}


# 全局审查器实例（单例）
_reviewer_instance: Optional[ContentReviewer] = None


def get_reviewer() -> ContentReviewer:
    """获取全局审查器实例"""
    global _reviewer_instance
    if _reviewer_instance is None:
        _reviewer_instance = ContentReviewer()
    return _reviewer_instance


def review_text(text: str) -> ReviewResult:
    """快捷函数：审查文本"""
    return get_reviewer().review(text)


def review_or_block(text: str) -> Optional[ReviewResult]:
    """
    审查文本，如果未通过则返回结果，通过返回 None
    用于全局中间件模式
    """
    result = get_reviewer().review(text)
    if not result.passed:
        return result
    return None
