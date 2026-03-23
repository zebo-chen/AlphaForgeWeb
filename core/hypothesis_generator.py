"""
假设生成模块 - 最小可执行框架
"""

import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from .llm_config import get_llm_config, create_llm_client


@dataclass
class Hypothesis:
    """因子假设数据结构"""
    id: str
    name: str
    logic: str
    economic_basis: str
    expected_direction: str
    applicable_regime: List[str]
    data_requirements: List[str]
    formula_idea: str
    confidence_score: float
    english_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "english_name": self.english_name,
            "logic": self.logic,
            "economic_basis": self.economic_basis,
            "expected_direction": self.expected_direction,
            "applicable_regime": self.applicable_regime,
            "data_requirements": self.data_requirements,
            "formula_idea": self.formula_idea,
            "confidence_score": self.confidence_score,
        }


class HypothesisGenerator:
    """假设生成器 - 最小可执行框架"""

    SYSTEM_PROMPT = """你是一位资深的利率债量化研究员，擅长生成市场择时因子假设。

【择时因子特点】
- 输出是时间序列信号（每个时间点一个值），用于判断市场方向
- 预测目标是市场指数的未来收益

【注意】
- 按照用户提出的数量要求进行假设生成

【输出格式】
```json
{
  "hypotheses": [
    {
      "name": "因子名称（中文）",
      "english_name": "EnglishName",
      "logic": "核心逻辑",
      "economic_basis": "经济学理论基础",
      "expected_direction": "positive/negative/nonlinear",
      "applicable_regime": ["bull", "bear", "range"],
      "data_requirements": ["price", "yield"],
      "formula_idea": "计算公式思路",
      "confidence_score": 8.5
    }
  ]
}
```"""

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化假设生成器

        Args:
            model_name: 模型名称（如 'claude-sonnet', 'gpt-4'）
            api_key: API密钥（可选，默认从环境变量读取）
        """
        self.config = get_llm_config(model_name)
        if api_key:
            self.config.api_key = api_key
        self.client = create_llm_client(self.config)
        self.messages: List[Dict] = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    @property
    def model(self) -> str:
        """兼容旧代码访问model"""
        return self.config.model

    def _detect_approval(self, user_input: str) -> bool:
        """用 LLM 判断用户是否在认可/确认当前假设"""
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=10,
            temperature=0,
            timeout=self.config.timeout,
            messages=[
                {"role": "system", "content": "你是意图识别助手，只输出 yes 或 no。"},
                {"role": "user", "content":
                    f"用户说：「{user_input}」\n"
                    "这句话是否在表示认可、确认或保存当前讨论的因子假设？只回答 yes 或 no。"}
            ]
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer.startswith("yes")

    def chat(self, user_input: str) -> str:
        """
        聊天模式：多轮对话，流式输出。
        若检测到用户认可意图，自动提示调用 extract()。

        Returns:
            LLM 的完整回复
        """
        # 先判断意图（非流式，快速）
        if self._detect_approval(user_input):
            print("💡 检测到认可意图，调用 gen.extract() 可将当前假设保存为结构化数据\n")

        self.messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.7,
                timeout=self.config.timeout,
                messages=self.messages,
                stream=True
            )

            content = ""
            print("🤖 ", end="", flush=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    content += delta
                    print(delta, end="", flush=True)
            print()

            self.messages.append({"role": "assistant", "content": content})
            return content

        except Exception as e:
            print(f"\n❌ 调用失败: {e}")
            return ""

    def extract(self, require_approval: bool = False) -> List[Hypothesis]:
        """
        从当前对话历史中提炼假设。

        Args:
            require_approval: 是否只提取被明确认可的假设。如果为False，会提取对话中提到的所有假设。

        Returns:
            List[Hypothesis]: 假设列表（可能为空）
        """
        if require_approval:
            extraction_prompt = (
                "请分析以上对话，提取出所有被用户明确认可/确认的因子假设。\n"
                "如果没有认可的假设，返回空数组。\n"
                "严格按以下 JSON 格式输出，不要包含其他内容：\n"
                '```json\n'
                '{"hypotheses":[\n'
                '  {"name":"","english_name":"","logic":"","economic_basis":"",'
                '"expected_direction":"positive/negative/nonlinear",'
                '"applicable_regime":["bull","bear","range"],'
                '"data_requirements":[],"formula_idea":"","confidence_score":8.0}\n'
                ']}\n'
                '```'
            )
        else:
            # 提取对话中提到的所有假设，不管是否确认
            extraction_prompt = (
                "请分析以上对话，提取出AI在回复中提到的所有因子假设（包括已确认的和未确认的）。\n"
                "如果没有任何假设，返回空数组。\n"
                "严格按以下 JSON 格式输出，不要包含其他内容：\n"
                '```json\n'
                '{"hypotheses":[\n'
                '  {"name":"","english_name":"","logic":"","economic_basis":"",'
                '"expected_direction":"positive/negative/nonlinear",'
                '"applicable_regime":["bull","bear","range"],'
                '"data_requirements":[],"formula_idea":"","confidence_score":8.0}\n'
                ']}\n'
                '```'
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                timeout=self.config.timeout,
                messages=self.messages + [{"role": "user", "content": extraction_prompt}]
            )
            content = resp.choices[0].message.content

            m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            data = json.loads(m.group(1) if m else content)

            hypotheses = []
            timestamp = datetime.now().strftime('%H%M%S')
            for i, h in enumerate(data.get("hypotheses", [])):
                hyp = Hypothesis(
                    id=f"HYP_{timestamp}_{i+1:02d}",
                    name=h.get("name", f"因子{i+1}"),
                    english_name=h.get("english_name", ""),
                    logic=h.get("logic", ""),
                    economic_basis=h.get("economic_basis", ""),
                    expected_direction=h.get("expected_direction", "positive"),
                    applicable_regime=h.get("applicable_regime", []),
                    data_requirements=h.get("data_requirements", []),
                    formula_idea=h.get("formula_idea", ""),
                    confidence_score=float(h.get("confidence_score", 7.0)),
                )
                hypotheses.append(hyp)

            print(f"✅ 已提取 {len(hypotheses)} 个假设")
            return hypotheses

        except Exception as e:
            print(f"❌ 提取失败: {e}")
            return []

    def reset(self):
        """清空对话历史"""
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def generate(self, user_input: str) -> List[Hypothesis]:
        """
        生成因子假设 (Streaming模式，智能判断数量)

        Parameters:
            user_input: 用户输入（市场背景/需求描述，可包含数量要求如"生成3个因子"）

        Returns:
            List[Hypothesis]: 假设列表
        """
        prompt = f"""{user_input}

【重要】请根据我的要求判断生成数量：
- 如果我明确说了数量（如"生成3个"、"来5个"），请按我说的数量生成
- 如果我没有明确说数量，请只生成1个假设
- 最多不要超过5个"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.3,
                timeout=self.config.timeout,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            # 流式收集完整响应
            content = ""
            print("🔄 生成中... ", end="", flush=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    content += delta
                    print("▌", end="", flush=True)
            print(" ✓")
            print(content)

            hypotheses = self._parse_response(content)
            return hypotheses

        except Exception as e:
            print(f"\n❌ LLM调用失败: {e}")
            return self._get_demo_hypotheses()

    def _parse_response(self, content: str) -> List[Hypothesis]:
        """解析LLM响应"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            data = json.loads(json_str)
            hypotheses = []

            for i, h in enumerate(data.get("hypotheses", [])):
                hypothesis = Hypothesis(
                    id=f"HYP_{i+1:03d}",
                    name=h["name"],
                    english_name=h.get("english_name", ""),
                    logic=h["logic"],
                    economic_basis=h["economic_basis"],
                    expected_direction=h.get("expected_direction", "positive"),
                    applicable_regime=h.get("applicable_regime", ["bull", "bear", "range"]),
                    data_requirements=h["data_requirements"],
                    formula_idea=h["formula_idea"],
                    confidence_score=float(h.get("confidence_score", 7.0)),
                )
                hypotheses.append(hypothesis)

            return hypotheses

        except Exception as e:
            print(f"解析失败: {e}")
            return self._get_demo_hypotheses()

    def _get_demo_hypotheses(self) -> List[Hypothesis]:
        """返回示例假设"""
        return [
            Hypothesis(
                id="HYP_001",
                name="收益率偏离度反转因子",
                english_name="YieldDeviationReversal",
                logic="当债券收益率偏离其长期均值达到一定程度时，存在均值回归效应",
                economic_basis="利率期限结构均值回归理论",
                expected_direction="negative",
                applicable_regime=["range", "bear"],
                data_requirements=["yield"],
                formula_idea="计算当前收益率与N日均值的偏离度，取负值作为因子",
                confidence_score=8.5,
            ),
            Hypothesis(
                id="HYP_002",
                name="流动性冲击因子",
                english_name="LiquidityShock",
                logic="高换手率伴随的价格下跌往往反映流动性冲击，后续存在修复",
                economic_basis="流动性溢价理论",
                expected_direction="positive",
                applicable_regime=["bull", "bear"],
                data_requirements=["turnover", "price"],
                formula_idea="识别高换手率+价格下跌的时点，构建反转信号",
                confidence_score=7.8,
            ),
            Hypothesis(
                id="HYP_003",
                name="久期调整动量因子",
                english_name="DurationAdjustedMomentum",
                logic="经久期调整后的价格动量更能反映真实的趋势强度",
                economic_basis="风险调整收益理论",
                expected_direction="positive",
                applicable_regime=["bull", "bear"],
                data_requirements=["price", "duration"],
                formula_idea="价格变化率除以久期，进行标准化处理",
                confidence_score=7.5,
            )
        ]
