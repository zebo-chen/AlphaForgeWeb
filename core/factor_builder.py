"""
因子构建模块 - 将假设转化为可执行代码
"""

import os
import re
import json
import hashlib
import traceback
from typing import List, Optional
from dataclasses import dataclass
from openai import OpenAI
from .llm_config import get_llm_config, create_llm_client
import pandas as pd
import numpy as np


@dataclass
class FactorCode:
    """因子代码数据结构"""
    name: str
    code: str
    description: dict
    safe_to_execute: bool = False
    english_name: str = ""  # 英文名称

    def show(self):
        """格式化打印代码（带语法高亮）"""
        try:
            from IPython.display import display, Markdown
            display(Markdown(f"```python\n{self.code}\n```"))
        except ImportError:
            # 非 Notebook 环境
            print(f"\n{'='*60}")
            print(f"📄 因子代码: {self.name}")
            print(f"{'='*60}")
            print(self.code)
            print(f"{'='*60}\n")

    def __repr__(self) -> str:
        return f"FactorCode(name='{self.name}', lines={self.code.count(chr(10))})"


class FactorBuilder:
    """因子构建器"""

    BASE_TEMPLATE = '''
import pandas as pd
import numpy as np
from typing import Dict, Optional

class {class_name}:
    """
    {factor_name} - 择时因子（单资产时间序列）

    {description}
    """

    def __init__(self, params: Optional[Dict] = None):
        default_params = {default_params}
        self.params = {{**default_params, **(params or {{}})}}
        self.name = "{factor_name}"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算择时因子值（单资产时间序列）

        Parameters:
            data: DataFrame with columns: {required_columns}
                  index: datetime（支持日频、小时频、分钟频等任意bar频率）

        Returns:
            Series: index=datetime, values=factor_value（因子值时间序列）
        """
        try:
            factor = self._compute(data)

            # 确保返回Series格式
            if isinstance(factor, pd.DataFrame):
                factor = factor.iloc[:, 0]
            elif not isinstance(factor, pd.Series):
                factor = pd.Series(factor, index=data.index)

            factor.name = self.name
            return factor

        except Exception as e:
            print(f"计算失败: {{e}}")
            return pd.Series(dtype=float)

    def _compute(self, data: pd.DataFrame):
        """核心计算逻辑"""
{calculation_code}

    def get_description(self) -> Dict:
        return {description_dict}
'''

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        初始化因子构建器

        Args:
            model_name: 模型名称（如 'claude-sonnet', 'gpt-4', 'kimi'）
            api_key: API密钥（可选，默认从环境变量读取）
        """
        # 使用配置管理器加载配置
        self.config = get_llm_config(model_name)

        # 如果传入api_key，覆盖配置中的值
        if api_key:
            self.config.api_key = api_key

        # 创建客户端
        self.client = create_llm_client(self.config)

    @property
    def model(self) -> str:
        """兼容旧代码访问model"""
        return self.config.model

    def build(self, hypothesis, columns: Optional[List[str]] = None,
              data: pd.DataFrame = None, max_attempts: int = 3) -> FactorCode:
        """
        构建因子代码

        Parameters:
            hypothesis: Hypothesis对象
            columns: 可选，实际数据的列名列表，用于生成匹配的代码
            data: 可选，如果提供，会在该数据上验证执行；失败时自动让LLM修复
            max_attempts: 最大尝试次数（仅当data提供时有效）

        Returns:
            FactorCode: 生成的因子代码
        """
        if columns is None and data is not None:
            columns = list(data.columns)

        prompt = self._build_prompt(hypothesis, columns)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                messages=[{"role": "user", "content": prompt}]
            )

            code = self._parse_code(response.choices[0].message.content)
            safe_code = self._sanitize_code(code)

            factor_code = FactorCode(
                name=hypothesis.name,
                code=safe_code,
                description=hypothesis.to_dict(),
                safe_to_execute=True
            )

        except Exception as e:
            print(f"代码生成失败: {e}")
            factor_code = self._build_fallback_code(hypothesis)

        # 如果提供了data，进入调试循环验证执行
        if data is not None:
            factor_code = self._execute_with_retry(factor_code, data, hypothesis, max_attempts)

        return factor_code

    def _build_prompt(self, hypothesis, columns: Optional[List[str]] = None) -> str:
        """构建代码生成Prompt"""
        if columns:
            col_info = f"实际可用列名: {columns}\n⚠️ 代码中只能使用以上列名，不得使用其他列"
        else:
            col_info = f"预期列名: {hypothesis.data_requirements}（实际列名以传入的data为准）"

        return f"""请基于以下因子假设，生成完整的Python计算代码。

【因子假设】
名称: {hypothesis.name}
逻辑: {hypothesis.logic}
公式思路: {hypothesis.formula_idea}
所需数据: {', '.join(hypothesis.data_requirements)}
预期方向: {hypothesis.expected_direction}

【数据格式】
输入data是一个DataFrame，格式如下：
- index: date (日期)
- {col_info}

【重要：避免未来数据】
⚠️ 生成因子时绝对不能使用未来数据！
- 只能使用当前时刻及之前的历史数据
- 禁止使用 shift(-n) 或任何向前位移的操作（n > 0）
- 只能使用 shift(n) 其中 n <= 0，表示向后看历史数据
- 因子在时间 t 的值只能依赖于 t 及之前的数据
- 计算收益率时使用 .pct_change() 或 shift(1) 组合，不能使用未来数据

【代码要求】
1. 必须是一个完整的Python类
2. 类名使用英文，基于因子名称转换（如：YieldDeviationFactor）
3. calculate方法接收data参数，返回pd.Series
4. 包含异常处理
5. 使用向量化运算，避免循环

【输出格式】
只输出Python代码，不要包含任何解释文字。代码格式如下：

```python
class XXXFactor:
    def __init__(self, params=None):
        ...

    def calculate(self, data):
        ...
        return factor
```
"""

    def _parse_code(self, content: str) -> str:
        """从响应中提取代码"""
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return content.strip()

    def _sanitize_code(self, code: str) -> str:
        """代码安全检查"""
        dangerous = ['import os', 'import sys', 'open(', 'exec(', 'eval(',
                     'subprocess', '__import__', 'file', 'socket']

        for d in dangerous:
            if d in code:
                raise ValueError(f"代码包含危险操作: {d}")

        return code

    def _build_fallback_code(self, hypothesis) -> FactorCode:
        calculation = self._generic_template(hypothesis)

        # 使用 english_name 作为类名，如果没有则转换
        class_name = hypothesis.english_name or self._to_class_name(hypothesis.name)
        # 确保类名以 Factor 结尾
        if not class_name.endswith("Factor"):
            class_name += "Factor"

        code = self.BASE_TEMPLATE.format(
            class_name=class_name,
            factor_name=hypothesis.name,
            description=hypothesis.logic,
            required_columns=", ".join(hypothesis.data_requirements),
            default_params=json.dumps({"window": 20}),
            calculation_code=calculation,
            description_dict=json.dumps(hypothesis.to_dict(), ensure_ascii=False)
        )

        return FactorCode(
            name=hypothesis.name,
            code=code,
            description=hypothesis.to_dict(),
            safe_to_execute=True,
            english_name=hypothesis.english_name
        )

    def modify_code(self, original_code: str, suggestion: str, class_name: str = "Factor") -> str:
        """
        根据用户建议修改因子代码

        Parameters:
            original_code: 原始代码
            suggestion: 修改建议
            class_name: 类名

        Returns:
            修改后的代码
        """
        modify_prompt = f"""请根据用户的修改建议，修改以下因子代码。

【原始代码】
```
{original_code}
```

【修改建议】
{suggestion}

【强制要求】
1. 必须在上述【原始代码】基础上进行修改，不能重新生成
2. 只改动与修改建议相关的部分，其他代码原样保留
3. 保持类名和 calculate 方法签名不变
4. 确保修改后代码语法正确，能正常执行
5. 直接输出修改后的完整代码，不需要任何解释

请输出修改后的完整代码（用```python包裹）："""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=0.3,
                timeout=self.config.timeout,
                messages=[{"role": "user", "content": modify_prompt}]
            )

            modified_code = self._parse_code(response.choices[0].message.content)

            # 安全检查
            safe_code = self._sanitize_code(modified_code)
            return safe_code

        except Exception as e:
            print(f"代码修改失败: {e}")
            # 返回原始代码
            return original_code

    def _to_class_name(self, name: str) -> str:
        """将中文/中英混合名称转为驼峰类名"""
        # 移除非字母数字字符，按空格/下划线分词
        parts = re.split(r'[\s_\-]+', name)
        return ''.join(p.capitalize() for p in parts if p.isascii() and p.isalnum()) or 'Factor'

    def _generic_template(self, hypothesis) -> str:
        """生成通用的fallback计算代码"""
        cols = hypothesis.data_requirements
        primary_col = cols[0] if cols else 'close'
        return f"""        # Fallback: 通用动量/偏离度模板
        window = self.params.get('window', 20)
        col = '{primary_col}' if '{primary_col}' in data.columns else data.columns[0]
        factor = data[col].pct_change(window)
        return factor"""

    def _execute_factor(self, factor_code: FactorCode, data: pd.DataFrame) -> pd.Series:
        """在沙箱中执行因子代码，返回因子值Series"""
        namespace = {'pd': pd, 'np': np}
        exec(factor_code.code, namespace)

        # 找到因子类（以Factor结尾）
        factor_class = None
        for name, obj in namespace.items():
            if isinstance(obj, type) and name.endswith('Factor'):
                factor_class = obj
                break

        if factor_class is None:
            raise RuntimeError("未找到以Factor结尾的类")

        instance = factor_class()
        result = instance.calculate(data)
        return result

    def _execute_with_retry(self, factor_code: FactorCode, data: pd.DataFrame,
                            hypothesis, max_attempts: int = 3) -> FactorCode:
        """
        执行验证并自动修复，最多重试 max_attempts 次。
        """
        sample = data.head(100)
        sample_info = f"DataFrame shape: {data.shape}\ncolumns: {list(data.columns)}\ndtypes:\n{data.dtypes.to_string()}\nsample (first 3 rows):\n{sample.head(3).to_string()}"

        for attempt in range(1, max_attempts + 1):
            try:
                result = self._execute_factor(factor_code, sample)
                if not isinstance(result, pd.Series):
                    raise TypeError(f"calculate() 应返回 pd.Series，实际返回 {type(result)}")
                print(f"  ✓ 第{attempt}次执行成功，因子shape: {result.shape}")
                return factor_code

            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"  ✗ 第{attempt}次执行失败: {e}")

                if attempt == max_attempts:
                    print(f"  ⚠️ 已达最大尝试次数({max_attempts})，返回最后一次生成的代码")
                    return factor_code

                print(f"  🔧 发送给LLM修复中...")
                factor_code = self._debug_with_llm(factor_code, err_msg, sample_info, hypothesis)

    def _debug_with_llm(self, factor_code: FactorCode, error_msg: str,
                        data_info: str, hypothesis) -> FactorCode:
        """让LLM根据报错信息修复因子代码"""
        debug_prompt = f"""以下Python因子代码在实际数据上执行时报错，请修复它。

【错误信息】
{error_msg}

【数据信息】
{data_info}

【当前代码】
```python
{factor_code.code}
```

【因子逻辑】
{hypothesis.logic}

【修复要求】
1. 只修改导致报错的部分，保持原有逻辑
2. 确保 calculate() 方法返回 pd.Series（index与输入data相同）
3. 使用向量化操作，避免循环
4. 处理NaN和除零情况
5. 只使用数据中存在的列: {data_info.split(chr(10))[1]}

只输出修复后的完整Python代码，不要包含任何解释。
```python
...
```"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=0.1,
                timeout=self.config.timeout,
                messages=[{"role": "user", "content": debug_prompt}]
            )
            fixed_code = self._parse_code(response.choices[0].message.content)
            fixed_code = self._sanitize_code(fixed_code)
            return FactorCode(
                name=factor_code.name,
                code=fixed_code,
                description=factor_code.description,
                safe_to_execute=True,
                english_name=factor_code.english_name
            )
        except Exception as e:
            print(f"  LLM修复失败: {e}")
            return factor_code

    def save_factor(self, factor_code: FactorCode, output_dir: str = "./factors"):
        """保存因子代码到文件"""
        os.makedirs(output_dir, exist_ok=True)

        # 使用英文名称作为文件名
        if factor_code.english_name:
            class_name = factor_code.english_name
        else:
            class_name = self._to_class_name(factor_code.name)

        # 确保类名以 Factor 结尾
        if not class_name.endswith("Factor"):
            class_name += "Factor"

        filename = f"{class_name}.py"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(factor_code.code)

        print(f"  因子代码已保存: {filepath}")
        return filepath
