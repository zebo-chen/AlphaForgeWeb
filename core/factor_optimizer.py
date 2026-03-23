"""
因子优化模块 - 参数优化与敏感性分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict
    best_score: float
    all_results: pd.DataFrame
    score_name: str  # 优化的指标名称 (如 'ir', 'ic_mean')


@dataclass
class SensitivityResult:
    """敏感性分析结果"""
    param_name: str
    param_values: List
    scores: List
    correlation: float  # 参数值与分数的相关性


# ==================== LLM 优化相关数据结构 ====================

@dataclass
class LLMAnalysisReport:
    """LLM 分析报告"""
    issues: List[str]  # 识别的问题
    suggestions: List[str]  # 改进建议
    overall_score: float  # 综合评分
    diagnosis: str  # 诊断文本


@dataclass
class ParamSuggestion:
    """参数建议"""
    param_name: str  # 参数名
    suggested_values: List  # 建议值列表
    rationale: str  # 理由


@dataclass
class OptimizationDecision:
    """优化决策"""
    should_optimize: bool  # 是否需要优化
    should_regenerate: bool  # 是否需要重新生成假设
    reason: str  # 决策理由
    suggested_next_steps: List[str]  # 下一步建议
    iterations_used: int  # 使用的迭代次数
    final_result: 'BacktestResult' = None  # 最终优化结果


# ==================== 因子代码解析函数 ====================

def extract_params_from_factor_code(factor_code: str) -> List[str]:
    """
    从因子代码中提取实际使用的参数名

    通过解析 __init__ 方法中的 params.get() 调用来提取参数名

    Parameters:
        factor_code: 因子类代码字符串

    Returns:
        List[str]: 参数名列表
    """
    import re

    if not factor_code:
        return []

    # 匹配 params.get('param_name', ...) 模式
    # 注意：代码中是 params.get()，不是 self.params.get()
    pattern = r"params\.get\s*\(\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, factor_code)

    # 去重并保持顺序
    unique_params = []
    seen = set()
    for param in matches:
        if param not in seen:
            unique_params.append(param)
            seen.add(param)

    return unique_params


def get_params_from_factor_class(factor_class) -> List[str]:
    """
    从因子类中提取实际使用的参数名

    优先使用 inspect 模块获取因子类的源代码，然后解析参数。
    如果失败（内置类等），则尝试通过实例化或签名分析获取参数。

    Parameters:
        factor_class: 因子类（不是实例）

    Returns:
        List[str]: 参数名列表
    """
    import inspect

    if factor_class is None:
        return []

    # 方法1: 尝试获取源代码
    try:
        source = inspect.getsource(factor_class)
        return extract_params_from_factor_code(source)
    except Exception:
        pass

    # 方法2: 尝试通过 inspect 获取 __init__ 的参数
    try:
        init_method = getattr(factor_class, '__init__', None)
        if init_method:
            # 获取签名
            sig = inspect.signature(init_method)
            params = []
            for name, param in sig.parameters.items():
                if name != 'self' and name != 'params':
                    params.append(name)
            if params:
                return params
    except Exception:
        pass

    # 方法3: 尝试实例化并获取 params 字典的键
    try:
        # 尝试用空参数实例化
        try:
            instance = factor_class(params={})
        except TypeError:
            instance = factor_class()

        # 获取默认 params
        if hasattr(instance, 'params') and instance.params:
            return list(instance.params.keys())
    except Exception:
        pass

    return []


# ==================== LLM 提示词模板 ====================

ANALYZE_RESULT_PROMPT = """你是一位资深的量化因子研究员。请分析以下因子回测结果：

【因子信息】
- 因子名称: {factor_name}
- 预期方向: {expected_direction}
- 假设逻辑: {logic}

【回测结果】
- IC均值: {ic_mean:.4f} (阈值: >0.02)
- IC标准差: {ic_std:.4f}
- IR比率: {ir:.4f} (阈值: >0.3)
- IC胜率: {ic_positive_ratio:.2%} (阈值: >55%)
- 年化收益: {annual_return}
- 夏普比率: {sharpe_ratio:.4f} (阈值: >0.5，越高说明风险调整收益越好)
- 胜率: {win_rate}
- 盈亏比: {profit_loss_ratio:.2f}
- 最大回撤: {max_drawdown} (注：数值为负，绝对值越大风险越高，如 -0.18 表示回撤18%)
- 换手率: {turnover:.2f}

⚠️ 特别注意：如果最大回撤绝对值超过5%，说明策略风险较高，需要重点关注回撤原因！

请从以下维度诊断问题：
1. IC相关问题 (过低、不稳定、方向错误)
2. 收益相关问题 (负收益、回撤过大、胜率低、盈亏比差、夏普比率过低)
3. 风险问题 (换手过高、回撤过大)

输出JSON格式：
{{"issues": ["问题1", "问题2"], "suggestions": ["建议1", "建议2"], "overall_score": 7.5, "diagnosis": "诊断总结"}}"""

SUGGEST_PARAMS_PROMPT = """基于以下因子代码、假设信息和问题，建议参数搜索范围。

【因子代码】
{factor_code}

【假设信息】
- 名称: {name}
- 逻辑: {logic}
- 数据需求: {data_requirements}
- 当前使用的默认参数: {default_params}

【因子代码中实际使用的参数名】（只允许推荐这些参数！）
{available_params}

【识别的问题】
{issues}

请根据上述"因子代码中实际使用的参数"，结合识别的问题，建议需要调优的参数及其候选值范围。

注意：
- ★ 必须只使用上述列表中的参数名，禁止推荐列表中不存在的参数！
- 参数名必须与 self.params.get('参数名') 完全一致
- 建议的候选值要合理，符合参数的物理含义

输出JSON格式：
{{"params": [{{"name": "window", "values": [10, 20, 30], "rationale": "增大窗口提高稳定性"}}]}}"""


class FactorOptimizer:
    """
    因子参数优化器

    支持:
    - 网格搜索 (Grid Search)
    - 敏感性分析 (Sensitivity Analysis)
    - LLM 驱动的智能优化 (LLM-guided Optimization)
    """

    SYSTEM_PROMPT = """你是一位资深的量化因子研究员，擅长分析因子回测结果并给出优化建议。"""

    def __init__(self, backtest_engine=None, max_iterations: int = 3):
        """
        初始化优化器

        Args:
            backtest_engine: BacktestEngine 实例，如果为None则创建
            max_iterations: 最大迭代次数（LLM优化模式）
        """
        from .backtest_engine import BacktestEngine
        self.backtest_engine = backtest_engine or BacktestEngine()
        self.max_iterations = max_iterations

        # 初始化 LLM 配置
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 配置"""
        try:
            from .llm_config import get_llm_config, create_llm_client
            self.config = get_llm_config()
            self.client = create_llm_client(self.config)
            self._has_llm = bool(self.config.api_key)
        except Exception as e:
            print(f"⚠️ LLM 初始化失败: {e}")
            self._has_llm = False
            self.client = None

    @property
    def model(self) -> str:
        """获取当前使用的模型名称"""
        return getattr(self.config, 'model', 'unknown') if hasattr(self, 'config') else 'unknown'

    def grid_search(self,
                   factor_class,
                   data: pd.DataFrame,
                   returns: pd.Series,
                   param_grid: Dict[str, List],
                   score_metric: str = 'sharpe',
                   **backtest_kwargs) -> OptimizationResult:
        """
        网格搜索最优参数

        Parameters:
            factor_class: 因子类（必须是可实例化的，有 calculate 方法）
            data: 输入数据 DataFrame
            returns: 收益率序列
            param_grid: 参数网格，如 {'window': [10, 20, 30], 'threshold': [0.5, 1.0]}
            score_metric: 优化指标 ('sharpe', 'ir', 'ic_mean', 'annual_return', 'max_drawdown', 'ic_positive_ratio')
            **backtest_kwargs: 传递给 backtest_engine.run 的其他参数

        Returns:
            OptimizationResult: 包含最优参数和所有结果
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        print(f"🔍 开始网格搜索，共 {len(all_combinations)} 种参数组合")
        print(f"   参数: {param_grid}")
        print(f"   优化指标: {score_metric}")

        results = []

        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            try:
                # 实例化因子并计算
                factor_instance = factor_class(params=params) if params else factor_class()
                factor_values = factor_instance.calculate(data)

                bkt_df = data.copy()
                bkt_df['factor'] = np.array(factor_values)
                bkt_df['return'] = np.array(returns)

                # 如果 data 已有 DatetimeIndex，从中提取 time_col（避免 run() 找不存在的列名）
                run_kwargs = dict(backtest_kwargs)
                if isinstance(bkt_df.index, pd.DatetimeIndex) and bkt_df.index.name:
                    run_kwargs['time_col'] = bkt_df.index.name

                # 回测
                result = self.backtest_engine.run(
                    bkt_data=bkt_df,
                    **run_kwargs
                )

                # 提取目标指标
                score = self._extract_score(result, score_metric)

                results.append({
                    'params': params,
                    score_metric: score,
                    'ic_mean': result.ic_mean,
                    'ic_std': result.ic_std,
                    'ir': result.ir,
                    'capital_gain': getattr(result, 'capital_gain', 0),
                    'annual_return': getattr(result, 'annual_return', 0),
                    'sharpe_ratio': getattr(result, 'sharpe_ratio', 0),
                    'ic_positive_ratio': result.ic_positive_ratio,
                    'max_drawdown': result.max_drawdown,
                    'turnover': result.turnover,
                    'pass_test': result.pass_test
                })

                status = "✓" if result.pass_test else "✗"
                print(f"  [{i+1}/{len(all_combinations)}] {params} => {score_metric}={score:.4f} {status}")

            except Exception as e:
                results.append({
                    'params': params,
                    score_metric: np.nan,
                    'ic_mean': np.nan,
                    'ic_std': np.nan,
                    'ir': np.nan,
                    'capital_gain': np.nan,
                    'annual_return': np.nan,
                    'ic_positive_ratio': np.nan,
                    'max_drawdown': np.nan,
                    'turnover': np.nan,
                    'pass_test': False,
                    'error': str(e)
                })
                print(f"  [{i+1}/{len(all_combinations)}] {params} => 错误: {e}")

        # 转换为 DataFrame
        results_df = pd.DataFrame(results)

        # 找到最优结果
        valid_results = results_df[results_df[score_metric].notna()]
        if len(valid_results) == 0:
            print("⚠️ 没有找到有效的参数组合")
            return OptimizationResult(
                best_params={},
                best_score=np.nan,
                all_results=results_df,
                score_name=score_metric
            )

        best_idx = valid_results[score_metric].idxmax()
        best_params = results_df.loc[best_idx, 'params']
        best_score = results_df.loc[best_idx, score_metric]

        print(f"\n🏆 最优参数: {best_params}")
        print(f"   最优 {score_metric}: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results_df,
            score_name=score_metric
        )

    def sensitivity_analysis(self,
                           factor_class,
                           data: pd.DataFrame,
                           returns: pd.Series,
                           param_name: str,
                           param_values: List,
                           score_metric: str = 'sharpe',
                           **backtest_kwargs) -> SensitivityResult:
        """
        单参数敏感性分析

        分析单个参数变化对因子的影响

        Parameters:
            factor_class: 因子类
            data: 输入数据
            returns: 收益率序列
            param_name: 要分析的参数名
            param_values: 参数值列表
            score_metric: 优化指标
            **backtest_kwargs: 其他回测参数

        Returns:
            SensitivityResult: 敏感性分析结果
        """
        print(f"📊 参数敏感性分析: {param_name}")
        print(f"   参数值: {param_values}")

        scores = []

        for i, value in enumerate(param_values):
            params = {param_name: value}

            try:
                factor_instance = factor_class(params=params)
                factor_values = factor_instance.calculate(data)

                # 准备回测数据（传入 time_col，确保 _prepare_bkt_data 添加时间列）
                bkt_df = self._prepare_bkt_data(factor_values, returns, time_col=backtest_kwargs.get('time_col'))

                run_kwargs = dict(backtest_kwargs)
                if isinstance(factor_values, (pd.Series, pd.DataFrame)):
                    idx = factor_values.index if isinstance(factor_values, pd.Series) else factor_values.index
                    if isinstance(idx, pd.DatetimeIndex) and idx.name:
                        run_kwargs['time_col'] = idx.name

                result = self.backtest_engine.run(
                    bkt_data=bkt_df,
                    **run_kwargs
                )

                score = self._extract_score(result, score_metric)
                scores.append(score)

                print(f"  [{i+1}/{len(param_values)}] {param_name}={value} => {score_metric}={score:.4f}")

            except Exception as e:
                scores.append(np.nan)
                print(f"  [{i+1}/{len(param_values)}] {param_name}={value} => 错误: {e}")

        # 计算相关性
        valid_pairs = [(v, s) for v, s in zip(param_values, scores) if not np.isnan(s)]
        if len(valid_pairs) > 1:
            values_arr = np.array([v for v, s in valid_pairs])
            scores_arr = np.array([s for v, s in valid_pairs])
            correlation = np.corrcoef(values_arr, scores_arr)[0, 1]
        else:
            correlation = np.nan

        print(f"   相关性: {correlation:.4f}")

        return SensitivityResult(
            param_name=param_name,
            param_values=param_values,
            scores=scores,
            correlation=correlation
        )

    def multi_sensitivity(self,
                        factor_class,
                        data: pd.DataFrame,
                        returns: pd.Series,
                        param_ranges: Dict[str, List],
                        score_metric: str = 'sharpe',
                        **backtest_kwargs) -> pd.DataFrame:
        """
        多参数敏感性分析（参数两两组合热力图）

        Parameters:
            factor_class: 因子类
            data: 输入数据
            returns: 收益率序列
            param_ranges: 参数范围字典，如 {'window': [10,20,30], 'threshold': [0.5,1.0]}
            score_metric: 优化指标
            **backtest_kwargs: 其他回测参数

        Returns:
            DataFrame: 包含所有结果
        """
        param_names = list(param_ranges.keys())

        if len(param_names) != 2:
            raise ValueError("目前只支持两个参数的热力图分析")

        print(f"🔬 多参数敏感性分析: {param_names}")

        results = []

        for val1 in param_ranges[param_names[0]]:
            for val2 in param_ranges[param_names[1]]:
                params = {param_names[0]: val1, param_names[1]: val2}

                try:
                    factor_instance = factor_class(params=params)
                    factor_values = factor_instance.calculate(data)

                    # 准备回测数据（传入 time_col，确保 _prepare_bkt_data 添加时间列）
                    bkt_df = self._prepare_bkt_data(factor_values, returns, time_col=backtest_kwargs.get('time_col'))

                    run_kwargs = dict(backtest_kwargs)
                    if isinstance(factor_values, (pd.Series, pd.DataFrame)):
                        idx = factor_values.index if isinstance(factor_values, pd.Series) else factor_values.index
                        if isinstance(idx, pd.DatetimeIndex) and idx.name:
                            run_kwargs['time_col'] = idx.name

                    result = self.backtest_engine.run(
                        bkt_data=bkt_df,
                        **run_kwargs
                    )

                    score = self._extract_score(result, score_metric)

                    results.append({
                        param_names[0]: val1,
                        param_names[1]: val2,
                        score_metric: score,
                        'ir': result.ir,
                        'ic_mean': result.ic_mean
                    })

                except Exception as e:
                    results.append({
                        param_names[0]: val1,
                        param_names[1]: val2,
                        score_metric: np.nan,
                        'ir': np.nan,
                        'ic_mean': np.nan
                    })

        return pd.DataFrame(results)

    def plot_heatmap(self,
                    sensitivity_df: pd.DataFrame,
                    param1: str,
                    param2: str,
                    score_metric: str = 'sharpe',
                    save_path: str = None):
        """
        绘制双参数热力图

        Parameters:
            sensitivity_df: multi_sensitivity 返回的 DataFrame
            param1: 第一个参数名
            param2: 第二个参数名
            score_metric: 显示的指标
            save_path: 保存路径
        """
        pivot = sensitivity_df.pivot(index=param1, columns=param2, values=score_metric)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=pivot.mean().mean())
        plt.title(f'{param1} vs {param2} - {score_metric}')
        plt.xlabel(param2)
        plt.ylabel(param1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_sensitivity(self,
                       sensitivity_result: SensitivityResult,
                       save_path: str = None):
        """
        绘制单参数敏感性曲线

        Parameters:
            sensitivity_result: sensitivity_analysis 返回的结果
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        plt.plot(sensitivity_result.param_values, sensitivity_result.scores,
                marker='o', linewidth=2, markersize=8)

        plt.xlabel(sensitivity_result.param_name, fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Sensitivity Analysis: {sensitivity_result.param_name}\n'
                  f'Correlation: {sensitivity_result.correlation:.4f}')
        plt.grid(True, alpha=0.3)

        # 标记最优点
        valid = [(v, s) for v, s in zip(sensitivity_result.param_values, sensitivity_result.scores)
                if not np.isnan(s)]
        if valid:
            best_v, best_s = max(valid, key=lambda x: x[1])
            plt.axvline(x=best_v, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_v}')
            plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def _extract_score(self, backtest_result, metric: str) -> float:
        """从回测结果中提取目标指标"""
        # 注意：max_drawdown 是负值，为了优化方便，取反转为正值
        metrics_map = {
            'ir': backtest_result.ir,
            'ic_mean': backtest_result.ic_mean,
            'capital_gain': getattr(backtest_result, 'capital_gain', 0),
            'annual_return': getattr(backtest_result, 'annual_return', 0),
            'ic_positive_ratio': backtest_result.ic_positive_ratio,
            'sharpe': getattr(backtest_result, 'sharpe_ratio', 0),
            'sharpe_ratio': getattr(backtest_result, 'sharpe_ratio', 0),
            'max_drawdown': -getattr(backtest_result, 'max_drawdown', 0),  # 取反，转为正值优化
        }

        return metrics_map.get(metric, backtest_result.ir)

    def _prepare_bkt_data(self, factor_values, returns, time_col: str = None) -> pd.DataFrame:
        """
        准备回测数据：将因子值和收益率组合成 DataFrame
        如果 factor_values 有 DatetimeIndex，尝试从中提取时间列
        time_col 参数优先于 index.name（因为用户因子代码可能丢失索引名）
        """
        if isinstance(factor_values, pd.Series):
            idx = factor_values.index
            aligned_returns = returns.reindex(idx)
            cols = {'factor': factor_values, 'return': aligned_returns}
        elif isinstance(factor_values, pd.DataFrame):
            factor_col = factor_values.iloc[:, 0] if len(factor_values.columns) > 0 else factor_values.iloc[:, 0]
            idx = factor_col.index
            aligned_returns = returns.reindex(idx)
            cols = {'factor': factor_col, 'return': aligned_returns}
        else:
            bkt_df = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()
            return bkt_df

        # 确定时间列名：优先用传入的 time_col，其次用 index.name
        _time_col = time_col or (idx.name if isinstance(idx, pd.DatetimeIndex) and idx.name else None)
        if _time_col and isinstance(idx, pd.DatetimeIndex):
            cols[_time_col] = idx.to_numpy()

        bkt_df = pd.DataFrame(cols).dropna()
        return bkt_df

    # ==================== LLM 驱动的方法 ====================

    def analyze_result(self, result, hypothesis=None) -> LLMAnalysisReport:
        """
        让 LLM 分析回测结果，识别问题并给出改进建议

        Parameters:
            result: BacktestResult 回测结果
            hypothesis: Hypothesis 假设对象（可选，提供更多上下文）

        Returns:
            LLMAnalysisReport: LLM 分析报告
        """
        if not self._has_llm:
            return self._fallback_analysis(result)

        # 准备提示词参数
        prompt_params = {
            'factor_name': result.factor_name,
            'expected_direction': getattr(hypothesis, 'expected_direction', 'unknown') if hypothesis else 'unknown',
            'logic': getattr(hypothesis, 'logic', 'N/A') if hypothesis else 'N/A',
            'ic_mean': result.ic_mean,
            'ic_std': result.ic_std,
            'ir': result.ir,
            'ic_positive_ratio': result.ic_positive_ratio,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'profit_loss_ratio': result.profit_loss_ratio,
            'max_drawdown': result.max_drawdown,
            'turnover': result.turnover,
        }

        prompt = ANALYZE_RESULT_PROMPT.format(**prompt_params)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                timeout=self.config.timeout,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content
            return self._parse_llm_analysis(content)

        except Exception as e:
            print(f"⚠️ LLM 分析失败: {e}")
            return self._fallback_analysis(result)

    def _parse_llm_analysis(self, content: str) -> LLMAnalysisReport:
        """解析 LLM 返回的分析结果"""
        import re
        import json

        try:
            # 尝试提取 JSON
            m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
            else:
                data = json.loads(content)

            return LLMAnalysisReport(
                issues=data.get('issues', []),
                suggestions=data.get('suggestions', []),
                overall_score=float(data.get('overall_score', 5.0)),
                diagnosis=data.get('diagnosis', '')
            )
        except Exception as e:
            print(f"⚠️ 解析 LLM 响应失败: {e}")
            return LLMAnalysisReport(
                issues=['解析失败'],
                suggestions=['请手动分析'],
                overall_score=5.0,
                diagnosis='LLM 响应解析失败'
            )

    def _fallback_analysis(self, result) -> LLMAnalysisReport:
        """当 LLM 不可用时的降级分析"""
        issues = []
        suggestions = []

        # 基础规则判断
        if abs(result.ic_mean) < 0.02:
            issues.append('IC均值过低')
            suggestions.append('考虑调整因子计算方式或增加预测周期')

        if result.ir < 0.3:
            issues.append('IR比率较低')
            suggestions.append('因子稳定性不足，考虑增加平滑或调整窗口参数')

        if result.ic_positive_ratio < 0.55:
            issues.append('IC胜率不足')
            suggestions.append('因子方向可能不稳定')

        if result.max_drawdown < -0.2:
            issues.append('最大回撤过大（超过20%）')
            suggestions.append('风险控制不足，建议添加止损或降低仓位')

        if result.pass_test:
            issues = []
            suggestions = ['因子已通过检验，无需优化']

        return LLMAnalysisReport(
            issues=issues,
            suggestions=suggestions,
            overall_score=7.0 if result.pass_test else 5.0,
            diagnosis='基于规则的分析（LLM不可用）'
        )

    def suggest_parameters(self,
                         hypothesis=None,
                         analysis: LLMAnalysisReport = None,
                         current_params: Dict = None,
                         factor_code: str = None,
                         factor_class=None) -> List[ParamSuggestion]:
        """
        LLM 建议参数搜索范围

        Parameters:
            hypothesis: Hypothesis 假设对象
            analysis: LLMAnalysisReport 分析报告
            current_params: Dict 当前参数
            factor_code: str 因子代码字符串（可选）
            factor_class: 因子类（可选，会自动获取源代码提取参数）

        Returns:
            List[ParamSuggestion]: 参数建议列表
        """
        if not self._has_llm or not analysis:
            return self._fallback_suggest_params(analysis, factor_code, factor_class)

        # 如果没有问题需要解决，返回空
        if not analysis.issues:
            return []

        # 从因子代码或因子类中提取实际使用的参数名
        available_params = []
        if factor_code:
            available_params = extract_params_from_factor_code(factor_code)
        elif factor_class:
            available_params = get_params_from_factor_class(factor_class)

        available_params_str = ', '.join(available_params) if available_params else '（无法从代码中提取参数，请查看代码中的 self.params.get() 调用）'

        print(f"   📋 当前参数: {current_params}, 可用调优参数: {available_params}")

        # 准备提示词参数
        prompt_params = {
            'factor_code': factor_code or '（未提供因子代码）',
            'name': getattr(hypothesis, 'name', 'Unknown') if hypothesis else 'Unknown',
            'logic': getattr(hypothesis, 'logic', 'N/A') if hypothesis else 'N/A',
            'data_requirements': ', '.join(getattr(hypothesis, 'data_requirements', [])) if hypothesis else 'N/A',
            'default_params': str(current_params) if current_params else '使用默认参数',
            'available_params': available_params_str,
            'issues': '\n'.join(f"- {issue}" for issue in analysis.issues),
        }

        prompt = SUGGEST_PARAMS_PROMPT.format(**prompt_params)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                timeout=self.config.timeout,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content
            return self._parse_param_suggestions(content)

        except Exception as e:
            print(f"⚠️ LLM 参数建议失败: {e}")
            return self._fallback_suggest_params(analysis, factor_code, factor_class)

    def _parse_param_suggestions(self, content: str) -> List[ParamSuggestion]:
        """解析 LLM 返回的参数建议"""
        import re
        import json

        try:
            m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
            else:
                data = json.loads(content)

            suggestions = []
            for p in data.get('params', []):
                suggestions.append(ParamSuggestion(
                    param_name=p.get('name', ''),
                    suggested_values=p.get('values', []),
                    rationale=p.get('rationale', '')
                ))

            return suggestions

        except Exception as e:
            print(f"⚠️ 解析参数建议失败: {e}")
            return []

    def _fallback_suggest_params(self, analysis: LLMAnalysisReport = None, factor_code: str = None, factor_class=None) -> List[ParamSuggestion]:
        """当 LLM 不可用时的降级参数建议"""
        suggestions = []

        if not analysis:
            return suggestions

        # 从因子代码或因子类中提取实际参数
        available_params = []
        if factor_code:
            available_params = extract_params_from_factor_code(factor_code)
        elif factor_class:
            available_params = get_params_from_factor_class(factor_class)

        # 通用参数建议映射（根据问题类型推荐参数）
        param_suggestions_map = {
            'window': {
                'issue_keywords': ['IC', '低'],
                'values': [20, 30, 40, 60],
                'rationale': '增大窗口提高IC稳定性'
            },
            'n': {
                'issue_keywords': ['IC', '低'],
                'values': [10, 20, 30, 40],
                'rationale': '调整滚动窗口大小'
            },
            'smoothing': {
                'issue_keywords': ['回撤', '风险', '波动'],
                'values': [1, 2, 3, 5],
                'rationale': '增加平滑减少波动'
            },
            'threshold': {
                'issue_keywords': ['换手', '交易'],
                'values': [0.3, 0.5, 0.7, 1.0],
                'rationale': '提高阈值减少交易频率'
            },
            'standardize': {
                'issue_keywords': ['标准化'],
                'values': [True, False],
                'rationale': '控制是否进行截面标准化'
            },
        }

        # 基于问题的简单规则建议
        for issue in analysis.issues:
            for param_name, config in param_suggestions_map.items():
                # 检查问题是否包含关键词
                if any(keyword in issue for keyword in config['issue_keywords']):
                    # 如果提供了因子代码，只推荐实际存在的参数
                    if available_params and param_name not in available_params:
                        continue
                    # 避免重复添加相同参数
                    if not any(s.param_name == param_name for s in suggestions):
                        suggestions.append(ParamSuggestion(
                            param_name=param_name,
                            suggested_values=config['values'],
                            rationale=config['rationale']
                        ))

        # 如果有可用参数但没有匹配到任何建议，至少推荐第一个可用参数
        if not suggestions and available_params:
            # 推荐第一个常用参数作为默认值
            first_param = available_params[0] if available_params else 'n'
            if first_param == 'n':
                suggestions.append(ParamSuggestion(
                    param_name=first_param,
                    suggested_values=[10, 20, 30],
                    rationale='调整参数优化因子性能'
                ))

        return suggestions

    def run_llm_guided_search(self,
                              factor_class,
                              data: pd.DataFrame,
                              returns: pd.Series,
                              param_suggestions: List[ParamSuggestion],
                              score_metric: str = 'sharpe',
                              **backtest_kwargs) -> OptimizationResult:
        """
        执行 LLM 引导的参数搜索

        Parameters:
            factor_class: 因子类
            data: 输入数据
            returns: 收益率序列
            param_suggestions: List[ParamSuggestion] LLM 建议的参数
            score_metric: 优化指标

        Returns:
            OptimizationResult: 优化结果
        """
        if not param_suggestions:
            print("⚠️ 没有参数建议，跳过搜索")
            return None

        # 构建参数网格
        param_grid = {}
        for suggestion in param_suggestions:
            param_grid[suggestion.param_name] = suggestion.suggested_values

        print(f"🔍 LLM 引导的参数搜索")
        for suggestion in param_suggestions:
            print(f"   {suggestion.param_name}: {suggestion.suggested_values} ({suggestion.rationale})")

        # 调用现有的网格搜索
        return self.grid_search(
            factor_class=factor_class,
            data=data,
            returns=returns,
            param_grid=param_grid,
            score_metric=score_metric,
            **backtest_kwargs
        )

    def run_sensitivity_llm(self,
                           factor_class,
                           data: pd.DataFrame,
                           returns: pd.Series,
                           param_name: str,
                           param_values: List,
                           score_metric: str = 'sharpe',
                           **backtest_kwargs) -> Tuple[SensitivityResult, str]:
        """
        运行敏感性分析并让 LLM 解读结果

        Parameters:
            factor_class: 因子类
            data: 输入数据
            returns: 收益率序列
            param_name: 参数名
            param_values: 参数值列表
            score_metric: 优化指标

        Returns:
            (SensitivityResult, LLM解读): 敏感性结果和LLM解读
        """
        # 先运行敏感性分析
        sensitivity_result = self.sensitivity_analysis(
            factor_class, data, returns,
            param_name, param_values, score_metric, **backtest_kwargs
        )

        # 构建 LLM 解读提示
        prompt = f"""请分析以下参数敏感性测试结果：

参数名称: {param_name}
参数值: {param_values}
得分: {sensitivity_result.scores}
相关性: {sensitivity_result.correlation:.4f}

请判断：
1. 该参数对因子性能的影响方向（正相关/负相关/无明显关系）
2. 最佳的参数值是多少
3. 参数敏感度如何（高/中/低）

输出JSON格式：
{{"interpretation": "解读", "best_value": 值, "sensitivity": "高/中/低"}}"""

        llm_interpretation = ""
        if self._has_llm:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1000,
                    temperature=0.1,
                    timeout=self.config.timeout,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                llm_interpretation = response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ LLM 敏感性解读失败: {e}")

        return sensitivity_result, llm_interpretation

    def generate_adjustment_advice(self,
                                   original_result,
                                   optimized_result,
                                   analysis: LLMAnalysisReport) -> str:
        """
        生成调整建议

        Parameters:
            original_result: 原始回测结果
            optimized_result: 优化后的回测结果
            analysis: LLM 分析报告

        Returns:
            str: 调整建议文本
        """
        if not self._has_llm:
            return self._fallback_adjustment_advice(original_result, optimized_result)

        # 计算改善
        ic_improvement = optimized_result.ic_mean - original_result.ic_mean
        ir_improvement = optimized_result.ir - original_result.ir
        sharpe_improvement = optimized_result.sharpe_ratio - original_result.sharpe_ratio
        dd_improvement = optimized_result.max_drawdown - original_result.max_drawdown  # 负值，越大越好

        prompt = f"""基于以下优化前后的对比，生成调整建议：

【优化前】
- IC均值: {original_result.ic_mean:.4f}
- IR: {original_result.ir:.4f}
- 夏普比率: {original_result.sharpe_ratio:.4f}
- 最大回撤: {original_result.max_drawdown:.2%}（负值，绝对值越大风险越高）
- 状态: {'通过' if original_result.pass_test else '未通过'}

【优化后】
- IC均值: {optimized_result.ic_mean:.4f}
- IR: {optimized_result.ir:.4f}
- 夏普比率: {optimized_result.sharpe_ratio:.4f}
- 最大回撤: {optimized_result.max_drawdown:.2%}
- 状态: {'通过' if optimized_result.pass_test else '未通过'}

【改善】
- IC提升: {ic_improvement:+.4f}
- IR提升: {ir_improvement:+.4f}
- 夏普比率: {sharpe_improvement:+.4f}
- 最大回撤: {dd_improvement:+.2%}（正值表示回撤减小）

【识别的问题】
{chr(10).join(f"- {s}" for s in analysis.issues)}

请给出简洁的调整建议（1-3句话）。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                timeout=self.config.timeout,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ LLM 建议生成失败: {e}")
            return self._fallback_adjustment_advice(original_result, optimized_result)

    def _fallback_adjustment_advice(self, original_result, optimized_result) -> str:
        """降级的调整建议"""
        if optimized_result.pass_test and not original_result.pass_test:
            return "因子已通过检验，优化成功！"
        sharpe_ok = optimized_result.sharpe_ratio > original_result.sharpe_ratio
        dd_ok = optimized_result.max_drawdown > original_result.max_drawdown  # 负值越大越好
        ir_ok = optimized_result.ir > original_result.ir
        if sharpe_ok and dd_ok:
            return f"夏普比率和最大回撤均改善，建议继续调优。"
        elif ir_ok:
            return f"IR有所改善（{original_result.ir:.3f} -> {optimized_result.ir:.3f}），建议继续调优。"
        else:
            return "优化未能显著改善，建议重新生成假设。"

    def should_regenerate_hypothesis(self,
                                    original_result,
                                    optimized_result,
                                    iteration: int) -> bool:
        """
        判断是否需要重新生成假设

        Parameters:
            original_result: 原始回测结果
            optimized_result: 优化后的回测结果
            iteration: 当前迭代次数

        Returns:
            bool: 是否需要重新生成假设
        """
        # 条件1：达到最大迭代次数
        if iteration >= self.max_iterations:
            return True

        # 条件2：优化后仍未通过检验，且 IC、IR、夏普、最大回撤均无改善
        if not optimized_result.pass_test:
            ir_improvement = optimized_result.ir - original_result.ir
            sharpe_improvement = optimized_result.sharpe_ratio - original_result.sharpe_ratio
            dd_improvement = optimized_result.max_drawdown - original_result.max_drawdown  # 负值越大越好
            ic_improvement = abs(optimized_result.ic_mean) - abs(original_result.ic_mean)  # 取绝对值比较方向
            if ir_improvement < 0.01 and sharpe_improvement < 0.05 and dd_improvement < 0.01 and ic_improvement < 0.002:
                return True

        return False

    def optimize(self,
                hypothesis,
                factor_class,
                data: pd.DataFrame,
                returns: pd.Series,
                original_result,
                initial_params: Dict = None,
                factor_code: str = None,
                **backtest_kwargs) -> OptimizationDecision:
        """
        主优化流程（带循环的智能优化）

        Parameters:
            hypothesis: Hypothesis 假设对象
            factor_class: 因子类
            data: 输入数据
            returns: 收益率序列
            original_result: 原始回测结果
            initial_params: 初始参数（可选）
            factor_code: 因子代码字符串（可选，用于提取参数）
            **backtest_kwargs: 其他回测参数

        Returns:
            OptimizationDecision: 优化决策
        """
        print("=" * 60)
        print("🤖 开始 LLM 驱动的因子优化")
        print("=" * 60)

        # 打印原始结果
        print("\n📊 原始回测结果:")
        print(f"   因子名称: {original_result.factor_name}")
        print(f"   IC均值: {original_result.ic_mean:.4f} (阈值: >0.02)")
        print(f"   IC标准差: {original_result.ic_std:.4f}")
        print(f"   IR: {original_result.ir:.4f} (阈值: >0.3)")
        print(f"   IC胜率: {original_result.ic_positive_ratio:.2%} (阈值: >55%)")
        print(f"   资本利得: {round(original_result.capital_gain,2)}%")
        print(f"   年化收益: {round(original_result.annual_return,2)}%")
        print(f"   胜率: {original_result.win_rate:.2%}")
        print(f"   盈亏比: {original_result.profit_loss_ratio:.2f}")
        print(f"   夏普比率: {original_result.sharpe_ratio:.4f} (阈值: >0.5)")
        print(f"   最大回撤: {round(original_result.max_drawdown,2)}% (阈值: >-20%)")
        print(f"   换手率: {original_result.turnover:.2f}")
        print(f"   状态: {'✓ 通过' if original_result.pass_test else '✗ 未通过'}")

        current_result = original_result
        current_params = initial_params or {}

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"📊 迭代 {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            # 1. LLM 分析当前结果
            print("\n[1/4] 🤖 LLM 分析回测结果...")
            analysis = self.analyze_result(current_result, hypothesis)

            print(f"\n   📋 LLM 诊断结果:")
            print(f"   综合评分: {analysis.overall_score:.1f}/10")
            if analysis.issues:
                print(f"   识别的问题:")
                for issue in analysis.issues:
                    print(f"      • {issue}")
            else:
                print(f"   识别的问题: 无")
            if analysis.suggestions:
                print(f"   改进建议:")
                for suggestion in analysis.suggestions:
                    print(f"      • {suggestion}")
            if analysis.diagnosis:
                print(f"   诊断详情: {analysis.diagnosis}")

            # 检查是否通过检验
            if current_result.pass_test:
                print("\n✅ 因子已通过检验，优化完成！")
                return OptimizationDecision(
                    should_optimize=False,
                    should_regenerate=False,
                    reason="因子已通过检验",
                    suggested_next_steps=["使用当前参数"],
                    iterations_used=iteration - 1,
                    final_result=current_result
                )

            # 2. LLM 建议参数
            print("\n[2/4] 🤖 LLM 建议参数搜索范围...")
            param_suggestions = self.suggest_parameters(hypothesis, analysis, current_params, factor_code=factor_code, factor_class=factor_class)

            if not param_suggestions:
                print("   ⚠️ 没有可用的参数建议")
                if iteration >= self.max_iterations:
                    break
                continue

            print(f"\n   📋 参数建议:")
            for suggestion in param_suggestions:
                print(f"      • {suggestion.param_name}: {suggestion.suggested_values}")
                print(f"        理由: {suggestion.rationale}")

            # 3. LLM 引导的参数搜索
            print("\n[3/4] 执行参数搜索...")
            opt_result = self.run_llm_guided_search(
                factor_class=factor_class,
                data=data,
                returns=returns,
                param_suggestions=param_suggestions,
                **backtest_kwargs
            )

            if opt_result is None or not opt_result.best_params:
                print("   ⚠️ 参数搜索未返回有效结果")
                continue

            # 记录最优参数，供 API 层查询
            self._last_best_params = opt_result.best_params
            self._last_best_score = opt_result.best_score

            # 使用最优参数重新计算因子
            try:
                best_factor = factor_class(params=opt_result.best_params)
                factor_values = best_factor.calculate(data)

                # 准备回测数据（传入 time_col，确保 _prepare_bkt_data 添加时间列）
                bkt_df = data.copy()
                bkt_df['factor'] = np.array(factor_values)
                bkt_df['return'] = np.array(returns)
                
                # 保留 backtest_kwargs 中已有的 time_col，不要被 factor_values 的索引名覆盖
                current_result = self.backtest_engine.run(
                    bkt_data=bkt_df,
                    **backtest_kwargs
                )
                current_params = opt_result.best_params

                # 打印参数搜索结果
                print(f"\n   📋 参数搜索结果:")
                print(f"   {'参数组合':<30} | {'IC':>7} | {'IR':>6} | {'夏普':>6} | {'最大回撤':>9} | {'状态':<6}")
                print(f"   {'-'*30}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}")

                for idx, row in opt_result.all_results.iterrows():
                    params_str = str(row['params'])[:28]
                    status = "✓" if row['pass_test'] else "✗"
                    dd_str = f"{row['max_drawdown']:.2%}" if not pd.isna(row['max_drawdown']) else '—'
                    print(f"   {params_str:<30} | {row['ic_mean']:>7.4f} | {row['ir']:>6.4f} | {row['sharpe_ratio']:>6.3f} | {dd_str:>9} | {status:<6}")

                # 打印最优参数
                print(f"\n   🏆 最优参数: {opt_result.best_params}")
                print(f"      最优 {opt_result.score_name}: {opt_result.best_score:.4f}")

                # 对比优化前后
                ic_change = current_result.ic_mean - original_result.ic_mean
                ir_change = current_result.ir - original_result.ir
                sharpe_change = current_result.sharpe_ratio - original_result.sharpe_ratio
                dd_change = current_result.max_drawdown - original_result.max_drawdown  # max_drawdown 负值，越大越好

                ic_arrow = "↑" if ic_change > 0 else ("↓" if ic_change < 0 else "→")
                ir_arrow = "↑" if ir_change > 0 else ("↓" if ir_change < 0 else "→")
                sharpe_arrow = "↑" if sharpe_change > 0 else ("↓" if sharpe_change < 0 else "→")
                dd_arrow = "↑" if dd_change > 0 else ("↓" if dd_change < 0 else "→")  # 回撤改善（负值变大）

                print(f"\n   📈 优化效果对比:")
                print(f"      IC: {original_result.ic_mean:.4f} → {current_result.ic_mean:.4f} ({ic_arrow}{abs(ic_change):.4f})")
                print(f"      IR: {original_result.ir:.4f} → {current_result.ir:.4f} ({ir_arrow}{abs(ir_change):.4f})")
                print(f"      夏普比率: {original_result.sharpe_ratio:.4f} → {current_result.sharpe_ratio:.4f} ({sharpe_arrow}{abs(sharpe_change):.4f})")
                print(f"      最大回撤: {original_result.max_drawdown}% → {current_result.max_drawdown}% ({dd_arrow}{abs(dd_change)}%)")
                print(f"      状态: {'通过' if current_result.pass_test else '未通过'}")

            except Exception as e:
                print(f"   ⚠️ 最优参数验证失败: {e}")
                continue

            # 4. 生成调整建议
            print("\n[4/4] 生成调整建议...")
            advice = self.generate_adjustment_advice(original_result, current_result, analysis)
            print(f"   {advice}")

            # 判断是否需要继续
            if self.should_regenerate_hypothesis(original_result, current_result, iteration):
                print(f"\n🔄 达到终止条件（迭代{iteration}次后仍末通过检验）")
                break

            print(f"\n➡️ 继续下一轮优化...")

        # 循环结束，输出最终决策
        final_decision = self.should_regenerate_hypothesis(
            original_result, current_result, self.max_iterations
        )

        # 打印最终结果汇总
        print(f"\n{'='*60}")
        print("📊 优化结果汇总")
        print(f"{'='*60}")
        print(f"   迭代次数: {iteration}/{self.max_iterations}")
        print(f"   最终 IC: {current_result.ic_mean:.4f} (原始: {original_result.ic_mean:.4f})")
        print(f"   最终 IR: {current_result.ir:.4f} (原始: {original_result.ir:.4f})")
        print(f"   最终夏普比率: {current_result.sharpe_ratio:.4f} (原始: {original_result.sharpe_ratio:.4f})")
        print(f"   最终最大回撤: {current_result.max_drawdown}% (原始: {original_result.max_drawdown}%)")
        print(f"   最终状态: {'✓ 通过' if current_result.pass_test else '✗ 未通过'}")
        print(f"\n   决策: {'需要重新生成假设' if final_decision else '优化完成，可使用当前参数'}")
        print(f"   原因: {'已达到最大迭代次数' if final_decision else '优化已完成'}")

        return OptimizationDecision(
            should_optimize=True,
            should_regenerate=final_decision,
            reason="已达到最大迭代次数" if final_decision else "优化已完成",
            suggested_next_steps=analysis.suggestions if analysis.issues else [],
            iterations_used=self.max_iterations,
            final_result=current_result
        )


def quick_optimize(factor_class,
                  data: pd.DataFrame,
                  returns: pd.Series,
                  param_grid: Dict[str, List],
                  **kwargs):
    """
    快速优化函数（简化调用）

    Example:
        result = quick_optimize(
            MyFactor,
            data,
            returns,
            {'window': [10, 20, 30], 'threshold': [0.5, 1.0]}
        )
        print(result.best_params)
    """
    optimizer = FactorOptimizer()
    return optimizer.grid_search(factor_class, data, returns, param_grid, **kwargs)
