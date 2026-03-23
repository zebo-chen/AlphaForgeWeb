"""
回测引擎 - 因子检验
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BacktestResult:
    """回测结果"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_positive_ratio: float
    ic_tstat: float
    annual_return: float
    max_drawdown: float
    turnover: float
    pass_test: bool
    capital_gain: float = 0.0  # 资本利得（累计收益）
    win_rate: float = 0.0  # 胜率
    profit_loss_ratio: float = 0.0  # 盈亏比
    sharpe_ratio: float = 0.0  # 夏普比率
    ic_series: pd.Series = None  # IC时间序列
    timing_returns: pd.Series = None  # 择时策略收益序列
    english_name: str = ""  # 英文名称（用于显示）
    factor_series: pd.Series = None  # 因子值序列（用于画图）
    price_series: pd.Series = None  # 价格序列（用于与因子对比）

    def to_dict(self) -> Dict:
        """
        转换为字典格式

        Returns:
            Dict: 包含所有回测指标的字典
        """
        return {
            'factor_name': self.factor_name,
            'english_name': self.english_name,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ir': self.ir,
            'ic_positive_ratio': self.ic_positive_ratio,
            'ic_tstat': self.ic_tstat,
            'capital_gain': self.capital_gain,  # 资本利得（累计收益）
            'annual_return': self.annual_return,  # 年化收益
            'win_rate': self.win_rate,  # 胜率
            'profit_loss_ratio': self.profit_loss_ratio,  # 盈亏比
            'sharpe_ratio': self.sharpe_ratio,  # 夏普比率
            'max_drawdown': self.max_drawdown,  # 最大回撤
            'turnover': self.turnover,  # 换手率
            'pass_test': self.pass_test,
        }

    def summary(self) -> str:
        """生成文字摘要"""
        return f"""
========== 因子回测报告: {self.factor_name} ==========
IC统计:
  - IC均值: {self.ic_mean:.4f}
  - IC标准差: {self.ic_std:.4f}
  - IR比率: {self.ir:.4f}
  - IC胜率: {self.ic_positive_ratio:.2%}
  - T统计量: {self.ic_tstat:.4f}

收益分析:
  - 资本利得: {self.capital_gain:.2%}
  - 年化收益: {self.annual_return:.2%}
  - 胜率: {self.win_rate:.2%}
  - 盈亏比: {self.profit_loss_ratio:.2f}
  - 夏普比率: {self.sharpe_ratio:.4f}
  - 最大回撤: {self.max_drawdown:.2%}
  - 换手率: {self.turnover:.2f}

综合评估: {'✓ 通过' if self.pass_test else '✗ 未通过'}
"""


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.ic_threshold = self.config.get('ic_threshold', 0.02)
        self.ir_threshold = self.config.get('ir_threshold', 0.3)
        self.sharpe_threshold = self.config.get('sharpe_threshold', 1)  # 夏普比率阈值
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', -5)  # 最大回撤阈值（负值）

    def run(self,
            bkt_data: pd.DataFrame,
            factor_name: str = "Unknown",
            english_name: str = "",
            horizon: int = 5,
            window: int = 60,
            time_col: str = None) -> BacktestResult:
        """
        运行单资产择时回测

        Parameters:
            bkt_data: 回测数据 DataFrame，必须包含以下列:
                      - 'factor': 因子值
                      - 'return': 收益率
                      - 'price': 价格（可选，用于画图）
            factor_name: 因子名称
            english_name: 英文名称
            horizon: 预测周期
            window: 滚动IC窗口
            time_col: 时间列名（必填），将作为 DatetimeIndex 用于年化收益计算和 IC 时序分析。

        Returns:
            BacktestResult: 回测结果
        """
        # 检查必要的列
        required_cols = ['factor', 'return']
        missing_cols = [col for col in required_cols if col not in bkt_data.columns]
        if missing_cols:
            raise ValueError(f"bkt_data 缺少必要列: {missing_cols}")

        # time_col 必须存在
        if not time_col:
            raise ValueError("time_col 参数不能为空，请指定时间列名")
        if time_col not in bkt_data.columns:
            raise ValueError(f"时间列 '{time_col}' 不存在，可用列: {list(bkt_data.columns)}")

        bkt_data[time_col] = pd.to_datetime(bkt_data[time_col])

        # 设为索引并转换为 DatetimeIndex
        bkt_data = bkt_data.set_index(time_col)
        bkt_data.index = pd.to_datetime(bkt_data.index, errors='coerce')
        if not isinstance(bkt_data.index, pd.DatetimeIndex) or bkt_data.index.isna().all():
            raise ValueError(f"时间列 '{time_col}' 无法转换为有效日期，请检查数据格式")

        # 提取数据
        factor = bkt_data['factor']
        returns = bkt_data['return']
        price_aligned = bkt_data['price'] if 'price' in bkt_data.columns else None

        # IC分析（时序）
        ic_stats = self._ic_analysis(factor, returns, horizon, window)

        # 择时信号回测（基于因子方向）
        timing_rets = self._timing_backtest(factor, returns)

        # 风险指标
        risk_metrics = self._timing_risk_analysis(timing_rets, returns)

        # 判断是否通过（使用夏普比率和最大回撤）
        pass_test = (
            risk_metrics['sharpe_ratio'] > self.sharpe_threshold and
            risk_metrics['max_drawdown'] > self.max_drawdown_threshold  # max_drawdown 是负值，这里是大于阈值（less negative）
        )

        return BacktestResult(
            factor_name=factor_name,
            ic_mean=ic_stats['ic_mean'],
            ic_std=ic_stats['ic_std'],
            ir=ic_stats['ir'],
            ic_positive_ratio=ic_stats['ic_positive_ratio'],
            ic_tstat=ic_stats['ic_tstat'],
            annual_return=risk_metrics['annual_return'],
            capital_gain=risk_metrics['capital_gain'],
            win_rate=risk_metrics['win_rate'],
            profit_loss_ratio=risk_metrics['profit_loss_ratio'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            max_drawdown=risk_metrics['max_drawdown'],
            turnover=risk_metrics['turnover'],
            pass_test=pass_test,
            ic_series=ic_stats.get('ic_series'),
            timing_returns=timing_rets,
            english_name=english_name,
            factor_series=factor,
            price_series=price_aligned
        )

    def _ic_analysis(self, factor: pd.Series,
                     returns: pd.Series,
                     horizon: int = 5,
                     window: int = 60) -> Dict:
        """
        单资产时序IC分析 - 检验择时因子的预测能力

        Parameters:
            factor: 因子值Series (index=datetime)
            returns: 收益率Series (index=datetime)
            horizon: 预测周期（未来N期收益）
            window: 滚动IC计算窗口
        """
        # 数据对齐
        df = pd.DataFrame({'factor': factor, 'ret': returns}).dropna()

        if len(df) < window + horizon:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'ir': np.nan,
                'ic_positive_ratio': np.nan,
                'ic_tstat': np.nan,
                'ic_series': pd.Series(dtype=float)
            }

        # 构建未来收益（Forward Return）
        # return 列（pct_change）是 price[t]/price[t-1]-1，属于当期收益
        future_ret = round(
            df['ret']
              .rolling(horizon).sum()
              .shift(-horizon + 1),
            6
        )

        # 合并并去除空值
        data = pd.concat([df['factor'], future_ret.rename('future_ret')], axis=1).dropna()

        if len(data) < window:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'ir': np.nan,
                'ic_positive_ratio': np.nan,
                'ic_tstat': np.nan,
                'ic_series': pd.Series(dtype=float)
            }

        # 计算全局IC（Spearman秩相关）
        ic_global = data['factor'].corr(data['future_ret'], method='spearman')

        # 计算滚动IC序列
        ic_series = data['factor'].rolling(window).corr(data['future_ret'])

        # 统计指标
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ic_mean / ic_std if ic_std > 0 else 0,
            'ic_positive_ratio': (ic_series > 0).mean(),
            'ic_tstat': ic_global / (ic_std / np.sqrt(len(data))) if ic_std > 0 else 0,
            'ic_series': ic_series,
            'ic_global': ic_global
        }

    def _timing_backtest(self, factor: pd.Series, returns: pd.Series) -> pd.Series:
        """
        择时信号回测
        基于因子方向生成仓位信号，计算策略收益
        """
        # 生成信号：因子>0做多，<0做空，=0空仓
        signal = np.sign(factor)

        # 信号在 t 时刻生成（基于 factor[t]），用于 t 到 t+1 的持仓
        # return 列是 pct_change = price[t]/price[t-1]-1，需 shift(1) 变为 price[t+1]/price[t]-1
        # 与 IC 保持一致：signal[t] * return[t+1]
        future_ret = returns.shift(-1)

        # 策略收益 = 信号 * 未来收益
        strategy_ret = signal * future_ret

        return strategy_ret.dropna()

    def _timing_risk_analysis(self, timing_rets: pd.Series, returns: pd.Series) -> Dict:
        """择时策略风险分析"""
        if timing_rets.empty:
            return {
                'annual_return': 0,
                'capital_gain': 0,
                'max_drawdown': 0,
                'turnover': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0
            }

        # 资本利得（累计收益）
        cumulative = (timing_rets).cumsum()
        capital_gain = float(cumulative.iloc[-1]) if len(cumulative) > 0 else 0.0

        # 计算实际交易天数
        trading_days = self._calculate_trading_days(timing_rets)

        # 年化收益（标准 252 交易日）
        TRADING_DAYS_PER_YEAR = 252
        if trading_days > 0:
            annual_return = capital_gain / trading_days * TRADING_DAYS_PER_YEAR
        else:
            annual_return = 0.0

        # 最大回撤
        rolling_max = cumulative.cummax()  # 累计收益的历史滚动最大值
        drawdown = cumulative - rolling_max  # 当前值相对历史高点的回撤
        max_dd = float(drawdown.min())    # 最深回撤（负值）

        # 胜率
        win_rate = float(np.sum(timing_rets > 0)/len(timing_rets))

        # 盈亏比
        avg_gain = timing_rets[timing_rets > 0].mean() if (timing_rets > 0).any() else 0
        avg_loss = abs(timing_rets[timing_rets < 0].mean()) if (timing_rets < 0).any() else 1
        pl_ratio = float(avg_gain / avg_loss) if avg_loss > 0 else 0.0

        # 换手率（信号变化频率）
        signal_changes = (np.sign(timing_rets).diff() != 0).sum()
        turnover = float(signal_changes / len(timing_rets)) if len(timing_rets) > 0 else 0.0

        # 夏普比率 = 年化收益 / 年化波动率（无风险收益率假设为 0）
        if len(timing_rets) > 1 and timing_rets.std() > 0:
            annual_volatility = timing_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            'annual_return': annual_return,
            'capital_gain': capital_gain,
            'max_drawdown': max_dd,
            'turnover': turnover,
            'win_rate': win_rate,
            'profit_loss_ratio': pl_ratio,
            'sharpe_ratio': sharpe_ratio
        }

    def _calculate_trading_days(self, series: pd.Series) -> float:
        """
        计算期间经过的交易日数。
        逻辑：日历天数 × (252/365)，估算起止日期间包含的交易日数量。
        适用于任意频率的数据（日内/日频），节假日/周末按比例估算。

        Parameters:
            series: 时间序列数据

        Returns:
            float: 估算的交易日数（最小为1）
        """
        TRADING_DAYS_PER_YEAR = 252
        CALENDAR_DAYS_PER_YEAR = 365

        if series.empty or series.index is None:
            return TRADING_DAYS_PER_YEAR

        # 确保索引是 DatetimeIndex
        idx = series.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception:
                return TRADING_DAYS_PER_YEAR

        calendar_days = (idx.max() - idx.min()).total_seconds() / (24 * 3600)
        trading_days = calendar_days * (TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)

        return max(trading_days, 1.0)

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """可视化结果 - 针对择时因子优化

        布局: 3行2列
        - 左边（列0）: IC时间序列、因子与价格对比、择时策略累计收益
        - 右边（列1）: 因子值分布、收益分布直方图、汇总信息
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # ===== 左边列（列0）=====
        # 1. IC时间序列 (左上)
        if result.ic_series is not None and not result.ic_series.empty:
            axes[0, 0].plot(result.ic_series.index, result.ic_series.values, color='steelblue')
            axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].axhline(y=result.ic_mean, color='g', linestyle='--', alpha=0.5, label=f'Mean: {result.ic_mean:.3f}')
            axes[0, 0].fill_between(result.ic_series.index, result.ic_series.values, 0, alpha=0.3)
            axes[0, 0].set_title('IC Time Series', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Rank IC')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No IC Series Data', ha='center', va='center')
            axes[0, 0].set_title('IC Time Series')

        # 2. 因子与价格对比 (中左)
        if result.factor_series is not None and not result.factor_series.empty:
            ax_factor = axes[1, 0]
            ax_price = ax_factor.twinx()

            # 标准化因子值以便与价格对比
            factor_norm = (result.factor_series - result.factor_series.mean()) / result.factor_series.std()

            ax_factor.plot(result.factor_series.index, factor_norm.values, color='#d62728', alpha=0.8, label='Factor (normalized)')
            ax_factor.axhline(y=0, color='#d62728', linestyle='--', alpha=0.4)
            ax_factor.set_ylabel('Factor (normalized)', color='#d62728', fontsize=11, fontweight='bold')
            ax_factor.tick_params(axis='y', labelcolor='#d62728', labelsize=10)

            if result.price_series is not None and not result.price_series.empty:
                ax_price.plot(result.price_series.index, result.price_series.values, color='#1f77b4', alpha=0.9, label='Price')
                ax_price.set_ylabel('Price', color='#1f77b4', fontsize=11, fontweight='bold')
                ax_price.tick_params(axis='y', labelcolor='#1f77b4', labelsize=10)

            ax_factor.set_title('Factor vs Price', fontsize=12, fontweight='bold')
            ax_factor.grid(True, alpha=0.3)
            ax_factor.legend(loc='upper left', framealpha=0.9)
            if result.price_series is not None:
                ax_price.legend(loc='upper right', framealpha=0.9)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Factor Series Data', ha='center', va='center')
            axes[1, 0].set_title('Factor vs Price')

        # 3. 择时策略累计收益 (下左)
        if result.timing_returns is not None and not result.timing_returns.empty:
            cumulative = (result.timing_returns).cumsum()
            axes[2, 0].plot(cumulative.index, cumulative.values, color='darkgreen')
            axes[2, 0].set_title('Timing Strategy Cumulative Return', fontsize=12, fontweight='bold')
            axes[2, 0].set_ylabel('Cumulative Return')
            axes[2, 0].grid(True, alpha=0.3)
            # 添加起始和结束标注
            # axes[2, 0].axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        else:
            axes[2, 0].text(0.5, 0.5, 'No Timing Returns Data', ha='center', va='center')
            axes[2, 0].set_title('Timing Strategy Cumulative Return')

        # ===== 右边列（列1）=====
        # 4. 因子值分布 (右上)
        if result.factor_series is not None and not result.factor_series.empty:
            axes[0, 1].hist(result.factor_series.dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
            axes[0, 1].axvline(x=result.factor_series.mean(), color='g', linestyle='--', label=f'Mean: {result.factor_series.mean():.4f}')
            axes[0, 1].set_title('Factor Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Factor Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        else:
            axes[0, 1].text(0.5, 0.5, 'No Factor Data', ha='center', va='center')
            axes[0, 1].set_title('Factor Distribution')

        # 5. 收益分布直方图 (右中)
        if result.timing_returns is not None and not result.timing_returns.empty:
            axes[1, 1].hist(result.timing_returns.dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].axvline(x=result.timing_returns.mean(), color='g', linestyle='--', label=f'Mean: {result.timing_returns.mean():.4f}')
            axes[1, 1].set_title('Return Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Return')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No Returns Data', ha='center', va='center')
            axes[1, 1].set_title('Return Distribution')

        # 6. 汇总信息 (右下)
        axes[2, 1].axis('off')
        # 使用英文名显示，如果没有则使用中文名
        display_name = result.english_name or result.factor_name
        summary_text = f"""Factor: {display_name}

IC Mean: {result.ic_mean:.4f}
IC Std: {result.ic_std:.4f}
IR: {result.ir:.4f}
IC Win Rate: {result.ic_positive_ratio:.2%}

Capital Gain: {round(result.capital_gain,2)}%
Annual Return: {round(result.annual_return,2)}%
Win Rate: {result.win_rate:.2%}
Profit/Loss: {result.profit_loss_ratio:.2f}
Sharpe Ratio: {result.sharpe_ratio:.4f}
Max Drawdown: {result.max_drawdown:.2%}
Turnover: {result.turnover:.2f}

Status: {'PASS' if result.pass_test else 'FAIL'}
"""
        axes[2, 1].text(0.1, 0.5, summary_text, fontsize=11,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
