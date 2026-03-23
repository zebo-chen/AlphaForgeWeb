"""
回测API
"""

import sys
import json
import re
from pathlib import Path
from flask import request, jsonify, Blueprint

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('backtest', __name__)
from core.backtest_engine import BacktestEngine


@api_bp.route('/backtest/run', methods=['POST'])
def run_backtest():
    """
    执行因子回测

    请求体:
    {
        "factor_code": "Python代码",
        "class_name": "CurveSlopeFactor",
        "factor_name": "因子名称",
        "params": {...},
        "horizon": 5,  // 预测周期
        "window": 60   // IC滚动窗口
    }

    返回:
    {
        "success": true,
        "ic_mean": 0.018,
        "ic_std": 0.075,
        "ir": 0.24,
        "ic_positive_ratio": 0.52,
        "annual_return": 0.102,
        "max_drawdown": -0.038,
        "pass_test": false,
        "ic_series": [...],
        "return_series": [...]
    }
    """
    try:
        data = request.get_json()
        factor_code = data.get('factor_code', '')
        class_name = data.get('class_name', '')
        factor_name = data.get('factor_name', 'Unknown')
        params = data.get('params', {})
        horizon = data.get('horizon', 5)
        window = data.get('window', 60)
        price_col_param = data.get('price_col', 'auto')  # 'auto' 或具体列名
        time_col_param = data.get('time_col', 'auto')    # 'auto' 或具体列名

        if not factor_code or not class_name:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        import pandas as pd
        import numpy as np

        # 优先使用后端缓存的全量数据（数据准备阶段已加载）
        from api.data import _cached_df as cached_df
        print(f"[回测] _cached_df 状态: {type(cached_df)}, 长度: {len(cached_df) if cached_df is not None else 'None'}", flush=True)
        if cached_df is not None and len(cached_df) > 0:
            df = cached_df.copy()
            print(f"[回测] 使用真实数据，回测行数: {len(df)}, 列: {list(df.columns)}", flush=True)
        else:
            return jsonify({'success': False, 'error': '请先在步骤2「数据准备」加载数据后再进行回测'}), 400

        # 确定时间列并设为索引（BacktestEngine 要求 DatetimeIndex）
        time_col = None
        if time_col_param and time_col_param != 'auto':
            if time_col_param not in df.columns:
                return jsonify({
                    'success': False,
                    'error': f'指定的时间列 `{time_col_param}` 不存在，可用列: {list(df.columns)}'
                }), 400
            time_col = time_col_param
        elif isinstance(df.index, pd.DatetimeIndex) and df.index.name:
            # 已有 DatetimeIndex，自动取其 name
            time_col = df.index.name
        elif 'date' in df.columns:
            time_col = 'date'

        if isinstance(df.index, pd.DatetimeIndex):
            if not df.index.name:
                df.index.name = time_col  # 确保 reset_index 后有列名
            pass  # 已有 DatetimeIndex，保持不变
        else:
            # 从 index name 恢复时间列，或用指定/自动检测的时间列
            col = time_col or (df.index.name if df.index.name else None)
            if col and col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df = df.set_index(col).sort_index()
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.set_index('date').sort_index()
            # 若仍无 DatetimeIndex，engine.run() 会抛错

        print(f"[回测] 索引类型: {type(df.index)}, 前3个: {list(df.index[:3])}", flush=True)

        # 确定价格列：优先使用用户指定的列，否则自动检测
        price_col = None
        if price_col_param and price_col_param != 'auto':
            if price_col_param in df.columns:
                price_col = price_col_param
            else:
                return jsonify({
                    'success': False,
                    'error': f'指定的价格列 `{price_col_param}` 不存在，可用列: {list(df.columns)}'
                }), 400
        if price_col is None:
            for col in ['close', 'price', 'settle', 'settlement', 'close_px']:
                if col in df.columns:
                    price_col = col
                    break
            if price_col is None and len(df.columns) > 0:
                numeric_cols = df.select_dtypes(include='number').columns
                price_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

        # 执行因子代码
        namespace = {'pd': pd, 'np': np}
        try:
            exec(factor_code, namespace)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'代码执行失败: {str(e)}'
            }), 400

        # 获取因子类
        factor_class = namespace.get(class_name)
        if not factor_class:
            return jsonify({
                'success': False,
                'error': f'未找到类: {class_name}'
            }), 400

        # 计算因子值
        try:
            instance = factor_class(params=params) if params else factor_class()
            factor_values = instance.calculate(df)
        except Exception as e:
            import traceback
            err_detail = str(e)
            import re
            missing = re.findall(r"KeyError: '([^']+)'", err_detail)
            col_info = f"，可用列: {list(df.columns)}"
            hint = f"（缺失列 `{missing[0]}`）" if missing else ""
            return jsonify({
                'success': False,
                'error': f'因子计算失败{hint}: {err_detail}{col_info}'
            }), 400

        # 检查因子值是否有效
        if factor_values is None:
            return jsonify({
                'success': False,
                'error': f'因子计算返回了空值，请检查因子代码或数据列是否正确，可用列: {list(df.columns)}'
            }), 400

        # 处理因子值格式（可能是 DataFrame）
        if isinstance(factor_values, pd.DataFrame):
            if 'factor' in factor_values.columns:
                factor_values = factor_values['factor']
            elif len(factor_values.columns) > 0:
                factor_values = factor_values.iloc[:, 0]

        # 找到 return 列（用户数据中的收益率列）
        return_col = None
        for col in ['return', 'ret', 'returns', 'pct_change']:
            if col in df.columns:
                return_col = col
                break

        # 准备回测数据：构建包含 [time_col, price, return, factor] 的 DataFrame
        # df 当前有 DatetimeIndex（index.name = time_col）
        # reset_index 把 time_col 变回列，然后按列拼接
        if isinstance(df.index, pd.DatetimeIndex) and df.index.name:
            df_with_time = df.reset_index()
        else:
            df_with_time = df

        # return 列：优先用数据中的，否则用 price 的 pct_change
        if return_col:
            bkt_df = df_with_time[[time_col, price_col, return_col]].copy()
            bkt_df.columns = [time_col, 'price', 'return']
        else:
            bkt_df = df_with_time[[time_col, price_col]].copy()
            bkt_df.columns = [time_col, 'price']
            bkt_df['return'] = bkt_df['price'].pct_change()

        # 因子值按整数位置拼入（不依赖索引）
        bkt_df['factor'] = factor_values.values

        # 初步过滤：factor 为空的行（BacktestEngine 内部也会再处理）
        bkt_df = bkt_df[bkt_df['factor'].notna()]
        print(f"[回测] 因子值长度:{len(factor_values)}, bkt_df行数:{len(bkt_df)}, return列:{return_col}, price列:{price_col}", flush=True)
        print(f"[回测] bkt_df索引类型: {type(bkt_df.index)}, 前3个: {list(bkt_df.index[:3])}", flush=True)

        if len(bkt_df) < 100:
            return jsonify({
                'success': False,
                'error': '数据点太少，无法回测'
            }), 400

        # 运行回测
        engine = BacktestEngine()
        result = engine.run(
            bkt_data=bkt_df,
            factor_name=factor_name,
            english_name=class_name,
            horizon=horizon,
            window=window,
            time_col=time_col
        )
        print(f"[回测] 回测完成，bkt_df行数:{len(bkt_df)}, ic_series长度:{len(result.ic_series) if result.ic_series is not None else 'None'}, ic_mean:{result.ic_mean:.4f}", flush=True)

        # 准备IC时间序列数据（用于前端绘图）
        ic_series = []
        if result.ic_series is not None and not result.ic_series.empty:
            for idx, val in result.ic_series.items():
                ic_series.append({
                    'date': str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                    'value': float(val) if not pd.isna(val) else None
                })

        # 准备累计收益数据
        return_series = []
        if result.timing_returns is not None and not result.timing_returns.empty:
            cum_ret = 0
            for idx, val in result.timing_returns.items():
                cum_ret += float(val) if not pd.isna(val) else 0
                return_series.append({
                    'date': str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                    'value': cum_ret
                })

        # 因子值序列（用于分布直方图）
        factor_series_data = []
        if result.factor_series is not None and not result.factor_series.empty:
            for idx, val in result.factor_series.items():
                factor_series_data.append({
                    'date': str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                    'value': float(val) if not pd.isna(val) else None
                })
            print(f"[因子序列] 前5条: {factor_series_data[:5]}, 总数: {len(factor_series_data)}", flush=True)

        # 择时收益序列（用于分布直方图）
        timing_returns_data = []
        if result.timing_returns is not None and not result.timing_returns.empty:
            for idx, val in result.timing_returns.items():
                timing_returns_data.append({
                    'date': str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                    'value': float(val) if not pd.isna(val) else None
                })

        # 价格序列（用于因子与价格对比图）
        # 使用 result.price_series，它和 factor_series 有相同的 index（已对齐），顺序完全一致
        price_series_data = []
        if result.price_series is not None and not result.price_series.empty:
            for idx, val in result.price_series.items():
                price_series_data.append({
                    'date': str(idx)[:10] if hasattr(idx, 'strftime') else str(idx),
                    'value': float(val) if not pd.isna(val) else None
                })
        print(f"[价格序列] 前5条: {price_series_data[:5]}, 总数: {len(price_series_data)}", flush=True)

        return jsonify({
            'success': True,
            'ic_mean': float(result.ic_mean),
            'ic_std': float(result.ic_std),
            'ir': float(result.ir),
            'ic_positive_ratio': float(result.ic_positive_ratio),
            'ic_tstat': float(result.ic_tstat),
            'annual_return': float(result.annual_return),
            'capital_gain': float(result.capital_gain),
            'win_rate': float(result.win_rate),
            'profit_loss_ratio': float(result.profit_loss_ratio),
            'sharpe_ratio': float(result.sharpe_ratio),
            'max_drawdown': float(result.max_drawdown),
            'turnover': float(result.turnover),
            'pass_test': bool(result.pass_test),
            'horizon': int(horizon),
            'window': int(window),
            'params': params,
            'factor_name': result.factor_name,
            'ic_series': ic_series,
            'return_series': return_series,
            'factor_series': factor_series_data,
            'timing_returns': timing_returns_data,
            'price_series': price_series_data
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/backtest/config', methods=['POST'])
def save_backtest_config():
    """
    保存回测配置
    """
    try:
        data = request.get_json()
        # 保存配置到session或数据库
        return jsonify({
            'success': True,
            'message': '配置已保存'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
