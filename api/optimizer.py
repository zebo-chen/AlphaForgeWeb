"""
因子优化API - LLM驱动的智能参数优化
"""

import sys
import json
import time
import threading
import queue
from pathlib import Path
from io import StringIO
from flask import request, jsonify, Blueprint, Response
from dataclasses import dataclass

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('optimizer', __name__)


# 轻量级 Hypothesis 对象（供 FactorOptimizer 使用）
@dataclass
class _HypothesisObj:
    name: str
    logic: str
    expected_direction: str = "正向"
    data_requirements: list = None

    def __post_init__(self):
        if self.data_requirements is None:
            self.data_requirements = []


# ==================== 优化会话状态（用于 SSE 流式日志）====================
_optimize_state = {
    'active': False,
    'queue': queue.Queue(),
    'done': False,
    'error': None,
}


def _build_bkt_df(df, time_col, price_col, return_col, factor_values):
    """
    构建包含 [time_col, price, return, factor] 的回测DataFrame，
    供 engine.run(time_col=...) 使用。
    """
    import pandas as pd

    # df 当前已有 DatetimeIndex（由调用方确保）
    if isinstance(df.index, pd.DatetimeIndex) and df.index.name:
        df_with_time = df.reset_index()
    else:
        df_with_time = df

    if return_col:
        bkt_df = df_with_time.copy()
    else:
        bkt_df = df_with_time.copy()
        bkt_df['return'] = bkt_df['price'].pct_change()

    bkt_df['factor'] = factor_values.values
    bkt_df = bkt_df[bkt_df['factor'].notna()]
    return bkt_df


# ==================== 流式优化状态 ====================
_stream_state = {
    'active': False,
    'queue': queue.Queue(),
    'done': False,
    'error': None,
}
_optimize_lock = threading.Lock()   # 防止并发优化
_running_thread = None             # 当前正在运行的优化线程


class _StdoutCapture:
    """线程安全的 stdout 捕获器，捕获的内容实时写入队列"""
    def __init__(self):
        self._original = sys.stdout
        self._buf = ''
        self._active = False
        self._queue = queue.Queue()

    def write(self, text):
        self._original.write(text)
        if self._active:
            self._buf += text
            while '\n' in self._buf:
                line, self._buf = self._buf.split('\n', 1)
                line = line.strip()
                if line:
                    self._queue.put(line)

    def flush(self):
        """强制刷新剩余缓冲区（处理无换行结尾的情况）"""
        self._original.flush()
        if self._active and self._buf.strip():
            self._queue.put(self._buf.strip())
            self._buf = ''

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


_capture = _StdoutCapture()


# ==================== 流式优化主端点 ====================
@api_bp.route('/optimize/run', methods=['POST'])
def run_optimize():
    """
    执行 LLM 驱动的因子参数优化（SSE 流式日志）

    返回: text/event-stream
      - type=log:  {"type":"log","msg":"..."}
      - type=donesuccess: {"type":"done","success":true,"result":{...}}
      - type=error: {"type":"error","msg":"错误信息"}
    """
    global _running_thread

    # 如果已有优化在运行，等待其完成（防止多次并发）
    if _optimize_lock.locked():
        return jsonify({'success': False, 'error': '优化正在进行中，请等待完成后重试'}), 429

    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求体为空'}), 400

    # 等待上一个线程完全结束（用户可能多次点击）
    if _running_thread is not None:
        _running_thread.join(timeout=5)
        _running_thread = None

    result_holder = [None]
    error_holder = [None]
    task_done = [False]

    # 初始化捕获器
    _capture._active = True
    _capture._buf = ''
    # 清空旧队列
    while not _capture._queue.empty():
        try: _capture._queue.get_nowait()
        except queue.Empty: break

    old_stdout = sys.stdout
    sys.stdout = _capture

    def run_task():
        """后台线程：运行优化，stdout 写入 _capture._queue"""
        try:
            result_holder[0] = _do_optimize_impl(data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_holder[0] = str(e)
        finally:
            task_done[0] = True
            # 刷新缓冲区，确保没有换行结尾的日志也被写入队列
            _capture.flush()
            # 放结束哨兵
            _capture._queue.put({'__done': True})

    thread = threading.Thread(target=run_task)
    thread.start()

    def stream_generate():
        """主线程（请求线程）消费队列，每收到一条日志 yield 一次 → werkzeug 实时 flush"""
        try:
            while not task_done[0]:
                try:
                    item = _capture._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if isinstance(item, dict) and item.get('__done'):
                    break
                yield f"data: {json.dumps({'type': 'log', 'msg': item})}\n\n"

            # 消费队列中剩余日志
            while not _capture._queue.empty():
                try:
                    item = _capture._queue.get_nowait()
                    if isinstance(item, dict) and item.get('__done'):
                        break
                    yield f"data: {json.dumps({'type': 'log', 'msg': item})}\n\n"
                except queue.Empty:
                    break

            # 发送最终结果
            error = error_holder[0]
            result = result_holder[0]
            if error:
                yield f"data: {json.dumps({'type': 'error', 'msg': error})}\n\n"
            elif result:
                yield f"data: {json.dumps({'type': 'done', 'success': result.get('success', False), 'result': result})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'msg': '优化未返回结果'})}\n\n"
        finally:
            _capture._active = False
            sys.stdout = old_stdout

    return Response(
        stream_generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@api_bp.route('/optimize/logs/<task_id>', methods=['GET'])
def get_optimize_logs(task_id):
    """轮询获取优化日志（实时推送到前端）"""
    current_task = _stream_state.get('task_id')
    if task_id != current_task:
        return jsonify({'logs': [], 'done': False, 'task_id': task_id})

    # 取出本轮所有新日志
    new_logs = []
    while not _stream_state['queue'].empty():
        try:
            msg = _stream_state['queue'].get_nowait()
            new_logs.append(msg)
            _stream_state['logs'].append(msg)
        except queue.Empty:
            break

    return jsonify({
        'logs': new_logs,
        'all_logs': _stream_state['logs'],
        'done': _stream_state['done'],
        'error': _stream_state['error'],
        'task_id': task_id
    })


@api_bp.route('/optimize/result/<task_id>', methods=['GET'])
def get_optimize_result(task_id):
    """获取优化最终结果"""
    current_task = _stream_state.get('task_id')
    if task_id != current_task:
        return jsonify({'success': False, 'error': '任务不存在'}), 404

    if not _stream_state['done']:
        return jsonify({'success': False, 'ready': False})

    if _stream_state['error']:
        return jsonify({'success': False, 'error': _stream_state['error']})

    return jsonify({'success': True, 'data': _stream_state['result']})


def _do_optimize_impl(data):
    """优化器的核心逻辑，返回最终 JSON 字典"""
    import pandas as pd
    import numpy as np

    factor_code = data.get('factor_code', '')
    class_name = data.get('class_name', '')
    factor_name = data.get('factor_name', 'Unknown')
    param_grid = data.get('param_grid', {})
    score_metric = data.get('score_metric', 'ir')
    max_iterations = int(data.get('max_iterations', 2))
    horizon = data.get('horizon', 5)
    window = data.get('window', 60)
    time_col_param = data.get('time_col', 'auto')

    if not factor_code or not class_name:
        return {'success': False, 'error': '缺少必要参数'}

    from api.data import _cached_df as cached_df
    df = None

    # 优先用服务器端缓存
    if cached_df is not None and len(cached_df) > 0:
        df = cached_df.copy()
    # 其次尝试从请求体中构建 DataFrame
    elif 'data' in data and 'columns' in data:
        try:
            import pandas as pd
            df = pd.DataFrame(data['data'], columns=data['columns'])
        except Exception:
            pass

    if df is None or len(df) == 0:
        return {'success': False, 'error': '请先在步骤2「数据准备」加载数据后再进行优化'}

    # 确定时间列
    time_col = None
    if time_col_param and time_col_param != 'auto':
        if time_col_param not in df.columns:
            return {'success': False, 'error': f'时间列 `{time_col_param}` 不存在'}
        time_col = time_col_param
    elif isinstance(df.index, pd.DatetimeIndex) and df.index.name:
        time_col = df.index.name
    elif 'date' in df.columns:
        time_col = 'date'

    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.name:
            df.index.name = time_col
    else:
        col = time_col or (df.index.name if df.index.name else None)
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.set_index(col).sort_index()
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.set_index('date').sort_index()

    time_col = df.index.name or time_col

    # 确定价格列
    price_col = None
    for col in ['close', 'price', 'settle', 'settlement', 'close_px']:
        if col in df.columns:
            price_col = col
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include='number').columns
        price_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

    # 确定 return 列
    return_col = None
    for col in ['return', 'ret', 'returns', 'pct_change']:
        if col in df.columns:
            return_col = col
            break

    # 如果传入了步骤3的原始回测结果，构造 BacktestResult 对象
    original_backtest_data = data.get('original_backtest_result')

    # 执行因子代码
    namespace = {'pd': pd, 'np': np}
    exec(factor_code, namespace)
    factor_class = namespace.get(class_name)
    if not factor_class:
        return {'success': False, 'error': f'未找到类: {class_name}'}

    # 自动识别参数
    from core.factor_optimizer import extract_params_from_factor_code
    detected_params = extract_params_from_factor_code(factor_code)
    if not param_grid and detected_params:
        print(f"[优化] 自动识别到参数: {detected_params}", flush=True)
        for pname in detected_params:
            param_grid[pname] = _generate_param_values(pname)

    detected_params_info = [{'name': p, 'values': param_grid.get(p, [])} for p in detected_params]

    from core.backtest_engine import BacktestEngine, BacktestResult
    engine = BacktestEngine()

    # 优先复用步骤3传入的原始回测结果，避免重复计算
    if original_backtest_data:
        # 从 dict 构造 BacktestResult 对象，供 optimizer.optimize() 使用
        original_result = BacktestResult(
            factor_name=factor_name,
            ic_mean=float(original_backtest_data.get('ic_mean', 0)),
            ic_std=float(original_backtest_data.get('ic_std', 0)),
            ir=float(original_backtest_data.get('ir', 0)),
            ic_positive_ratio=float(original_backtest_data.get('ic_positive_ratio', 0)),
            ic_tstat=float(original_backtest_data.get('ic_tstat', 0)),
            annual_return=float(original_backtest_data.get('annual_return', 0)),
            max_drawdown=float(original_backtest_data.get('max_drawdown', 0)),
            turnover=float(original_backtest_data.get('turnover', 0)),
            pass_test=bool(original_backtest_data.get('pass_test', False)),
            capital_gain=float(original_backtest_data.get('capital_gain', 0)),
            win_rate=float(original_backtest_data.get('win_rate', 0)),
            profit_loss_ratio=float(original_backtest_data.get('profit_loss_ratio', 0)),
            sharpe_ratio=float(original_backtest_data.get('sharpe_ratio', 0)),
        )
        print(f"[优化] 复用步骤4回测结果: IC={original_result.ic_mean:.4f}, IR={original_result.ir:.4f}, pass={original_result.pass_test}", flush=True)
        # 用步骤3的参数计算一次因子值（供 grid_search 后续使用）
        orig_params = original_backtest_data.get('params') or original_backtest_data.get('current_params') or {}
        try:
            instance = factor_class(params=orig_params) if orig_params else factor_class()
        except Exception:
            instance = factor_class()
        factor_values = instance.calculate(df)
        if isinstance(factor_values, pd.DataFrame):
            if 'factor' in factor_values.columns:
                factor_values = factor_values['factor']
            elif len(factor_values.columns) > 0:
                factor_values = factor_values.iloc[:, 0]
        # 构建初始回测DataFrame（供后续 returns_series 使用）
        bkt_df_initial = _build_bkt_df(df, time_col, price_col, return_col, factor_values)
    else:
        # 无传入结果时，自己跑初始回测
        print("[优化] 未传入原始回测结果，将重新计算初始回测...", flush=True)
        try:
            instance = factor_class(params={})
            factor_values = instance.calculate(df)
        except Exception as e:
            return {'success': False, 'error': f'因子计算失败: {str(e)}'}

        if isinstance(factor_values, pd.DataFrame):
            if 'factor' in factor_values.columns:
                factor_values = factor_values['factor']
            elif len(factor_values.columns) > 0:
                factor_values = factor_values.iloc[:, 0]

        bkt_df_initial = _build_bkt_df(df, time_col, price_col, return_col, factor_values)
        if len(bkt_df_initial) < 100:
            return {'success': False, 'error': '数据点太少，无法优化'}

        original_result = engine.run(
            bkt_data=bkt_df_initial,
            factor_name=factor_name,
            english_name=class_name,
            horizon=horizon,
            window=window,
            time_col=time_col
        )
        print(f"[优化] 初始回测完成: IC={original_result.ic_mean:.4f}, IR={original_result.ir:.4f}, pass={original_result.pass_test}", flush=True)

    # 调用优化器
    hypothesis = _HypothesisObj(
        name=factor_name,
        logic=f"因子 {factor_name} 的参数优化",
        expected_direction="正向",
        data_requirements=[price_col, return_col or 'pct_change(price)']
    )
    # 用过滤后的 bkt_df_initial（去掉因子为空行）提取 returns，
    # 保证 returns_series 和 optimizer 内部 factor_class.calculate(data) 的返回值长度一致
    # bkt_df_initial 已是 DatetimeIndex（time_col 为索引名），用 .sort_index() 确保顺序
    bkt_df_initial = bkt_df_initial.sort_index()
    returns_series = bkt_df_initial['return']  # Series，index = DatetimeIndex (time_col)
    # data 用过滤后的 bkt_df_initial，不做额外 set_index，避免 _build_bkt_df 里的 reset_index
    # 与 time_col 列名冲突（reset_index 后原索引名变成列名，但 time_col 可能是不同的列名）
    df_for_optimize = bkt_df_initial.copy()

    from core.factor_optimizer import FactorOptimizer
    optimizer = FactorOptimizer(backtest_engine=engine, max_iterations=max_iterations)
    print("[优化] 开始 LLM 驱动优化 ...", flush=True)
    decision = optimizer.optimize(
        hypothesis=hypothesis,
        factor_class=factor_class,
        factor_code=factor_code,
        data=df_for_optimize,
        returns=returns_series,
        original_result=original_result,
        initial_params={},
        horizon=horizon,
        window=window,
        time_col=time_col
    )
    print(f"[优化] 优化完成: iterations={decision.iterations_used}, pass={decision.final_result.pass_test if decision.final_result else 'N/A'}", flush=True)

    final_result = decision.final_result if decision.final_result else original_result

    print(f"[优化] 完成！最优参数: {optimizer._last_best_params}", flush=True)

    return {
        'success': True,
        'best_params': getattr(optimizer, '_last_best_params', None) or {},
        'best_score': float(getattr(optimizer, '_last_best_score', float(final_result.ir))),
        'score_metric': score_metric,
        'detected_params': detected_params_info,
        'iterations_used': decision.iterations_used,
        'decision': {
            'should_optimize': decision.should_optimize,
            'should_regenerate': decision.should_regenerate,
            'reason': decision.reason,
            'suggested_next_steps': decision.suggested_next_steps,
        },
        'final_metrics': {
            'ic_mean': float(final_result.ic_mean),
            'ic_std': float(final_result.ic_std),
            'ir': float(final_result.ir),
            'ic_positive_ratio': float(final_result.ic_positive_ratio),
            'annual_return': float(final_result.annual_return),
            'capital_gain': float(final_result.capital_gain),
            'max_drawdown': float(final_result.max_drawdown),
            'sharpe_ratio': float(final_result.sharpe_ratio),
            'pass_test': bool(final_result.pass_test),
        }
    }


def _generate_param_values(param_name: str) -> list:
    """
    根据参数名自动生成合理的候选值列表

    规则：
    - window/n/period/day 类：整数滚动窗口 → [10, 20, 30, 40, 60]
    - threshold/stop 类：小数阈值 → [0.1, 0.3, 0.5, 0.7, 1.0]
    - ratio/mult 类：小数比例 → [0.5, 1.0, 1.5, 2.0]
    - smoothing/alpha 类：(0,1) 区间 → [0.1, 0.3, 0.5, 0.7, 0.9]
    - standardize/use_xxx 类：布尔 → [True, False]
    - top 类：整数排名 → [5, 10, 20, 30, 50]
    - 默认：整数 → [5, 10, 20, 30]
    """
    name = param_name.lower()

    if any(k in name for k in ['window', 'n=', 'n ', 'period', 'day', '_n', 'lookback']):
        return [10, 20, 30, 40, 60]
    if any(k in name for k in ['threshold', 'stop', 'limit']):
        return [0.1, 0.3, 0.5, 0.7, 1.0]
    if any(k in name for k in ['ratio', 'mult', 'weight']):
        return [0.5, 1.0, 1.5, 2.0]
    if any(k in name for k in ['smoothing', 'alpha', 'decay']):
        return [0.1, 0.3, 0.5, 0.7, 0.9]
    if any(k in name for k in ['standardize', 'use_', 'with_', 'enable', 'normalize']):
        return [True, False]
    if any(k in name for k in ['top', 'rank', 'num']):
        return [5, 10, 20, 30, 50]
    if name.startswith('N') or name.startswith('n'):
        return [10, 20, 30, 40, 60]
    # 默认：整数
    return [5, 10, 20, 30]


@api_bp.route('/optimize/diagnose', methods=['POST'])
def run_diagnose():
    """
    基于真实回测结果，运行 AI 诊断并返回分析报告

    请求体:
    {
        "factor_name": "xxx",
        "ic_mean": 0.018,
        "ic_std": 0.075,
        "ir": 0.24,
        "ic_positive_ratio": 0.52,
        "ic_tstat": 1.2,
        "annual_return": 0.102,
        "capital_gain": 0.18,
        "win_rate": 0.55,
        "profit_loss_ratio": 1.2,
        "sharpe_ratio": 0.45,
        "max_drawdown": -0.18,
        "turnover": 12.5,
        "pass_test": false
    }

    返回:
    {
        "success": true,
        "issues": ["IC均值偏低", "IR不稳定"],
        "suggestions": ["建议增大窗口参数", "增加平滑处理"],
        "overall_score": 5.5,
        "diagnosis": "..."
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少回测指标数据'}), 400

        # 构建轻量级结果对象，供 FactorOptimizer.analyze_result 使用
        class _BacktestResultObj:
            def __init__(self, d):
                self.factor_name = d.get('factor_name', 'Unknown')
                self.ic_mean = float(d.get('ic_mean', 0))
                self.ic_std = float(d.get('ic_std', 0))
                self.ir = float(d.get('ir', 0))
                self.ic_positive_ratio = float(d.get('ic_positive_ratio', 0))
                self.ic_tstat = float(d.get('ic_tstat', 0))
                self.annual_return = float(d.get('annual_return', 0))
                self.capital_gain = float(d.get('capital_gain', 0))
                self.win_rate = float(d.get('win_rate', 0))
                self.profit_loss_ratio = float(d.get('profit_loss_ratio', 0))
                self.sharpe_ratio = float(d.get('sharpe_ratio', 0))
                self.max_drawdown = float(d.get('max_drawdown', 0))
                self.turnover = float(d.get('turnover', 0))
                self.pass_test = bool(d.get('pass_test', False))

        # 构建 hypothesis
        hypothesis = _HypothesisObj(
            name=data.get('factor_name', 'Unknown'),
            logic=f"因子 {data.get('factor_name', 'Unknown')} 的参数优化",
            expected_direction="正向",
            data_requirements=[]
        )

        bkt_result = _BacktestResultObj(data)

        from core.factor_optimizer import FactorOptimizer
        optimizer = FactorOptimizer(max_iterations=1)
        report = optimizer.analyze_result(bkt_result, hypothesis)

        print(f"[诊断] 因子={data.get('factor_name')}, "
              f"IC={bkt_result.ic_mean:.4f}, IR={bkt_result.ir:.4f}, "
              f"评分={report.overall_score:.1f}, "
              f"问题数={len(report.issues)}", flush=True)

        return jsonify({
            'success': True,
            'issues': report.issues,
            'suggestions': report.suggestions,
            'overall_score': report.overall_score,
            'diagnosis': report.diagnosis,
            'pass_test': bkt_result.pass_test,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/optimize/analyze', methods=['POST'])
def analyze_result():
    """
    分析回测结果，返回优化建议
    """
    try:
        data = request.get_json()

        ic_mean = data.get('ic_mean', 0)
        ir = data.get('ir', 0)

        issues = []
        suggestions = []

        if abs(ic_mean) < 0.02:
            issues.append(f'IC均值偏低 ({ic_mean:.3f} < 0.02)')
            suggestions.append('建议增大窗口参数提高稳定性')

        if ir < 0.3:
            issues.append(f'IR不稳定 ({ir:.3f} < 0.3)')
            suggestions.append('考虑增加平滑处理')

        overall_score = min(10, max(1, (abs(ic_mean) / 0.02 + ir / 0.3) * 5))

        return jsonify({
            'success': True,
            'issues': issues,
            'suggestions': suggestions,
            'overall_score': overall_score
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
