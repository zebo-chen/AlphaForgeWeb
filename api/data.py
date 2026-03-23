"""
数据处理API
"""

import sys
import json
import math
from pathlib import Path
from flask import request, jsonify, Blueprint

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('data', __name__)

# 全局存储最后一次加载的 DataFrame（用于图表）
_cached_df = None
# 后台加载标记
_cached_loading = False


@api_bp.route('/data/sample', methods=['GET', 'POST'])
def get_sample_data():
    """
    获取样本数据

    返回:
    {
        "success": true,
        "symbol": "T.CHT",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "count": 1217,
        "preview": [...]
    }
    """
    try:
        import pandas as pd
        import numpy as np
        import math

        # 生成国债期货样本数据
        dates = pd.date_range('2020-01-01', periods=1217, freq='B')  # 工作日
        np.random.seed(42)

        # 生成价格序列：小幅上涨趋势（年化约5%），便于展示因子择时效果
        drift = 0.0002  # 每天约0.02%的漂移
        returns = drift + np.random.randn(1217) * 0.005
        prices = 100 * np.exp(np.cumsum(returns))

        data = {
            'date': [str(d.date()) for d in dates],
            'open': [(p * (1 + np.random.randn() * 0.002)).__round__(2) for p in prices],
            'high': [(p * (1 + np.abs(np.random.randn()) * 0.005)).__round__(2) for p in prices],
            'low': [(p * (1 - np.abs(np.random.randn()) * 0.005)).__round__(2) for p in prices],
            'close': [round(p, 2) for p in prices],
            'volume': [int(np.random.randint(10000, 100000)) for _ in dates],
            'return': [round(r, 6) for r in returns]
        }

        df = pd.DataFrame(data)

        # 缓存用于图表
        global _cached_df
        _cached_df = df

        # 返回预览数据（前10行）
        preview = df.head(10).replace({math.nan: None}).to_dict(orient='records')

        return jsonify({
            'success': True,
            'symbol': 'T.CHT',
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'count': len(df),
            'columns': list(df.columns),
            'numeric_columns': df.select_dtypes(include='number').columns.tolist(),
            'date_columns': ['date'],
            'preview': preview,
            'min_price': float(df['close'].min()),
            'max_price': float(df['close'].max())
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/data/upload', methods=['POST'])
def upload_data():
    """
    上传数据文件：读取全量数据，支持全量分页
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有上传文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400

        import pandas as pd
        import math

        read_index = request.form.get('read_index', 'false').lower() == 'true'

        filename = file.filename
        filepath = BASE_DIR / 'data' / filename
        file.save(filepath)

        # 读取全量数据
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, index_col=False)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filepath, index_col=None)
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

        # 如果有 date 列，确保它是 datetime 类型（用于时间序列分析）
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

        # 去重：基于时间列删除重复行（保留最后一条），防止 set_index 时出现 duplicate labels
        df = _deduplicate_by_time(df)

        df = df.replace({math.nan: None})

        # 缓存全量数据
        global _cached_df, _cached_loading
        _cached_df = df
        _cached_loading = False

        # 返回第一页（1000行）
        page_data = _build_chart_page(df, 1, 1000)
        page_data['filename'] = filename
        page_data['loading_full'] = False

        return jsonify({
            'success': True,
            **page_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/data/code', methods=['POST'])
def run_data_code():
    """
    运行自定义数据获取代码
    """
    try:
        data = request.get_json()
        code = data.get('code', '')

        if not code:
            return jsonify({'success': False, 'error': '缺少代码'}), 400

        # 执行代码（安全限制）
        import pandas as pd
        import numpy as np
        import math

        namespace = {
            'pd': pd,
            'np': np,
            'date_range': pd.date_range,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'代码执行失败: {str(e)}'
            }), 400

        # 查找返回的DataFrame
        result_df = None
        for name, obj in namespace.items():
            if isinstance(obj, pd.DataFrame):
                result_df = obj
                break

        if result_df is None:
            return jsonify({
                'success': False,
                'error': '代码未返回DataFrame'
            }), 400

        # 缓存完整数据用于图表
        global _cached_df
        result_df = _deduplicate_by_time(result_df)
        _cached_df = result_df

        # 直接返回第一页数据（避免前端再发一次请求）
        page_data = _build_chart_page(result_df.replace({math.nan: None}), 1, 1000)
        page_data['count'] = len(result_df)
        return jsonify({'success': True, **page_data})

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _deduplicate_by_time(df):
    """
    基于时间列删除重复行，防止 set_index 时出现 duplicate labels 错误。
    自动识别最精细的时间列进行去重，保留最后一条。
    """
    import pandas as pd
    time_col_candidates = ['exch_time', 'datetime', 'timestamp', 'date', 'trade_date', 'trading_date', 'time']
    time_col = None
    for col in time_col_candidates:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        # 尝试找任意 datetime 类型列
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
                break

    if time_col is not None:
        before = len(df)
        df = df.drop_duplicates(subset=[time_col], keep='last').reset_index(drop=True)
        after = len(df)
        if before > after:
            print(f"[数据去重] 基于列 `{time_col}` 删除 {before - after} 条重复行", flush=True)

    return df


def _build_chart_page(df, page, page_size):
    """构建分页数据（用于表格展示）"""
    import pandas as pd
    all_cols = df.columns.tolist()
    total_rows = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]
    chart_data = {}
    for col in all_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            chart_data[col] = page_df[col].dt.strftime('%Y-%m-%d').tolist()
        else:
            chart_data[col] = page_df[col].tolist()
    return {
        'columns': all_cols,
        'numeric_columns': df.select_dtypes(include='number').columns.tolist(),
        'date_columns': [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c]) or
                        c.lower() in ('date', 'datetime', 'time', 'exch_time', 'trade_date', 'trading_date', 'dt')],
        'data': chart_data,
        'row_count': total_rows,
        'page': page,
        'page_size': page_size,
        'total_pages': (total_rows + page_size - 1) // page_size
    }


@api_bp.route('/data/chart', methods=['GET'])
def get_chart_data():
    """
    获取图表数据，支持分页
    - page: 页码（从1开始），不传则返回所有数据
    - page_size: 每页行数，默认1000
    - cols: 只返回指定列（逗号分隔），用于图表优化
    - downsample: 降采样最大点数，默认3000
    """
    global _cached_df, _cached_loading
    import pandas as pd

    if _cached_df is None:
        return jsonify({'success': False, 'error': '请先加载数据'}), 400

    # 等待后台加载完成（最多等待5秒）
    import time
    waited = 0
    while _cached_loading and waited < 5:
        time.sleep(0.2)
        waited += 0.2

    # 如果请求的页超出已加载范围，返回错误
    page = request.args.get('page', type=int)
    page_size = request.args.get('page_size', default=1000, type=int)
    if page is not None:
        total_rows = len(_cached_df)
        if page > (total_rows + page_size - 1) // page_size:
            return jsonify({
                'success': False,
                'error': '数据仍在后台加载中，请稍后刷新',
                'loading': True
            }), 200

    df = _cached_df
    total_rows = len(df)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    if page is not None:
        return jsonify({'success': True, **_build_chart_page(df, page, page_size)})
    else:
        # 全量/优化模式（图表渲染时使用）
        cols_param = request.args.get('cols', '')
        target_cols = [c.strip() for c in cols_param.split(',') if c.strip() in all_cols] if cols_param else all_cols

        max_points = request.args.get('downsample', default=3000, type=int)
        use_df = df
        sampled = False
        if total_rows > max_points:
            step = total_rows / max_points
            indices = sorted(set(int(i * step) for i in range(max_points)))
            use_df = df.iloc[indices]
            sampled = True

        chart_data = {}
        for col in target_cols:
            if col not in use_df.columns:
                continue
            if pd.api.types.is_datetime64_any_dtype(use_df[col]):
                chart_data[col] = use_df[col].dt.strftime('%Y-%m-%d').tolist()
            else:
                chart_data[col] = use_df[col].tolist()

        return jsonify({
            'success': True,
            'columns': target_cols,
            'numeric_columns': [c for c in numeric_cols if c in target_cols],
            'date_columns': [c for c in target_cols if pd.api.types.is_datetime64_any_dtype(df[c]) or
                            c.lower() in ('date', 'datetime', 'time', 'exch_time', 'trade_date', 'trading_date', 'dt')],
            'data': chart_data,
            'row_count': len(use_df),
            'total_rows': total_rows,
            'sampled': sampled
        })


@api_bp.route('/data/status', methods=['GET'])
def get_data_status():
    """查询数据加载状态"""
    global _cached_df, _cached_loading
    if _cached_df is None:
        return jsonify({'loaded': False, 'loading': False, 'row_count': 0})
    return jsonify({
        'loaded': not _cached_loading,
        'loading': _cached_loading,
        'row_count': len(_cached_df)
    })


@api_bp.route('/data/full', methods=['GET'])
def get_full_data():
    """获取完整缓存数据，供其他API（如优化）直接使用"""
    global _cached_df
    import pandas as pd
    if _cached_df is None or len(_cached_df) == 0:
        return jsonify({'success': False, 'error': '请先在步骤2加载数据'}), 400
    # 返回所有列所有行，转为 JSON 兼容格式
    df = _cached_df.copy()
    result = {'success': True, 'data': {}, 'columns': list(df.columns)}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result['data'][col] = df[col].dt.strftime('%Y-%m-%d').tolist()
        else:
            result['data'][col] = df[col].tolist()
    return jsonify(result)
