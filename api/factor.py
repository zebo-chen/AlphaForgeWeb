"""
因子构建API
"""

import sys
import json
import re
from pathlib import Path
from flask import request, jsonify, Blueprint

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('factor', __name__)
from core.factor_builder import FactorBuilder, FactorCode


@api_bp.route('/factor/build', methods=['POST'])
def build_factor():
    """
    根据假设构建因子代码（SSE流式日志）
    """
    from flask import Response
    import pandas as pd
    import numpy as np
    import time

    # 必须在 generator 创建前读取 request，否则后续 request context 会失效
    payload = request.get_json()
    hypothesis_data = payload.get('hypothesis', {})
    columns = payload.get('columns', [])
    sample_data = payload.get('sample_data', {})

    if not hypothesis_data:
        return jsonify({'success': False, 'error': '缺少假设数据'}), 400

    class Hypothesis:
        def __init__(self, d):
            self.name = d.get('name', 'Unknown')
            self.logic = d.get('logic', '')
            self.data_requirements = d.get('data_req', [])
            self.expected_direction = d.get('direction', 'positive')
            self.formula_idea = d.get('formula_idea', '')
            self.english_name = d.get('english_name', '')

        def to_dict(self):
            return {
                'name': self.name,
                'logic': self.logic,
                'data_requirements': self.data_requirements,
                'expected_direction': self.expected_direction
            }

    hypothesis = Hypothesis(hypothesis_data)
    df = pd.DataFrame(sample_data) if sample_data else None
    builder = FactorBuilder()

    def log(msg):
        yield f"data: {json.dumps({'type': 'log', 'msg': msg})}\n\n"

    def generate():
        try:
            yield from log(f"[1/4] 构建 Prompt...")
            t0 = time.time()
            prompt = builder._build_prompt(hypothesis, columns=columns or None)
            yield from log(f"[2/4] 调用 LLM 生成代码（模型: {builder.config.model}）...")

            raw_content = ""
            stream_response = builder.client.chat.completions.create(
                model=builder.config.model,
                max_tokens=builder.config.max_tokens,
                temperature=builder.config.temperature,
                timeout=builder.config.timeout,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream_response:
                delta = chunk.choices[0].delta.content or ""
                raw_content += delta
                if len(raw_content) <= 200:
                    yield from log(f"[2/4] {raw_content}")
                elif len(raw_content) % 100 < 20:
                    yield from log(f"[2/4] ...（已输出 {len(raw_content)} 字）")
            yield from log(f"[2/4] LLM 响应完成，耗时 {time.time()-t0:.1f}s，共 {len(raw_content)} 字")

            code = builder._parse_code(raw_content)
            code = builder._sanitize_code(code)

            if df is not None:
                yield from log(f"[3/4] 执行代码验证（样本: {len(df.head(100))} 行）...")
                try:
                    ns = {'pd': pd, 'np': np}
                    exec(code, ns)
                    factor_class = next((obj for name, obj in ns.items()
                                        if isinstance(obj, type) and name.endswith('Factor')), None)
                    if factor_class is None:
                        raise RuntimeError('未找到以Factor结尾的类')
                    result = factor_class().calculate(df.head(100))
                    if not isinstance(result, pd.Series):
                        raise TypeError(f'calculate() 应返回 pd.Series，实际返回 {type(result)}')
                    yield from log(f"[3/4] 代码验证通过 ✓")
                except Exception as e:
                    yield from log(f"[3/4] 首次验证失败，LLM 自动修复中...")
                    try:
                        debug_prompt = f"""以下Python因子代码执行报错，请修复。

错误: {e}

代码:
{code}

数据列: {list(df.columns)}

要求：只修改导致报错的部分，确保返回 pd.Series，直接输出修复后的完整代码（用```python包裹）："""
                        debug_content = ""
                        stream_resp2 = builder.client.chat.completions.create(
                            model=builder.config.model,
                            max_tokens=builder.config.max_tokens,
                            temperature=0.1,
                            timeout=builder.config.timeout,
                            messages=[{"role": "user", "content": debug_prompt}],
                            stream=True
                        )
                        for chunk in stream_resp2:
                            delta = chunk.choices[0].delta.content or ""
                            debug_content += delta
                            if len(debug_content) <= 200:
                                yield from log(f"[3/4] 修复: {debug_content}")
                            elif len(debug_content) % 100 < 20:
                                yield from log(f"[3/4] ...（已输出 {len(debug_content)} 字）")
                        code = builder._parse_code(debug_content)
                        code = builder._sanitize_code(code)
                        ns2 = {'pd': pd, 'np': np}
                        exec(code, ns2)
                        fc2 = next((obj for n, obj in ns2.items()
                                   if isinstance(obj, type) and n.endswith('Factor')), None)
                        if fc2:
                            fc2().calculate(df.head(100))
                        yield from log(f"[3/4] 自动修复成功 ✓")
                    except Exception as e2:
                        yield from log(f"[3/4] 自动修复失败: {e2}")
                        yield f"data: {json.dumps({'type': 'error', 'msg': f'代码修复失败: {e2}'})}\n\n"
                        return
            else:
                yield from log(f"[3/4] 跳过验证（无数据）")

            class_name = re.search(r'class\s+(\w+Factor)', code)
            class_name = class_name.group(1) if class_name else 'Factor'
            yield from log(f"[4/4] 代码生成完成 ✓")
            yield f"data: {json.dumps({'type': 'done', 'success': True, 'code': code, 'class_name': class_name, 'description': hypothesis.to_dict()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@api_bp.route('/factor/validate', methods=['POST'])
def validate_factor():
    """
    验证因子代码

    请求体:
    {
        "code": "Python代码",
        "data": {...}  // 测试数据
    }

    返回:
    {
        "success": true,
        "valid": true,
        "error": null
    }
    """
    try:
        data = request.get_json()
        code = data.get('code', '')

        if not code:
            return jsonify({'success': False, 'error': '缺少代码'}), 400

        # 简单验证：检查是否有危险操作
        dangerous = ['import os', 'import sys', 'open(', 'exec(', 'eval(',
                     'subprocess', '__import__', 'file', 'socket']

        for d in dangerous:
            if d in code:
                return jsonify({
                    'success': False,
                    'valid': False,
                    'error': f'代码包含危险操作: {d}'
                }), 400

        return jsonify({
            'success': True,
            'valid': True,
            'error': None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/factor/run', methods=['POST'])
def run_factor():
    """
    运行因子代码

    请求体:
    {
        "code": "Python代码",
        "class_name": "CurveSlopeFactor",
        "params": {...},
        "data": [...]
    }

    返回:
    {
        "success": true,
        "factor_values": [...],
        "min": -0.15,
        "max": 0.12
    }
    """
    try:
        data = request.get_json()
        code = data.get('code', '')
        class_name = data.get('class_name', '')
        params = data.get('params', {})
        input_data = data.get('data', {})

        if not code or not class_name:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 执行代码
        import pandas as pd
        import numpy as np

        # 创建命名空间
        namespace = {'pd': pd, 'np': np}

        try:
            exec(code, namespace)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'代码执行失败: {str(e)}'
            }), 400

        # 查找因子类
        factor_class = namespace.get(class_name)
        if not factor_class:
            return jsonify({
                'success': False,
                'error': f'未找到类: {class_name}'
            }), 400

        # 创建DataFrame
        if input_data:
            df = pd.DataFrame(input_data)
        else:
            # 使用示例数据
            dates = pd.date_range('2020-01-01', periods=500, freq='D')
            np.random.seed(42)
            df = pd.DataFrame({
                'price': 100 + np.random.randn(500).cumsum(),
                'close': 100 + np.random.randn(500).cumsum(),
                'volume': np.random.randint(10000, 100000, 500)
            }, index=dates)
            df.index.name = 'date'

        # 运行因子
        try:
            instance = factor_class(params=params) if params else factor_class()
            factor_values = instance.calculate(df)

            # 转换为列表
            if hasattr(factor_values, 'tolist'):
                values = factor_values.tolist()
            else:
                values = list(factor_values)

            return jsonify({
                'success': True,
                'factor_values': values[-20:] if len(values) > 20 else values,  # 限制返回数量
                'min': float(factor_values.min()) if len(factor_values) > 0 else 0,
                'max': float(factor_values.max()) if len(factor_values) > 0 else 0,
                'count': len(factor_values)
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'因子计算失败: {str(e)}'
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/factor/modify', methods=['POST'])
def modify_factor():
    """
    根据用户建议修改因子代码（调用LLM）

    请求体:
    {
        "original_code": "原始代码",
        "suggestion": "修改建议",
        "class_name": "类名"
    }

    返回:
    {
        "success": true,
        "code": "修改后的代码"
    }
    """
    try:
        data = request.get_json()
        original_code = data.get('original_code', '')
        suggestion = data.get('suggestion', '')
        class_name = data.get('class_name', 'Factor')

        if not original_code or not suggestion:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 调用 LLM 修改代码
        builder = FactorBuilder()
        modified_code = builder.modify_code(original_code, suggestion, class_name)

        return jsonify({
            'success': True,
            'code': modified_code
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/factor/save', methods=['POST'])
def save_factor():
    """
    将因子保存到因子库

    请求体:
    {
        "factor_name": "CurveSlope",
        "class_name": "CurveSlopeFactor",
        "factor_code": "...",
        "hypothesis": {...},
        "metrics": { "ic_mean": 0.028, "ir": 0.42, ... }
    }

    返回:
    {
        "success": true,
        "factor_id": "CurveSlope_20260322_143052"
    }
    """
    import uuid
    import datetime

    try:
        data = request.get_json()
        factor_name = data.get('factor_name', '').strip()
        class_name = data.get('class_name', '')
        factor_code = data.get('factor_code', '')
        hypothesis = data.get('hypothesis', {})
        metrics = data.get('metrics', {})

        if not factor_name or not factor_code or not class_name:
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400

        # 生成唯一 ID
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        factor_id = f"{factor_name}_{ts}"

        # 构建保存数据
        factor_data = {
            'id': factor_id,
            'factor_name': factor_name,
            'class_name': class_name,
            'factor_code': factor_code,
            'hypothesis': hypothesis,
            'metrics': metrics,
            'saved_at': datetime.datetime.now().isoformat(),
        }

        # 确保目录存在
        factors_dir = BASE_DIR / 'factors_generated'
        factors_dir.mkdir(exist_ok=True)

        # 保存为 JSON 文件
        file_path = factors_dir / f'{factor_id}.json'
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(factor_data, f, ensure_ascii=False, indent=2)

        print(f"[因子库] 保存因子: {factor_id}", flush=True)

        return jsonify({
            'success': True,
            'factor_id': factor_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/factor/list', methods=['GET'])
def list_factors():
    """
    获取因子库列表

    返回:
    {
        "success": true,
        "factors": [
            {
                "id": "CurveSlope_20260322_143052",
                "factor_name": "CurveSlope",
                "class_name": "CurveSlopeFactor",
                "metrics": { "ic_mean": 0.028, "ir": 0.42 },
                "saved_at": "2026-03-22T14:30:52"
            },
            ...
        ]
    }
    """
    try:
        factors_dir = BASE_DIR / 'factors_generated'
        factors_dir.mkdir(exist_ok=True)

        factors = []
        for file_path in sorted(factors_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                factors.append({
                    'id': data.get('id', ''),
                    'factor_name': data.get('factor_name', ''),
                    'class_name': data.get('class_name', ''),
                    'metrics': data.get('metrics', {}),
                    'saved_at': data.get('saved_at', ''),
                })
            except Exception:
                continue

        return jsonify({
            'success': True,
            'factors': factors
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/factor/<factor_id>', methods=['DELETE'])
def delete_factor(factor_id):
    """
    删除因子
    """
    try:
        factors_dir = BASE_DIR / 'factors_generated'
        file_path = factors_dir / f'{factor_id}.json'
        if not file_path.exists():
            return jsonify({'success': False, 'error': '因子不存在'}), 404
        file_path.unlink()
        print(f"[因子库] 删除因子: {factor_id}", flush=True)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/factor/<factor_id>', methods=['GET'])
def get_factor(factor_id):
    """
    获取单个因子的完整信息

    返回:
    {
        "success": true,
        "factor": { ...完整数据... }
    }
    """
    try:
        factors_dir = BASE_DIR / 'factors_generated'
        file_path = factors_dir / f'{factor_id}.json'

        if not file_path.exists():
            return jsonify({'success': False, 'error': '因子不存在'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return jsonify({
            'success': True,
            'factor': data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500, 500
