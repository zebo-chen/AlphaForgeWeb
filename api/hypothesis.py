"""
假设生成API - 使用HypothesisGenerator的chat模式真正调用LLM，支持流式输出
"""

import sys
import json
from pathlib import Path
from flask import request, jsonify, Blueprint, Response, stream_with_context
from flask_cors import cross_origin

# 添加项目根目录到路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('hypothesis', __name__)

# 全局存储当前对话生成器
_hypothesis_generator = {}


def stream_llm_response(generator, message, session_id, step):
    """流式输出LLM响应"""
    try:
        # 设置超时
        generator.config.timeout = 120

        # 调用chat方法（流式输出）
        generator.messages.append({"role": "user", "content": message})

        import requests

        config = generator.config
        url = f"{config.base_url}/chat/completions" if config.base_url else "https://api.minimaxi.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": config.model,
            "messages": generator.messages,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": True
        }

        # 流式请求
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)

        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'API错误: {response.status_code}'})}\n\n"
            return

        full_content = ""

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                full_content += content
                                # 流式发送
                                yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"
                    except:
                        pass

        # 保存完整响应到messages
        generator.messages.append({"role": "assistant", "content": full_content})

        # 完成
        new_step = (step + 1) % 3
        show_hypotheses = new_step == 0

        yield f"data: {json.dumps({'content': '', 'done': True, 'step': new_step, 'show_hypotheses': show_hypotheses, 'session_id': session_id})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@api_bp.route('/hypothesis/chat', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat_hypothesis():
    """
    假设聊天对话接口 - 流式输出
    """
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        step = data.get('step', 0)

        if not message:
            return jsonify({'success': False, 'error': '请输入消息'}), 400

        # 获取或创建会话的生成器
        if session_id not in _hypothesis_generator:
            from core.hypothesis_generator import HypothesisGenerator
            _hypothesis_generator[session_id] = HypothesisGenerator()

        generator = _hypothesis_generator[session_id]

        # 检查API Key是否有效
        if not generator.config.api_key:
            return jsonify({
                'success': False,
                'error': '未配置有效的API Key'
            }), 500

        # 流式返回
        return Response(
            stream_with_context(stream_llm_response(generator, message, session_id, step)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/hypothesis/chat_simple', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat_hypothesis_simple():
    """非流式版本（备用）"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        session_id = data.get('session_id', 'default')
        step = data.get('step', 0)

        if not message:
            return jsonify({'success': False, 'error': '请输入消息'}), 400

        if session_id not in _hypothesis_generator:
            from core.hypothesis_generator import HypothesisGenerator
            _hypothesis_generator[session_id] = HypothesisGenerator()

        generator = _hypothesis_generator[session_id]

        if not generator.config.api_key:
            return jsonify({'success': False, 'error': '未配置有效的API Key'}), 500

        # 非流式调用
        generator.config.timeout = 120
        llm_response = generator.chat(message)

        new_step = (step + 1) % 3

        return jsonify({
            'success': True,
            'response': llm_response,
            'session_id': session_id,
            'step': new_step,
            'show_hypotheses': new_step == 0
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/hypothesis/extract', methods=['POST', 'OPTIONS'])
@cross_origin()
def extract_hypotheses():
    """从当前对话中提取假设"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')

        if session_id not in _hypothesis_generator:
            return jsonify({'success': False, 'error': '会话不存在'}), 400

        generator = _hypothesis_generator[session_id]
        hypotheses = generator.extract()

        result = []
        for h in hypotheses:
            result.append({
                'id': h.id,
                'name': h.name,
                'english_name': h.english_name,
                'logic': h.logic,
                'confidence': int(h.confidence_score * 10),
                'data_req': h.data_requirements,
                'direction': h.expected_direction,
                'economic_basis': h.economic_basis,
                'formula_idea': h.formula_idea
            })

        return jsonify({
            'success': True,
            'hypotheses': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/hypothesis/reset', methods=['POST', 'OPTIONS'])
@cross_origin()
def reset_hypothesis():
    """重置对话"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', 'default')

        if session_id in _hypothesis_generator:
            _hypothesis_generator[session_id].reset()

        return jsonify({
            'success': True,
            'message': '对话已重置'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
