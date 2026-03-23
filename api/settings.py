"""
设置API - 读取和保存 settings.py 配置
"""

import re
import sys
from pathlib import Path
from flask import request, jsonify, Blueprint

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

api_bp = Blueprint('settings', __name__)

SETTINGS_FILE = BASE_DIR / 'core' / 'settings.py'

# 需要暴露给前端的字段及元信息
FIELDS = {
    # LLM 配置
    'MODEL_NAME':            {'section': 'LLM 配置', 'label': '模型', 'type': 'select',
        'options': ['claude-sonnet', 'claude-haiku', 'gpt-4', 'gpt-4o', 'custom']},
    'ANTHROPIC_API_KEY':    {'section': 'LLM 配置', 'label': 'Anthropic API Key', 'type': 'password'},
    'OPENAI_API_KEY':       {'section': 'LLM 配置', 'label': 'OpenAI API Key', 'type': 'password'},
    'CUSTOM_MODEL_NAME':    {'section': 'LLM 配置', 'label': '自定义模型名', 'type': 'text',
        'placeholder': '如 MiniMax-M2.7'},
    'CUSTOM_API_KEY':       {'section': 'LLM 配置', 'label': '自定义 API Key', 'type': 'password'},
    'CUSTOM_BASE_URL':      {'section': 'LLM 配置', 'label': '自定义接口地址', 'type': 'text',
        'placeholder': '如 https://api.minimaxi.com/v1'},
    'MAX_TOKENS':           {'section': 'LLM 配置', 'label': '最大 Token 数', 'type': 'number'},
    'TEMPERATURE':          {'section': 'LLM 配置', 'label': 'Temperature', 'type': 'number', 'step': '0.1'},

    # 路径配置
    'FACTORS_DIR':          {'section': '路径配置', 'label': '因子目录', 'type': 'text'},
    'DATA_DIR':             {'section': '路径配置', 'label': '数据目录', 'type': 'text'},
    'OUTPUT_DIR':           {'section': '路径配置', 'label': '输出目录', 'type': 'text'},
}


def _parse_line_value(line):
    """解析一行 Python 设置行中的值，正确处理引号和列表"""
    # 去掉行内注释（# 后面且不在引号内的内容）
    in_quote = False
    quote_char = None
    for i, ch in enumerate(line):
        if ch in ('"', "'") and (i == 0 or line[i-1] != '\\'):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif ch == quote_char:
                in_quote = False
                quote_char = None
        elif ch == '#' and not in_quote:
            line = line[:i].rstrip()
            break

    line = line.strip()
    if not line:
        return None

    # 提取 = 右侧的内容
    eq_idx = line.find('=')
    if eq_idx == -1:
        return None
    raw = line[eq_idx+1:].strip()

    # 判断是否为 Python 字符串（带引号）
    if (raw.startswith('"') or raw.startswith("'")) and not (raw.startswith('["') or raw.startswith("['") or raw.startswith('{"') or raw.startswith("{'")):
        # 去掉外层引号
        if len(raw) >= 2 and raw[-1] == raw[0]:
            return raw[1:-1]
    return raw


def parse_settings():
    """解析 settings.py 文件，返回字段名->值的字典"""
    content = SETTINGS_FILE.read_text()
    result = {}
    for field in FIELDS:
        pattern = rf'^{re.escape(field)}\s*='
        for line in content.splitlines():
            if re.match(pattern, line):
                raw = _parse_line_value(line)
                if raw is not None:
                    result[field] = raw
                break
    return result


def _get_original_format(content, field):
    """获取原始文件中某字段 = 右侧的原始值（保留引号），用于判断写入格式"""
    pattern = rf'^{re.escape(field)}\s*='
    for line in content.splitlines():
        if re.match(pattern, line):
            eq_idx = line.find('=')
            raw = line[eq_idx+1:].strip()
            return raw
    return None


def build_settings_lines(values):
    """根据当前 settings.py 内容，只更新指定字段，保持其余内容不变"""
    content = SETTINGS_FILE.read_text()
    lines = content.splitlines(keepends=True)
    for field, new_value in values.items():
        if field not in FIELDS:
            continue
        orig_format = _get_original_format(content, field)

        # 确定写入格式
        if orig_format is not None and (orig_format.startswith('[') or orig_format.startswith('{')):
            # 列表/字典：直接写入
            formatted = new_value
        elif orig_format is not None and (orig_format.startswith('"') or orig_format.startswith("'")):
            # 字符串：加引号
            formatted = f'"{new_value}"'
        else:
            # 数字或其他：直接写入
            formatted = new_value

        # 找到对应行并替换
        for i, line in enumerate(lines):
            if re.match(rf'^{re.escape(field)}\s*=', line):
                # 保留行内注释
                comment = ''
                in_quote = False
                for j, ch in enumerate(line):
                    if ch in ('"', "'"):
                        in_quote = not in_quote
                    elif ch == '#' and not in_quote:
                        comment = line[j:]
                        line = line[:j].rstrip()
                        break
                lines[i] = f'{field} = {formatted}{comment}\n'
                break

    return ''.join(lines)


@api_bp.route('/settings', methods=['GET'])
def get_settings():
    """读取当前配置"""
    try:
        values = parse_settings()
        # 把 API key 脱敏（只显示前4后4位）
        for key in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'CUSTOM_API_KEY']:
            v = values.get(key, '').strip('"').strip("'")
            if v and len(v) > 8:
                values[key + '_masked'] = v[:4] + '****' + v[-4:]
            else:
                values[key + '_masked'] = ''
        return jsonify({'success': True, 'settings': values, 'fields': FIELDS})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/settings', methods=['POST'])
def save_settings():
    """保存配置"""
    try:
        data = request.get_json()
        values = data.get('settings', {})

        # 验证必填字段
        model = values.get('MODEL_NAME', '').strip('"').strip("'")
        if model == 'custom':
            custom_key = values.get('CUSTOM_API_KEY', '').strip('"').strip("'")
            custom_url = values.get('CUSTOM_BASE_URL', '').strip('"').strip("'")
            if not custom_key or not custom_url:
                return jsonify({'success': False, 'error': '使用自定义模型时，CUSTOM_API_KEY 和 CUSTOM_BASE_URL 均不能为空'}), 400

        new_content = build_settings_lines(values)
        SETTINGS_FILE.write_text(new_content)

        return jsonify({'success': True})
    except PermissionError:
        return jsonify({'success': False, 'error': '没有写入权限，请检查 core/settings.py 文件权限'}), 403
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
