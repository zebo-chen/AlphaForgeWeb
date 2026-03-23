"""
AlphaForge Web - Flask后端主入口
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from flask import Flask, render_template, jsonify
from flask_cors import CORS

# 导入API蓝图
from api.hypothesis import api_bp as hypothesis_bp
from api.factor import api_bp as factor_bp
from api.backtest import api_bp as backtest_bp
from api.optimizer import api_bp as optimizer_bp
from api.data import api_bp as data_bp
from api.settings import api_bp as settings_bp

# 创建Flask应用
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# 配置
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'alphaforge-secret-key-2024')
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 最大500MB上传

# 启用CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 注册蓝图
app.register_blueprint(hypothesis_bp, url_prefix='/api')
app.register_blueprint(factor_bp, url_prefix='/api')
app.register_blueprint(backtest_bp, url_prefix='/api')
app.register_blueprint(optimizer_bp, url_prefix='/api')
app.register_blueprint(data_bp, url_prefix='/api')
app.register_blueprint(settings_bp, url_prefix='/api')

# 创建必要的目录
os.makedirs(BASE_DIR / 'data', exist_ok=True)
os.makedirs(BASE_DIR / 'factors_generated', exist_ok=True)
os.makedirs(BASE_DIR / 'output', exist_ok=True)


@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')


@app.errorhandler(413)
def handle_file_too_large(e):
    return jsonify({'success': False, 'error': '文件大小超过500MB限制，请拆分文件或使用代码方式加载'}), 413


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'message': 'AlphaForge Web 运行中'})


if __name__ == '__main__':
    print("=" * 60)
    print("AlphaForge Web 服务器启动中...")
    print("访问地址: http://localhost:5050")
    print("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5050,
        debug=True,
        threaded=True,
        use_reloader=False
    )
