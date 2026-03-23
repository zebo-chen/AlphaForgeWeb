"""
API路由 - 导出所有蓝图
"""

from api.hypothesis import api_bp as hypothesis_bp
from api.factor import api_bp as factor_bp
from api.backtest import api_bp as backtest_bp
from api.optimizer import api_bp as optimizer_bp
from api.data import api_bp as data_bp

__all__ = ['hypothesis_bp', 'factor_bp', 'backtest_bp', 'optimizer_bp', 'data_bp']
