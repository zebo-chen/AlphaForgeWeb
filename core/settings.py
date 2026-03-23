"""
AlphaForge 配置文件 - 手动修改此文件来配置LLM
"""

# ============================================================
# LLM 配置 - 在这里修改你的API密钥和模型选择
# ============================================================

# 选择要使用的模型
# 内置预设: claude-sonnet, claude-haiku, gpt-4, gpt-4o
# 第三方自定义: custom
MODEL_NAME = "custom"

# --------------------------------------------------------
# 方案1: 使用官方API（Claude/OpenAI）
# --------------------------------------------------------
ANTHROPIC_API_KEY = ""  # Claude 系列的密钥

OPENAI_API_KEY = ""     # OpenAI 系列的密钥

# --------------------------------------------------------
# 方案2: 使用第三方API（OpenAI兼容格式）
# 使用第三方模型时，必须同时填写下面两项
# --------------------------------------------------------
CUSTOM_MODEL_NAME = "MiniMax-M2.7"
CUSTOM_API_KEY = "sk-cp-yHprN4zu-hCynvPk6ATG1Gh-1XhTi5yzQQcRNKtdKbhj-mX_CIuwA2I1hfFoFuNx5AfPObnEfdtmNrHeppCiu7CUcBftIC1MgqUta_dxYsbB-5Zm7Qi2etg"     # 第三方API密钥
CUSTOM_BASE_URL = "https://api.minimaxi.com/v1"    # 第三方API地址，如: https://api.example.com/v1/

# --------------------------------------------------------
# 模型参数（一般不需要修改）
# --------------------------------------------------------
MAX_TOKENS = 2048
TEMPERATURE = 0.3

# ============================================================
# 路径配置
# ============================================================
FACTORS_DIR = "./factors_generated"
DATA_DIR = "./data"
OUTPUT_DIR = "./output"

# ============================================================
# 完整配置示例:
#
# 1. 官方 Claude:
#    MODEL_NAME = "claude-sonnet"
#    ANTHROPIC_API_KEY = "sk-ant-..."
#
# 2. 官方 OpenAI:
#    MODEL_NAME = "gpt-4"
#    OPENAI_API_KEY = "sk-..."
#
# 3. 第三方 Kimi:
#    MODEL_NAME = "kimi"
#    CUSTOM_MODEL_NAME = "kimi-k25"
#    CUSTOM_API_KEY = "sk-kimi-xxx"
#    CUSTOM_BASE_URL = "https://api.kimi.com/v1/"
#
# 4. 第三方 Minimax:
#    MODEL_NAME = "minimax"
#    CUSTOM_MODEL_NAME = "abab6.5-chat"
#    CUSTOM_API_KEY = "sk-minimax-xxx"
#    CUSTOM_BASE_URL = "https://api.minimax.chat/v1/"
#
# 5. 第三方 DeepSeek:
#    MODEL_NAME = "deepseek"
#    CUSTOM_MODEL_NAME = "deepseek-chat"
#    CUSTOM_API_KEY = "sk-deepseek-xxx"
#    CUSTOM_BASE_URL = "https://api.deepseek.com/v1/"
# ============================================================
