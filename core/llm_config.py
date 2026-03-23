"""
LLM配置管理模块 - 支持多模型配置和灵活切换
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """支持的LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    CUSTOM = "custom"  # 自定义/OpenAI兼容接口


@dataclass
class LLMConfig:
    """LLM配置数据结构"""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: int = 60
    # 额外参数
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class LLMConfigManager:
    """LLM配置管理器"""

    # 预定义的模型配置模板
    PRESET_MODELS = {
        "claude-sonnet": LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            api_key="${ANTHROPIC_API_KEY}",
            max_tokens=4000,
            temperature=0.3
        ),
        "claude-haiku": LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-haiku-20240307",
            api_key="${ANTHROPIC_API_KEY}",
            max_tokens=4000,
            temperature=0.3
        ),
        "gpt-4": LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="${OPENAI_API_KEY}",
            max_tokens=4000,
            temperature=0.3
        ),
        "gpt-4o": LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="${OPENAI_API_KEY}",
            max_tokens=4000,
            temperature=0.3
        ),
        "kimi": LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="saas-kimi-k25",
            api_key="${KIMI_API_KEY}",
            base_url="${KIMI_BASE_URL}",
            max_tokens=4000,
            temperature=0.3
        ),
    }

    def __init__(self):
        self._current_config: Optional[LLMConfig] = None
        self._env_prefix = "ALPHAFORGE_"

    def get_config(self, model_name: str = None) -> LLMConfig:
        """
        获取配置，优先级：
        1. 传入的model_name参数
        2. 配置文件 settings.py
        3. 环境变量 ALPHAFORGE_MODEL
        4. 默认配置（kimi）
        """
        # 1. 尝试从 settings.py 读取配置
        settings_config = self._load_from_settings()
        if settings_config:
            self._current_config = settings_config
            return settings_config

        # 2. 确定模型名称（从环境变量或默认）
        if model_name is None:
            model_name = os.getenv(f"{self._env_prefix}MODEL", "kimi")

        # 2. 如果是预设模型，获取模板并填充环境变量
        if model_name in self.PRESET_MODELS:
            config = self._resolve_env_vars(self.PRESET_MODELS[model_name])
        else:
            # 3. 自定义模型配置
            config = self._load_custom_config(model_name)

        self._current_config = config
        return config

    def _resolve_env_vars(self, config: LLMConfig) -> LLMConfig:
        """解析配置中的环境变量占位符"""
        # 复制一份新的配置
        resolved = LLMConfig(
            provider=config.provider,
            model=config.model,
            api_key=self._resolve_value(config.api_key),
            base_url=self._resolve_value(config.base_url) if config.base_url else None,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            extra_params=config.extra_params.copy()
        )
        return resolved

    def _resolve_value(self, value: str) -> str:
        """解析值中的环境变量，如 ${API_KEY} -> 实际值"""
        if value is None:
            return None
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, "")
        return value

    def _load_custom_config(self, model_name: str) -> LLMConfig:
        """从环境变量加载自定义配置"""
        # 尝试从环境变量加载
        provider_str = os.getenv(f"{self._env_prefix}PROVIDER", "custom")
        provider = LLMProvider(provider_str)

        api_key = os.getenv(f"{self._env_prefix}API_KEY", "")
        base_url = os.getenv(f"{self._env_prefix}BASE_URL")

        # 如果没有设置API key，尝试提供商特定的环境变量
        if not api_key:
            if provider == LLMProvider.ANTHROPIC:
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif provider == LLMProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY", "")

        return LLMConfig(
            provider=provider,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(os.getenv(f"{self._env_prefix}MAX_TOKENS", "4000")),
            temperature=float(os.getenv(f"{self._env_prefix}TEMPERATURE", "0.3")),
        )

    def _load_from_settings(self) -> Optional[LLMConfig]:
        """从 settings.py 文件加载配置"""
        try:
            # 获取项目根目录（core目录）
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent  # core目录

            # 检查 settings.py 是否存在
            settings_path = project_root / "settings.py"
            if not settings_path.exists():
                print(f"[警告] settings.py 不存在于 {settings_path}")
                return None

            # 动态导入 settings 模块
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            import settings

            # 检查是否有配置
            model_name = getattr(settings, 'MODEL_NAME', None)
            if not model_name:
                return None

            # 根据 MODEL_NAME 获取对应的 API key
            api_key = ""
            base_url = None
            provider = LLMProvider.CUSTOM

            if model_name.startswith('claude'):
                api_key = getattr(settings, 'ANTHROPIC_API_KEY', '')
                provider = LLMProvider.ANTHROPIC
                # 从预设模型获取实际模型名称
                if model_name in self.PRESET_MODELS:
                    model_name = self.PRESET_MODELS[model_name].model
            elif model_name.startswith('gpt'):
                api_key = getattr(settings, 'OPENAI_API_KEY', '')
                provider = LLMProvider.OPENAI
                if model_name in self.PRESET_MODELS:
                    model_name = self.PRESET_MODELS[model_name].model
            else:
                # 第三方自定义API（OpenAI兼容格式）
                # 必须同时填写 CUSTOM_API_KEY 和 CUSTOM_BASE_URL
                api_key = getattr(settings, 'CUSTOM_API_KEY', '')
                base_url = getattr(settings, 'CUSTOM_BASE_URL', '')

                # 检查是否填写完整
                if not api_key or not base_url:
                    print(f"[配置] 使用第三方模型 '{model_name}' 需要同时填写 CUSTOM_API_KEY 和 CUSTOM_BASE_URL")
                    return None

                provider = LLMProvider.CUSTOM
                # 检查是否有自定义模型名称
                custom_model = getattr(settings, 'CUSTOM_MODEL_NAME', '')
                if custom_model:
                    model_name = custom_model
                elif model_name in self.PRESET_MODELS:
                    model_name = self.PRESET_MODELS[model_name].model

            # 如果没有 API key，返回 None（让系统回退到其他配置方式）
            if not api_key:
                print(f"[配置] settings.py 中找到模型 '{model_name}' 但 API key 为空，使用模拟模式")
                return None

            print(f"[配置] 从 settings.py 加载配置: {provider.value}/{model_name}")

            return LLMConfig(
                provider=provider,
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=getattr(settings, 'MAX_TOKENS', 4000),
                temperature=getattr(settings, 'TEMPERATURE', 0.3),
            )

        except Exception as e:
            # 如果加载失败，静默返回 None
            return None

    def create_client(self, config: LLMConfig = None):
        """根据配置创建对应的客户端"""
        if config is None:
            config = self._current_config or self.get_config()

        if config.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_client(config)
        else:
            # OpenAI兼容接口
            return self._create_openai_compatible_client(config)

    def _create_anthropic_client(self, config: LLMConfig):
        """创建Anthropic客户端"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("请安装anthropic包: pip install anthropic")

    def _create_openai_compatible_client(self, config: LLMConfig):
        """创建OpenAI兼容客户端"""
        from openai import OpenAI

        client_kwargs = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }

        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        return OpenAI(**client_kwargs)

    def list_available_models(self) -> Dict[str, str]:
        """列出所有可用的预设模型"""
        return {
            name: f"{config.provider.value}/{config.model}"
            for name, config in self.PRESET_MODELS.items()
        }

    def get_default_prompt_config(self) -> Dict[str, Any]:
        """获取默认的Prompt调用参数"""
        config = self._current_config or self.get_config()
        return {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }


# 全局配置管理器实例
_config_manager = LLMConfigManager()


def get_llm_config(model_name: str = None) -> LLMConfig:
    """获取LLM配置的便捷函数"""
    return _config_manager.get_config(model_name)


def create_llm_client(config: LLMConfig = None):
    """创建LLM客户端的便捷函数"""
    return _config_manager.create_client(config)


def list_models():
    """列出可用模型的便捷函数"""
    return _config_manager.list_available_models()
