"""
Configuration management for the Advanced Verifying Agent.
Supports environment variables, YAML config files, and Pydantic validation.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import yaml

load_dotenv()


class ModelConfig(BaseModel):
    """Model-specific configuration"""
    provider: str = Field(default="mock", description="LLM provider: mock, openai, anthropic")
    model_name: Optional[str] = Field(default=None, description="Specific model name")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout: int = Field(default=30, ge=1)


class CacheConfig(BaseModel):
    """Cache configuration"""
    enabled: bool = True
    ttl_seconds: int = Field(default=3600, ge=0)
    max_entries: int = Field(default=500, ge=0)
    cache_dir: str = Field(default="./.agent_cache")
    use_redis: bool = False
    redis_url: Optional[str] = None


class VerificationConfig(BaseModel):
    """Verification settings"""
    max_rounds: int = Field(default=2, ge=1, le=5)
    use_chain_of_thought: bool = True
    enable_external_tools: bool = True
    tools: List[str] = Field(default_factory=list)
    strict_mode: bool = False  # If True, fail on any uncertainty


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file: Optional[str] = None
    enable_console: bool = True


class AgentConfig(BaseModel):
    """Main configuration for the agent"""
    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Custom prompts
    system_prompt_generate: Optional[str] = None
    system_prompt_verify: Optional[str] = None
    system_prompt_correct: Optional[str] = None
    
    # Performance
    max_parallel_requests: int = Field(default=5, ge=1)
    enable_streaming: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load configuration from environment variables"""
        return cls(
            model=ModelConfig(
                provider=os.getenv("LLM_PROVIDER", "mock"),
                model_name=os.getenv("MODEL_NAME"),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1024"))
            ),
            cache=CacheConfig(
                enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
                ttl_seconds=int(os.getenv("CACHE_TTL", "3600")),
                max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "500"))
            )
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


# Default configuration instance
default_config = AgentConfig()