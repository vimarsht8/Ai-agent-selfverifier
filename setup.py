"""
Setup configuration for Advanced Verifying Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced-verifying-agent",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent AI agent that self-verifies and corrects answers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/agent-verify",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiofiles>=23.0.0",
        "pyyaml>=6.0.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.100.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent=advanced_agent_v2:main",
        ],
    },
)