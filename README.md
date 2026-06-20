<!-- Add this at the VERY TOP of README.md -->

<div align="center">

# 🤖 Advanced Verifying Agent

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/YOUR_USERNAME/agent-verify/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/agent-verify/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/agent-verify)](https://github.com/YOUR_USERNAME/agent-verify/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/agent-verify)](https://github.com/YOUR_USERNAME/agent-verify/network)

**An intelligent AI agent that self-verifies and corrects its answers using LLMs**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [Contributing](#-contributing)

</div>

---

## 📖 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Tools](#-tools)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)
# 🤖 Advanced Verifying Agent

An intelligent AI agent that can answer questions, verify its own answers, and automatically correct mistakes using LLMs (OpenAI, Anthropic) with advanced features.

## ✨ Features

- **Self-Verification**: Automatically checks and corrects its own answers
- **Async Support**: Fully async for high performance
- **Persistent Cache**: Disk-based caching to save API costs
- **Chain-of-Thought**: Step-by-step verification for better accuracy
- **External Tools**: Calculator, web search, and knowledge base integration
- **Metrics Collection**: Track performance and usage statistics
- **Multiple LLM Providers**: OpenAI, Anthropic, or mock for testing
- **Streaming Support**: Real-time response streaming

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt