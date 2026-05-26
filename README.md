# Advanced Verifying Agent

An LLM‑based agent that generates an answer, verifies it with a second LLM, and corrects iteratively.

## Features
- Supports **mock** (no API key), **OpenAI**, and **Anthropic** backends
- Multi‑round verification and correction
- Result caching to reduce API calls
- CLI interface
- Extensible prompt templates

## Installation

```bash
git clone https://github.com/vimarsht8/verifying-agent.git
cd verifying-agent
pip install -r requirements.txt