#!/usr/bin/env python3
"""
AdvancedVerifyingAgent v2.0
Async, persistent cache, chain-of-thought verification, external tools, retries, streaming.
"""

import asyncio
import os
import re
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Awaitable
from functools import partial
import argparse

import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Optional LLM SDKs
try:
    import openai
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

load_dotenv()
logger = logging.getLogger("AdvancedVerifyingAgent")


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------
class AgentConfig(BaseModel):
    provider: str = "mock"
    model_name: Optional[str] = None
    max_verification_rounds: int = Field(default=2, ge=1, le=5)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    cache_dir: str = "./.agent_cache"
    max_cache_entries: int = Field(default=500, ge=0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_base_delay: float = 1.0
    use_chain_of_thought_verification: bool = True
    external_tools: List[str] = []          # e.g., ["calculator"]
    system_prompt_generate: Optional[str] = None
    system_prompt_verify: Optional[str] = None
    system_prompt_correct: Optional[str] = None

    @validator("provider")
    def validate_provider(cls, v):
        allowed = {"mock", "openai", "anthropic"}
        if v not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v

    @validator("model_name", always=True)
    def set_default_model(cls, v, values):
        if v is None:
            provider = values.get("provider")
            if provider == "openai":
                return "gpt-4o-mini"
            elif provider == "anthropic":
                return "claude-3-haiku-20240307"
            else:
                return "mock-llm"
        return v


# ---------------------------------------------------------------------------
# Async Persistent Cache
# ---------------------------------------------------------------------------
class PersistentCache:
    def __init__(self, cache_dir: str, ttl_seconds: int, max_entries: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        # In-memory index: key -> (timestamp, file_name)
        self.index: Dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()

    def _key_to_filename(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest() + ".json"

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self.index.get(key)
            if entry:
                ts, fname = entry
                if time.time() - ts <= self.ttl:
                    file_path = self.cache_dir / fname
                    try:
                        async with aiofiles.open(file_path, "r") as f:
                            data = await f.read()
                        return json.loads(data)["result"]
                    except Exception:
                        pass
                # expired or corrupted -> remove
                await self._remove_entry(key)
        return None

    async def set(self, key: str, value: str):
        async with self._lock:
            fname = self._key_to_filename(key)
            file_path = self.cache_dir / fname
            data = {"result": value, "timestamp": time.time()}
            try:
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(data))
                self.index[key] = (time.time(), fname)
                await self._evict_if_needed()
            except Exception as e:
                logger.error(f"Cache write failed: {e}")

    async def _remove_entry(self, key: str):
        if key in self.index:
            _, fname = self.index.pop(key)
            try:
                await aiofiles.os.remove(self.cache_dir / fname)
            except Exception:
                pass

    async def _evict_if_needed(self):
        while len(self.index) > self.max_entries:
            # Evict oldest by timestamp
            oldest_key = min(self.index, key=lambda k: self.index[k][0])
            await self._remove_entry(oldest_key)


# ---------------------------------------------------------------------------
# Tool registry (simple external verifiers)
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., Awaitable[str]]] = {}

    def register(self, name: str, func: Callable[..., Awaitable[str]]):
        self._tools[name] = func

    async def execute(self, name: str, **kwargs) -> str:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not registered")
        return await tool(**kwargs)


# ---------------------------------------------------------------------------
# Core Agent
# ---------------------------------------------------------------------------
class AdvancedVerifyingAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.max_rounds = config.max_verification_rounds
        self.temperature = config.temperature
        self.cache = PersistentCache(config.cache_dir, config.cache_ttl_seconds, config.max_cache_entries)
        self.tools = ToolRegistry()

        # Set up clients
        self.client = None
        if self.provider == "openai" and HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set; falling back to mock")
                self.provider = "mock"
            else:
                self.client = AsyncOpenAI(api_key=api_key)
        elif self.provider == "anthropic" and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set; falling back to mock")
                self.provider = "mock"
            else:
                self.client = AsyncAnthropic(api_key=api_key)

        # Register built-in tools if requested
        if "calculator" in self.config.external_tools:
            self.tools.register("calculator", self._calculator_tool)

    async def _calculator_tool(self, expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Very limited eval for demo; use a real sandbox in production
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def _get_cache_key(self, prompt: str, context: str = "llm") -> str:
        return hashlib.md5(f"{prompt}|{context}".encode()).hexdigest()

    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None,
                        use_cache: bool = True, stream: bool = False) -> str:
        """Unified async LLM call with caching and retries."""
        cache_key = self._get_cache_key(prompt, "llm")
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit")
                return cached

        result = await self._call_llm_with_retry(prompt, system_prompt, stream)

        if use_cache:
            await self.cache.set(cache_key, result)
        return result

    async def _call_llm_with_retry(self, prompt: str, system_prompt: Optional[str] = None,
                                   stream: bool = False) -> str:
        """Call LLM with exponential backoff retries."""
        last_exception = None
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return await self._call_llm_no_cache(prompt, system_prompt, stream)
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        raise last_exception

    async def _call_llm_no_cache(self, prompt: str, system_prompt: Optional[str] = None,
                                 stream: bool = False) -> str:
        if self.provider == "mock":
            return await self._mock_llm_response(prompt)

        elif self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            if stream:
                full_text = ""
                stream_resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True
                )
                async for chunk in stream_resp:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_text += content
                        print(content, end="", flush=True)  # live print
                return full_text
            else:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

        elif self.provider == "anthropic":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _mock_llm_response(self, prompt: str) -> str:
        """Extensible mock based on prompt patterns."""
        prompt_lower = prompt.lower()

        # Add more test cases here as needed
        test_cases = [
            ("capital of france", "lyon",
             "ERROR: The capital of France is Paris, not Lyon."),
            ("capital of france", "paris",
             "VERIFIED: Correct."),
            ("15% of 247", "35.05",
             "ERROR: 15% of 247 is 247 * 0.15 = 37.05, not 35.05."),
            ("15% of 247", "37.05",
             "VERIFIED: Correct."),
            ("square root of 144", "10",
             "ERROR: Square root of 144 is 12, not 10."),
            ("square root of 144", "12",
             "VERIFIED: Correct."),
        ]

        for keyword, wrong_answer, response in test_cases:
            if keyword in prompt_lower and wrong_answer in prompt_lower:
                return response

        # Default: assume verified
        return "VERIFIED: Answer seems correct (mock)."

    async def think(self, question: str, stream: bool = False) -> str:
        """Generate initial answer (async)."""
        system = self.config.system_prompt_generate or ""
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"
        if self.provider == "mock":
            # Intentional mistakes for demo
            q = question.lower()
            if "capital of france" in q:
                return "The capital of France is Lyon."
            elif "15% of 247" in question:
                return "15% of $247 is $35.05"
            elif "square root of 144" in q:
                return "The square root of 144 is 10."
        return await self._call_llm(prompt, system_prompt=system, stream=stream)

    async def verify(self, answer: str, question: str) -> Dict[str, Any]:
        """Verify answer using LLM or external tools (async)."""
        feedback_parts = []

        # 1. Chain-of-thought verification
        if self.config.use_chain_of_thought_verification:
            cot_prompt = (
                f"Question: {question}\n"
                f"Proposed answer: {answer}\n\n"
                "Let's verify step by step. Think about the question, compute or reason logically, "
                "then compare with the proposed answer. If there is any mistake, state 'ERROR:' followed by the correct information. "
                "If correct, state 'VERIFIED:'."
            )
        else:
            cot_prompt = (
                f"Question: {question}\n"
                f"Proposed answer: {answer}\n\n"
                "Is this answer completely correct? If there is any error, state 'ERROR:' followed by the correct information. "
                "If correct, state 'VERIFIED:'."
            )

        system_v = self.config.system_prompt_verify or ""
        response = await self._call_llm(cot_prompt, system_prompt=system_v, use_cache=True)
        feedback_parts.append(response)

        # 2. External tools verification (e.g., calculator)
        if "calculator" in self.config.external_tools:
            # Heuristically detect math expressions
            math_pattern = re.search(r'(\d+[\+\-\*/]\d+|\d+%\s*of\s*\d+)', question + " " + answer, re.IGNORECASE)
            if math_pattern:
                expr = math_pattern.group(1)
                # Convert "15% of 247" to "247*0.15"
                if "% of" in expr:
                    parts = expr.split("% of")
                    expr = f"{parts[1].strip()}*{parts[0].strip()}/100"
                tool_result = await self.tools.execute("calculator", expression=expr)
                feedback_parts.append(f"[CALCULATOR]: expression={expr} result={tool_result}")

        combined_feedback = "\n".join(feedback_parts)
        verdict = "ERROR" if "ERROR" in response else "VERIFIED"
        error_desc = ""
        if verdict == "ERROR":
            match = re.search(r"ERROR:\s*(.*)", response, re.IGNORECASE)
            error_desc = match.group(1) if match else response

        return {
            "verdict": verdict,
            "feedback": combined_feedback,
            "extracted_error": error_desc
        }

    async def correct(self, wrong_answer: str, error_feedback: str, question: str) -> str:
        """Generate corrected answer (async)."""
        correction_prompt = (
            f"The original answer was wrong.\n"
            f"Question: {question}\n"
            f"Wrong answer: {wrong_answer}\n"
            f"Error feedback: {error_feedback}\n\n"
            "Please provide the fully corrected answer only (no extra commentary).\n"
            "Corrected answer:"
        )
        system_c = self.config.system_prompt_correct or ""
        return await self._call_llm(correction_prompt, system_prompt=system_c, use_cache=False)

    async def answer_with_verification(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """Main async method: think -> verify -> correct loop."""
        print(f"Question: {question}\n")
        current_answer = await self.think(question, stream=stream)
        if stream:
            print("\n")
        verification_log = []
        was_corrected = False

        for round_idx in range(self.max_rounds):
            logger.info(f"Verification round {round_idx+1}/{self.max_rounds}")
            check = await self.verify(current_answer, question)
            verification_log.append({
                "round": round_idx + 1,
                "answer": current_answer,
                "verdict": check["verdict"],
                "feedback": check["feedback"]
            })

            if check["verdict"] == "VERIFIED":
                logger.info("Answer verified, stopping.")
                break

            was_corrected = True
            print(f"  ❌ Error detected. Correcting...")
            current_answer = await self.correct(current_answer, check["extracted_error"], question)
            if stream:
                print(f"  ✅ Corrected answer: {current_answer}\n")

        return {
            "final_answer": current_answer,
            "verification_log": verification_log,
            "corrected": was_corrected,
            "rounds_used": len(verification_log)
        }


# ---------------------------------------------------------------------------
# Rich CLI
# ---------------------------------------------------------------------------
async def async_main(config: AgentConfig, question: str, stream: bool = False):
    agent = AdvancedVerifyingAgent(config)
    result = await agent.answer_with_verification(question, stream=stream)
    print("\n" + "="*50)
    print("Final answer:", result["final_answer"])
    print("Rounds used:", result["rounds_used"])
    print("Was corrected:", result["corrected"])
    if not stream:
        print("Verification log:")
        for entry in result["verification_log"]:
            print(f"  Round {entry['round']}: {entry['verdict']} - {entry['feedback'][:100]}...")
    return result


def cli_main():
    parser = argparse.ArgumentParser(description="Advanced Self‑Verifying Agent v2.0")
    parser.add_argument("question", help="Question to ask the agent")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"])
    parser.add_argument("--model", help="Model name (default depends on provider)")
    parser.add_argument("--rounds", type=int, default=2, help="Max verification rounds")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cache-dir", default="./.agent_cache")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought verification")
    parser.add_argument("--tools", nargs="*", default=[], choices=["calculator"], help="External tools to enable")
    parser.add_argument("--stream", action="store_true", help="Stream output in real time")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Build config
    config = AgentConfig(
        provider=args.provider,
        model_name=args.model,
        max_verification_rounds=args.rounds,
        temperature=args.temperature,
        cache_ttl_seconds=args.cache_ttl,
        cache_dir=args.cache_dir,
        use_chain_of_thought_verification=not args.no_cot,
        external_tools=args.tools or [],
    )

    asyncio.run(async_main(config, args.question, stream=args.stream))


if __name__ == "__main__":
    cli_main()