"""
Advanced Verifying Agent v2.0
Integrates configuration, metrics, tools, and all advanced features.
"""

import asyncio
import os
import re
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from functools import wraps

# Import our modules
from config import AgentConfig, default_config
from metrics import MetricsCollector, metrics
from tools import ToolRegistry, tools, ToolResult

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timed(func):
    """Decorator to time method execution"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start = time.time()
        try:
            result = await func(self, *args, **kwargs)
            duration = time.time() - start
            metrics.record_llm_call(
                provider=self.config.model.provider,
                model=self.config.model.model_name,
                duration=duration,
                tokens_used=0  # We'll update this if we track tokens
            )
            return result
        except Exception as e:
            metrics.record_error(
                error_type=func.__name__,
                error_msg=str(e)
            )
            raise
    return wrapper


class PersistentCache:
    """Enhanced cache with disk persistence and TTL"""
    
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = config.ttl_seconds
        self.max_entries = config.max_entries
        self.index: Dict[str, tuple[float, str]] = {}
        self._lock = asyncio.Lock()
        self._load_index()
    
    def _load_index(self):
        """Load index from disk"""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    self.index = {
                        k: (v["timestamp"], v["filename"])
                        for k, v in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
    
    def _save_index(self):
        """Save index to disk"""
        index_file = self.cache_dir / "index.json"
        data = {
            k: {"timestamp": ts, "filename": fname}
            for k, (ts, fname) in self.index.items()
        }
        try:
            with open(index_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _key_to_filename(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest() + ".json"
    
    async def get(self, key: str) -> Optional[str]:
        if not self.config.enabled:
            return None
        
        async with self._lock:
            entry = self.index.get(key)
            if entry:
                ts, fname = entry
                if time.time() - ts <= self.ttl:
                    file_path = self.cache_dir / fname
                    if file_path.exists():
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            return data["result"]
                        except Exception:
                            pass
                # Expired or corrupted
                await self._remove_entry(key)
        return None
    
    async def set(self, key: str, value: str):
        if not self.config.enabled:
            return
        
        async with self._lock:
            fname = self._key_to_filename(key)
            file_path = self.cache_dir / fname
            data = {"result": value, "timestamp": time.time()}
            try:
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                self.index[key] = (time.time(), fname)
                self._save_index()
                await self._evict_if_needed()
            except Exception as e:
                logger.error(f"Cache write failed: {e}")
    
    async def _remove_entry(self, key: str):
        if key in self.index:
            _, fname = self.index.pop(key)
            try:
                (self.cache_dir / fname).unlink(missing_ok=True)
                self._save_index()
            except Exception:
                pass
    
    async def _evict_if_needed(self):
        while len(self.index) > self.max_entries:
            oldest_key = min(self.index, key=lambda k: self.index[k][0])
            await self._remove_entry(oldest_key)
    
    async def clear(self):
        """Clear entire cache"""
        async with self._lock:
            for fname in self.cache_dir.glob("*.json"):
                fname.unlink()
            self.index.clear()
            self._save_index()


class AdvancedVerifyingAgentV2:
    """
    Enhanced version of the self-verifying agent with:
    - Async support
    - Persistent caching
    - Chain-of-thought verification
    - External tools
    - Metrics collection
    - Config management
    - Error handling
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or default_config
        self.cache = PersistentCache(self.config.cache)
        self.tools = tools
        
        # Initialize LLM client
        self.client = None
        if self.config.model.provider == "openai" and HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set; falling back to mock")
                self.config.model.provider = "mock"
            else:
                self.client = AsyncOpenAI(api_key=api_key)
        elif self.config.model.provider == "anthropic" and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set; falling back to mock")
                self.config.model.provider = "mock"
            else:
                self.client = AsyncAnthropic(api_key=api_key)
    
    def _get_cache_key(self, prompt: str, context: str = "llm") -> str:
        return hashlib.md5(f"{prompt}|{context}".encode()).hexdigest()
    
    @timed
    async def _call_llm(self, prompt: str, 
                       system_prompt: Optional[str] = None,
                       use_cache: bool = True) -> str:
        """Unified LLM call with caching and retries"""
        cache_key = self._get_cache_key(prompt)
        
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit")
                return cached
        
        # Retry logic
        for attempt in range(3):
            try:
                result = await self._call_llm_no_cache(prompt, system_prompt)
                if use_cache:
                    await self.cache.set(cache_key, result)
                return result
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    
    async def _call_llm_no_cache(self, prompt: str, 
                                 system_prompt: Optional[str] = None) -> str:
        """Actual LLM call without caching"""
        if self.config.model.provider == "mock":
            return await self._mock_llm_response(prompt)
        
        elif self.config.model.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.config.model.model_name,
                messages=messages,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.config.model.provider == "anthropic":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = await self.client.messages.create(
                model=self.config.model.model_name,
                max_tokens=self.config.model.max_tokens,
                temperature=self.config.model.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unknown provider: {self.config.model.provider}")
    
    async def _mock_llm_response(self, prompt: str) -> str:
        """Enhanced mock with more test cases"""
        prompt_lower = prompt.lower()
        
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
            ("2 + 2", "3",
             "ERROR: 2 + 2 = 4, not 3."),
            ("2 + 2", "4",
             "VERIFIED: Correct."),
        ]
        
        for keyword, wrong_answer, response in test_cases:
            if keyword in prompt_lower and wrong_answer in prompt_lower:
                return response
        
        # Use calculator tool if available
        if self.config.verification.enable_external_tools:
            for tool_name in self.config.verification.tools:
                if tool_name == "calculator":
                    # Try to extract expression
                    match = re.search(r'(\d+[\+\-\*/]\s*\d+)', prompt)
                    if match:
                        expr = match.group(1)
                        result = await self.tools.execute("calculator", expression=expr)
                        if result.success:
                            return f"VERIFIED: {expr} = {result.result}"
        
        return "VERIFIED: Answer seems correct (mock)."
    
    async def think(self, question: str) -> str:
        """Generate initial answer"""
        system = self.config.system_prompt_generate or ""
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"
        
        if self.config.model.provider == "mock":
            # Intentional mistakes for demo
            q = question.lower()
            if "capital of france" in q:
                return "The capital of France is Lyon."
            elif "15% of 247" in question:
                return "15% of $247 is $35.05"
            elif "square root of 144" in q:
                return "The square root of 144 is 10."
        
        return await self._call_llm(prompt, system_prompt=system)
    
    async def verify(self, answer: str, question: str) -> Dict[str, Any]:
        """Verify answer using LLM and tools"""
        feedback_parts = []
        
        # Chain-of-thought verification
        if self.config.verification.use_chain_of_thought:
            prompt = (
                f"Question: {question}\n"
                f"Proposed answer: {answer}\n\n"
                "Let's verify step by step. Think about the question, reason logically, "
                "then compare with the proposed answer. If there is any mistake, "
                "state 'ERROR:' followed by the correct information. "
                "If correct, state 'VERIFIED:'."
            )
        else:
            prompt = (
                f"Question: {question}\n"
                f"Proposed answer: {answer}\n\n"
                "Is this answer completely correct? If there is any error, "
                "state 'ERROR:' followed by the correct information. "
                "If correct, state 'VERIFIED:'."
            )
        
        system_v = self.config.system_prompt_verify or ""
        response = await self._call_llm(prompt, system_prompt=system_v, use_cache=True)
        feedback_parts.append(response)
        
        # External tools verification
        if self.config.verification.enable_external_tools:
            for tool_name in self.config.verification.tools:
                if tool_name == "calculator":
                    math_pattern = re.search(
                        r'(\d+[\+\-\*/]\s*\d+|\d+%\s*of\s*\d+)', 
                        question + " " + answer, 
                        re.IGNORECASE
                    )
                    if math_pattern:
                        expr = math_pattern.group(1)
                        if "% of" in expr:
                            parts = expr.split("% of")
                            expr = f"({parts[1].strip()}) * ({parts[0].strip()}) / 100"
                        tool_result = await self.tools.execute("calculator", expression=expr)
                        if tool_result.success:
                            feedback_parts.append(
                                f"[CALCULATOR]: {expr} = {tool_result.result}"
                            )
        
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
        """Generate corrected answer"""
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
    
    async def answer_with_verification(self, question: str) -> Dict[str, Any]:
        """Main method: think -> verify -> correct loop"""
        start_time = time.time()
        
        print(f"🤔 Question: {question}\n")
        current_answer = await self.think(question)
        
        verification_log = []
        was_corrected = False
        
        for round_idx in range(self.config.verification.max_rounds):
            logger.info(f"Verification round {round_idx+1}/{self.config.verification.max_rounds}")
            
            check = await self.verify(current_answer, question)
            verification_log.append({
                "round": round_idx + 1,
                "answer": current_answer,
                "verdict": check["verdict"],
                "feedback": check["feedback"]
            })
            
            if check["verdict"] == "VERIFIED":
                logger.info("✅ Answer verified, stopping.")
                break
            
            was_corrected = True
            print(f"  ❌ Error detected. Correcting...")
            current_answer = await self.correct(current_answer, check["extracted_error"], question)
            print(f"  ✅ Corrected to: {current_answer}\n")
        
        # Record metrics
        metrics.record_verification(
            question=question,
            rounds=len(verification_log),
            corrected=was_corrected,
            final_verdict=check.get("verdict", "ERROR")
        )
        
        return {
            "final_answer": current_answer,
            "verification_log": verification_log,
            "corrected": was_corrected,
            "rounds_used": len(verification_log),
            "duration": time.time() - start_time
        }
    
    async def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions in parallel"""
        tasks = [self.answer_with_verification(q) for q in questions]
        return await asyncio.gather(*tasks)


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Verifying Agent v2.0")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock")
    parser.add_argument("--rounds", type=int, default=2, help="Max verification rounds")
    parser.add_argument("--tools", nargs="*", default=[], help="Tools to enable (calculator, web_search)")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--metrics", action="store_true", help="Export metrics after run")
    parser.add_argument("--batch", help="JSON file with list of questions")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = AgentConfig.from_yaml(args.config)
    else:
        config = AgentConfig()
    
    # Override from CLI
    if args.provider:
        config.model.provider = args.provider
    if args.rounds:
        config.verification.max_rounds = args.rounds
    if args.tools:
        config.verification.tools = args.tools
    if args.no_cot:
        config.verification.use_chain_of_thought = False
    if args.no_cache:
        config.cache.enabled = False
    
    # Initialize agent
    agent = AdvancedVerifyingAgentV2(config)
    
    # Process batch
    if args.batch:
        with open(args.batch, 'r') as f:
            questions = json.load(f)
        results = await agent.batch_answer(questions)
        print(json.dumps(results, indent=2))
        return
    
    # Single question
    if not args.question:
        parser.print_help()
        return
    
    result = await agent.answer_with_verification(args.question)
    
    print("\n" + "="*60)
    print(f"✅ Final Answer: {result['final_answer']}")
    print(f"📊 Rounds used: {result['rounds_used']}")
    print(f"🔄 Corrected: {result['corrected']}")
    print(f"⏱️  Duration: {result['duration']:.2f}s")
    print("="*60)
    
    if args.metrics:
        metrics.export_to_json("metrics_export.json")
        print("📈 Metrics exported to metrics_export.json")


if __name__ == "__main__":
    asyncio.run(main())