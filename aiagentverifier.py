import os
import re
import json
import hashlib
import logging
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Try importing real LLM libraries
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedVerifyingAgent:
    """
    LLM-based agent with verification, correction, caching, and CLI support.
    """

    def __init__(self,
                 model_provider: str = "mock",
                 model_name: Optional[str] = None,
                 max_verification_rounds: int = 2,
                 temperature: float = 0.2,
                 cache_ttl_seconds: int = 3600,
                 verification_prompt_template: Optional[str] = None):
        self.provider = model_provider
        self.max_rounds = max_verification_rounds
        self.temperature = temperature
        self.cache_ttl = cache_ttl_seconds
        self._cache = {}  # {hash: (timestamp, result)}

        # Optional custom verification prompt
        self.verification_prompt_template = verification_prompt_template or \
            "You are a strict verifier.\nQuestion: {question}\nProposed Answer: {answer}\n\nIs this answer completely correct?\nIf there is any error, state 'ERROR:' followed by the correct information.\nIf correct, state 'VERIFIED:'. Be specific."

        # Set default model names
        if model_name:
            self.model_name = model_name
        else:
            if self.provider == "openai":
                self.model_name = "gpt-3.5-turbo"
            elif self.provider == "anthropic":
                self.model_name = "claude-3-haiku-20240307"
            else:
                self.model_name = "mock-llm"

        # Initialize clients
        self.client = None
        if self.provider == "openai" and HAS_OPENAI:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OPENAI_API_KEY not set; falling back to mock")
                self.provider = "mock"
            else:
                self.client = openai.OpenAI()
        elif self.provider == "anthropic" and HAS_ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set; falling back to mock")
                self.provider = "mock"
            else:
                self.client = Anthropic(api_key=api_key)
        elif self.provider not in ["mock", "openai", "anthropic"]:
            logger.warning(f"Provider {self.provider} unknown; using mock")
            self.provider = "mock"

    def _get_cache_key(self, question: str, answer: str, context: str = "verify") -> str:
        """Generate a cache key for a Q&A pair."""
        data = f"{question}|{answer}|{context}"
        return hashlib.md5(data.encode()).hexdigest()

    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cache entry exists and is still fresh."""
        if cache_key not in self._cache:
            return False
        timestamp, _ = self._cache[cache_key]
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, use_cache: bool = True) -> str:
        """Unified LLM call with caching."""
        cache_key = self._get_cache_key(prompt, "", context="llm")
        if use_cache and self._is_cached_valid(cache_key):
            logger.debug("Cache hit for LLM call")
            return self._cache[cache_key][1]

        result = self._call_llm_no_cache(prompt, system_prompt)

        if use_cache:
            self._cache[cache_key] = (datetime.now(), result)
            # Simple cache cleanup
            if len(self._cache) > 100:
                oldest = min(self._cache.items(), key=lambda x: x[1][0])
                del self._cache[oldest[0]]

        return result

    def _call_llm_no_cache(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Actual LLM call without caching."""
        if self.provider == "mock":
            return self._mock_llm_response(prompt)

        elif self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _mock_llm_response(self, prompt: str) -> str:
        """Enhanced mock that reliably catches errors."""
        prompt_lower = prompt.lower()
        # Capital of France - wrong
        if "capital of france" in prompt_lower and "lyon" in prompt_lower:
            return "ERROR: The capital of France is Paris, not Lyon."
        # Capital of France - correct
        if "capital of france" in prompt_lower and "paris" in prompt_lower:
            return "VERIFIED: Correct."
        # Percentage wrong
        if "15% of 247" in prompt and "35.05" in prompt:
            return "ERROR: 15% of 247 is 247 * 0.15 = 37.05, not 35.05."
        # Percentage correct
        if "15% of 247" in prompt and "37.05" in prompt:
            return "VERIFIED: Correct."
        # Add more test cases
        if "square root of 144" in prompt_lower and "10" in prompt_lower:
            return "ERROR: Square root of 144 is 12, not 10."
        if "square root of 144" in prompt_lower and "12" in prompt_lower:
            return "VERIFIED: Correct."
        # Generic fallback
        return "VERIFIED: Answer seems correct (mock)."

    def think(self, question: str) -> str:
        """Generate initial answer."""
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"
        if self.provider == "mock":
            # Keep intentional mistakes for demo
            q = question.lower()
            if "capital of france" in q:
                return "The capital of France is Lyon."
            elif "15% of 247" in question:
                return "15% of $247 is $35.05"
            elif "square root of 144" in q:
                return "The square root of 144 is 10."
        return self._call_llm(prompt)

    def verify(self, answer: str, question: str) -> Dict[str, Any]:
        """Verify answer using LLM."""
        verify_prompt = self.verification_prompt_template.format(question=question, answer=answer)
        response = self._call_llm(verify_prompt, use_cache=True)

        verdict = "ERROR" if "ERROR" in response else "VERIFIED"
        error_desc = ""
        if verdict == "ERROR":
            match = re.search(r"ERROR:\s*(.*)", response, re.IGNORECASE)
            error_desc = match.group(1) if match else response

        return {
            "verdict": verdict,
            "feedback": response,
            "extracted_error": error_desc
        }

    def correct(self, wrong_answer: str, error_feedback: str, question: str) -> str:
        """Generate corrected answer."""
        correction_prompt = f"""The original answer was wrong.
Question: {question}
Wrong answer: {wrong_answer}
Error feedback: {error_feedback}

Please provide the fully corrected answer only (no extra commentary).
Corrected answer:"""
        return self._call_llm(correction_prompt, use_cache=False)  # don't cache corrections

    def answer_with_verification(self, question: str) -> Dict[str, Any]:
        """Main public method."""
        current_answer = self.think(question)
        verification_log = []
        was_corrected = False

        for round_idx in range(self.max_rounds):
            logger.info(f"Verification round {round_idx+1}/{self.max_rounds}")
            check = self.verify(current_answer, question)
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
            current_answer = self.correct(current_answer, check["extracted_error"], question)
            logger.info(f"Corrected to: {current_answer[:100]}...")

        return {
            "final_answer": current_answer,
            "verification_log": verification_log,
            "corrected": was_corrected,
            "rounds_used": len(verification_log)
        }


def cli_main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Advanced Verifying Agent")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic"],
                        help="LLM provider")
    parser.add_argument("--rounds", type=int, default=2, help="Max verification rounds")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    args = parser.parse_args()

    agent = AdvancedVerifyingAgent(
        model_provider=args.provider,
        max_verification_rounds=args.rounds,
        temperature=args.temperature
    )
    result = agent.answer_with_verification(args.question)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    # If arguments provided, use CLI; else run demos
    import sys
    if len(sys.argv) > 1:
        cli_main()
    else:
        # Demo
        agent = AdvancedVerifyingAgent(model_provider="mock", max_verification_rounds=2)

        print("=== Test 1: Capital of France ===")
        result = agent.answer_with_verification("What is the capital of France?")
        print(f"Final answer: {result['final_answer']}")
        print(f"Corrected? {result['corrected']}")
        print(f"Rounds used: {result['rounds_used']}\n")

        print("=== Test 2: 15% of 247 ===")
        result2 = agent.answer_with_verification("What is 15% of 247?")
        print(f"Final answer: {result2['final_answer']}")
        print(f"Corrected? {result2['corrected']}")
        print(f"Rounds used: {result2['rounds_used']}\n")

        print("=== Test 3: Square root of 144 ===")
        result3 = agent.answer_with_verification("What is the square root of 144?")
        print(f"Final answer: {result3['final_answer']}")
        print(f"Corrected? {result3['corrected']}")
        print(f"Rounds used: {result3['rounds_used']}")