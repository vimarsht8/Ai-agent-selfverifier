import os
import re
import logging
from typing import Dict, Any, Optional

# Try importing real LLM libraries (optional)
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
    An LLM‑based agent that generates an answer, verifies it with a second LLM,
    and corrects iteratively until the answer passes verification or max_rounds is reached.
    """

    def __init__(self, 
                 model_provider: str = "mock",   # "openai", "anthropic", "mock"
                 model_name: Optional[str] = None,
                 max_verification_rounds: int = 2,
                 temperature: float = 0.2):
        """
        Args:
            model_provider: "openai", "anthropic", or "mock"
            model_name: e.g. "gpt-4o-mini", "claude-3-haiku-20240307"
            max_verification_rounds: number of correction attempts allowed
            temperature: sampling temperature for LLM calls
        """
        self.provider = model_provider
        self.max_rounds = max_verification_rounds
        self.temperature = temperature

        # Set default models
        if model_name:
            self.model_name = model_name
        else:
            if self.provider == "openai":
                self.model_name = "gpt-3.5-turbo"
            elif self.provider == "anthropic":
                self.model_name = "claude-3-haiku-20240307"
            else:
                self.model_name = "mock-llm"

        # Initialize clients if real provider requested and available
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

    # ---------- Core LLM call ----------
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Unified method to call the chosen LLM or mock."""
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
        """Simulates an LLM that can detect the known errors."""
        prompt_lower = prompt.lower()
        # Capital of France errors
        if "capital of france" in prompt_lower and "lyon" in prompt_lower:
            return "ERROR: The capital of France is Paris, not Lyon."
        if "capital of france" in prompt_lower and "paris" in prompt_lower:
            return "VERIFIED: Correct."
        # Percentage errors
        if "15% of 247" in prompt and "35.05" in prompt:
            return "ERROR: 15% of 247 is 247 * 0.15 = 37.05, not 35.05."
        if "15% of 247" in prompt and "37.05" in prompt:
            return "VERIFIED: Correct."
        # Generic fallback
        return "VERIFIED: Answer seems correct (mock)."

    # ---------- Agent methods ----------
    def think(self, question: str) -> str:
        """Generate an initial answer (may be wrong intentionally for testing)."""
        prompt = f"Answer the following question concisely:\n\nQuestion: {question}\nAnswer:"
        # For the mock we keep the intentional mistakes to match original demo
        if self.provider == "mock":
            if "capital of France" in question.lower():
                return "The capital of France is Lyon."  # wrong
            elif "15% of 247" in question:
                return "15% of $247 is $35.05"          # wrong
        # For real LLMs we ask normally (no intentional mistake)
        return self._call_llm(prompt)

    def verify(self, answer: str, question: str) -> Dict[str, Any]:
        """
        Ask the LLM to judge correctness.
        Returns a dict with 'verdict' ('ERROR' or 'VERIFIED'), 'feedback' (full message),
        and optionally 'extracted_error' for correction.
        """
        verify_prompt = f"""You are a strict verifier.  
Question: {question}  
Proposed Answer: {answer}  

Is this answer completely correct?  
If there is any error, state "ERROR:" followed by the correct information.  
If correct, state "VERIFIED:". Be specific.  
"""
        response = self._call_llm(verify_prompt)
        verdict = "ERROR" if "ERROR" in response else "VERIFIED"
        # Simple extraction of error description (everything after "ERROR:")
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
        """Generate a corrected answer given the error feedback."""
        correction_prompt = f"""The original answer was wrong.  
Question: {question}  
Wrong answer: {wrong_answer}  
Error feedback: {error_feedback}  

Please provide the fully corrected answer only (no extra commentary).  
Corrected answer:"""
        return self._call_llm(correction_prompt)

    def answer_with_verification(self, question: str) -> Dict[str, Any]:
        """
        Main public method: think → verify → correct (repeat until max_rounds or verified).
        Returns final answer, full log, and metadata.
        """
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

            # Otherwise correct and continue
            was_corrected = True
            current_answer = self.correct(current_answer, check["extracted_error"], question)
            logger.info(f"Corrected to: {current_answer[:100]}...")

        return {
            "final_answer": current_answer,
            "verification_log": verification_log,
            "corrected": was_corrected,
            "rounds_used": len(verification_log)
        }


# ========== DEMONSTRATION ==========
if __name__ == "__main__":
    # Using mock mode (no API key needed) – reproduces the original demo behaviour
    agent = AdvancedVerifyingAgent(model_provider="mock", max_verification_rounds=2)

    print("=== Test 1: Capital of France ===")
    result = agent.answer_with_verification("What is the capital of France?")
    print(f"Final answer: {result['final_answer']}")
    print(f"Corrected? {result['corrected']}")
    print(f"Verification rounds: {result['rounds_used']}")
    print(f"Verification log (round 1): {result['verification_log'][0]['feedback']}\n")

    print("=== Test 2: 15% of 247 ===")
    result2 = agent.answer_with_verification("What is 15% of 247?")
    print(f"Final answer: {result2['final_answer']}")
    print(f"Corrected? {result2['corrected']}")
    print(f"Verification rounds: {result2['rounds_used']}")