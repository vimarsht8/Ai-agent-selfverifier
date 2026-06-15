import sys
import os
import asyncio

# If your project structure places the agent code in a sibling directory,
# adjust the import path as needed. Example:
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new agent and its configuration model
from advanced_agent import AdvancedVerifyingAgent, AgentConfig

# -------------------------------------------------------------------
# Async test runners
# -------------------------------------------------------------------

async def _run_test(question: str, expected_substring: str, expect_correction: bool = None):
    """Helper to instantiate agent, run verification, and assert."""
    # Build a config for a mock agent with 2 verification rounds
    config = AgentConfig(
        provider="mock",
        max_verification_rounds=2,
        temperature=0.2,
        # Other fields will use defaults; we keep it simple
    )
    agent = AdvancedVerifyingAgent(config)

    result = await agent.answer_with_verification(question)

    # Check that the final answer contains the expected substring
    final = result["final_answer"]
    assert expected_substring.lower() in final.lower(), \
        f"Expected '{expected_substring}' in final answer, got: {final}"

    # If we know whether correction should have occurred, check it
    if expect_correction is not None:
        assert result["corrected"] == expect_correction, \
            f"Expected corrected={expect_correction}, got {result['corrected']}"

    print(f"✓ {question}: {final}")
    return result


async def test_mock_capital():
    await _run_test(
        question="What is the capital of France?",
        expected_substring="Paris",
        expect_correction=True   # initial answer is Lyon -> corrected
    )


async def test_mock_percentage():
    await _run_test(
        question="What is 15% of 247?",
        expected_substring="37.05",
        expect_correction=True   # initial answer is 35.05 -> corrected
    )


async def test_mock_sqrt():
    await _run_test(
        question="What is the square root of 144?",
        expected_substring="12",
        expect_correction=True   # initial answer is 10 -> corrected
    )


async def main():
    """Run all tests sequentially (or concurrently if desired)."""
    await test_mock_capital()
    await test_mock_percentage()
    await test_mock_sqrt()
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Run the async test suite
    asyncio.run(main())