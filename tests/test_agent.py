import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiagentverifier import AdvancedVerifyingAgent

def test_mock_capital():
    agent = AdvancedVerifyingAgent(model_provider="mock", max_verification_rounds=2)
    result = agent.answer_with_verification("What is the capital of France?")
    assert "Paris" in result["final_answer"]
    assert result["corrected"] == True

def test_mock_percentage():
    agent = AdvancedVerifyingAgent(model_provider="mock")
    result = agent.answer_with_verification("What is 15% of 247?")
    assert "37.05" in result["final_answer"]

def test_mock_sqrt():
    agent = AdvancedVerifyingAgent(model_provider="mock")
    result = agent.answer_with_verification("What is the square root of 144?")
    assert "12" in result["final_answer"]

if __name__ == "__main__":
    test_mock_capital()
    test_mock_percentage()
    test_mock_sqrt()
    print("All tests passed!")