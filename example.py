from aiagentverifier import AdvancedVerifyingAgent

agent = AdvancedVerifyingAgent(model_provider="mock")
result = agent.answer_with_verification("What is the capital of France?")
print(result["final_answer"])  # "The capital of France is Paris."