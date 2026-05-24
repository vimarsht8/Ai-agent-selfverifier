class MyFirstVerifyingAgent:
    def think(self, question):
        # Simulate an initial (maybe wrong) answer
        # In real life, this calls GPT/Claude
        if "capital of France" in question.lower():
            return "The capital of France is Lyon."  # Wrong on purpose!
        elif "15% of 247" in question:
            return "15% of $247 is $35.05"  # Should be 37.05
        else:
            return f"I think: {question} is complicated."
    
    def ask_llm(self, prompt):
        # Simulate a verification LLM (in reality: OpenAI API call)
        # This mock catches errors
        if "Lyon" in prompt and "capital" in prompt:
            return "ERROR: The capital of France is Paris, not Lyon."
        elif "35.05" in prompt and "15%" in prompt:
            return "ERROR: 15% of 247 is 247 * 0.15 = 37.05, not 35.05."
        else:
            return "VERIFIED: Answer seems correct."
    
    def verify(self, answer, question):
        # Use the mock LLM to check
        return self.ask_llm(f"Question: {question}\nAnswer: {answer}\nIs this correct? List any errors.")
    
    def correct(self, wrong_answer, error_feedback):
        # Simple correction logic (real version would call an LLM again)
        if "Lyon" in wrong_answer:
            return "The capital of France is Paris."
        elif "35.05" in wrong_answer:
            return "15% of $247 is $37.05"
        else:
            return wrong_answer + " [corrected based on: " + error_feedback + "]"
    
    def answer_with_verification(self, question):
        answer_1 = self.think(question)
        check_result = self.verify(answer_1, question)
        
        if "ERROR" in check_result:
            answer_2 = self.correct(answer_1, check_result)
        else:
            answer_2 = answer_1
        
        return {
            "final_answer": answer_2,
            "verification_log": check_result,
            "corrected": answer_1 != answer_2
        }

# ----- NOW ACTUALLY RUN IT -----
agent = MyFirstVerifyingAgent()
result = agent.answer_with_verification("What is the capital of France?")

print("=== OUTPUT ===")
print("Final answer:", result["final_answer"])
print("Verification log:", result["verification_log"])
print("Was corrected?", result["corrected"])