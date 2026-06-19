"""
Interactive demo for Advanced Verifying Agent
Run this to have a conversation with the agent!
"""

import asyncio
import sys
from advanced_agent_v2 import AdvancedVerifyingAgentV2
from config import AgentConfig

async def interactive_mode():
    print("=" * 60)
    print("🤖 Advanced Verifying Agent - Interactive Mode")
    print("=" * 60)
    print("\nType 'exit' or 'quit' to stop\n")
    
    # Configure agent
    config = AgentConfig()
    config.model.provider = "mock"  # Use mock for demo
    config.verification.enable_external_tools = True
    config.verification.tools = ["calculator"]
    config.verification.max_rounds = 3
    
    agent = AdvancedVerifyingAgentV2(config)
    
    question_num = 1
    
    while True:
        try:
            question = input(f"\n❓ Question {question_num}: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n⏳ Thinking...")
            
            result = await agent.answer_with_verification(question)
            
            print(f"\n✅ Answer: {result['final_answer']}")
            print(f"🔄 Corrected: {'Yes' if result['corrected'] else 'No'}")
            print(f"📊 Rounds: {result['rounds_used']}")
            print(f"⏱️  Time: {result['duration']:.2f}s")
            print("-" * 60)
            
            if result["verification_log"]:
                print("\n📝 Verification Log:")
                for entry in result["verification_log"]:
                    status = "✅" if entry["verdict"] == "VERIFIED" else "❌"
                    print(f"  Round {entry['round']}: {status} {entry['verdict']}")
            
            question_num += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try another question.")

if __name__ == "__main__":
    asyncio.run(interactive_mode())