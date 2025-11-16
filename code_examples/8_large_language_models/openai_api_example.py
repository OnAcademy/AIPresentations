"""
Large Language Models: Using OpenAI API
ChatGPT, GPT-4, and other OpenAI models
"""

import os
from typing import List, Dict, Optional
import time

# Note: Install with: pip install openai


# ============================================================================
# EXAMPLE 1: BASIC API USAGE
# ============================================================================
def basic_chat_completion():
    """
    Simple chat completion using OpenAI API
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Chat Completion")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        # Initialize client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        # Extract response
        answer = response.choices[0].message.content
        print(f"\nQuestion: What is 2+2?")
        print(f"Answer: {answer}")
        
        # Show usage statistics
        print(f"\nUsage Statistics:")
        print(f"  Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"  Completion Tokens: {response.usage.completion_tokens}")
        print(f"  Total Tokens: {response.usage.total_tokens}")
        
    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")


# ============================================================================
# EXAMPLE 2: SYSTEM PROMPTS & PERSONA
# ============================================================================
def system_prompt_example():
    """
    Using system prompts to set AI behavior/personality
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: System Prompts & Persona")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Different personas with different system prompts
        personas = {
            "Pirate": "You are a pirate speaking in pirate dialect.",
            "Shakespeare": "You speak like William Shakespeare from the 16th century.",
            "Scientist": "You are a brilliant physicist explaining concepts simply.",
            "Comedian": "You are a stand-up comedian, make everything funny."
        }
        
        user_message = "Tell me about artificial intelligence"
        
        for persona_name, system_prompt in personas.items():
            print(f"\n{persona_name}:")
            print("-" * 50)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            print(response.choices[0].message.content)
    
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 3: CONVERSATION WITH MEMORY
# ============================================================================
def conversation_with_memory():
    """
    Maintaining conversation context across multiple turns
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Turn Conversation")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Conversation history
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
        
        # Simulate a conversation
        conversation = [
            "What is machine learning?",
            "Can you give me an example?",
            "How is that different from deep learning?"
        ]
        
        print("Conversation:")
        print("-" * 50)
        
        for user_input in conversation:
            print(f"\nUser: {user_input}")
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            # Extract assistant response
            assistant_response = response.choices[0].message.content
            print(f"Assistant: {assistant_response}")
            
            # Add to history for context
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Small delay to avoid rate limits
            time.sleep(1)
    
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 4: TEMPERATURE & CREATIVITY
# ============================================================================
def temperature_demonstration():
    """
    Show how temperature affects output creativity
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Temperature Effect on Creativity")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        prompt = "Write a one-sentence creative story starter"
        temperatures = [0.0, 0.3, 0.7, 1.0]
        
        print(f"Prompt: {prompt}\n")
        print(f"{'Temperature':<15} {'Output':<55}")
        print("-" * 70)
        
        for temp in temperatures:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=50
            )
            
            output = response.choices[0].message.content.strip()
            temp_desc = {
                0.0: "(Deterministic)",
                0.3: "(Conservative)",
                0.7: "(Balanced)",
                1.0: "(Creative)"
            }
            
            print(f"{temp} {temp_desc[temp]:<10} {output[:50]}")
            time.sleep(1)
    
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 5: ERROR HANDLING & RATE LIMITING
# ============================================================================
class APIClient:
    """
    Robust OpenAI API client with error handling and rate limiting
    """
    
    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        """Initialize client with error handling"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            self.max_retries = max_retries
        except ImportError:
            raise ImportError("Install OpenAI library: pip install openai")
    
    def chat(self, messages: List[Dict], model: str = "gpt-3.5-turbo", 
             temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        Send chat message with retry logic
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count"""
        # Rough approximation: ~4 chars per token
        return len(text) // 4


def robust_api_example():
    """
    Demonstrate robust error handling
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Robust API Client with Error Handling")
    print("=" * 70)
    
    try:
        client = APIClient()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain AI in one sentence."}
        ]
        
        print("\nSending request with error handling...")
        response = client.chat(messages, temperature=0.5, max_tokens=100)
        print(f"Response: {response}")
        
        # Check token usage
        token_estimate = client.get_token_count(response)
        print(f"Estimated tokens in response: {token_estimate}")
    
    except Exception as e:
        print(f"Error: {e}")


# ============================================================================
# EXAMPLE 6: COST CALCULATION
# ============================================================================
def calculate_api_costs():
    """
    Calculate API costs for different models
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: API Cost Analysis")
    print("=" * 70)
    
    # Pricing as of 2024 (prices change, check OpenAI for current rates)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # per 1K tokens
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    print("\nEstimated cost for 1M input + 1M output tokens:\n")
    print(f"{'Model':<20} {'Input Cost':<15} {'Output Cost':<15} {'Total'}")
    print("-" * 65)
    
    for model, rates in pricing.items():
        input_cost = (1_000_000 / 1_000) * rates["input"]
        output_cost = (1_000_000 / 1_000) * rates["output"]
        total = input_cost + output_cost
        
        print(f"{model:<20} ${input_cost:<14.2f} ${output_cost:<14.2f} ${total:.2f}")


# ============================================================================
# EXAMPLE 7: BEST PRACTICES
# ============================================================================
def best_practices():
    """
    Demonstrate LLM API best practices
    """
    print("\n" + "=" * 70)
    print("BEST PRACTICES FOR LLM APIs")
    print("=" * 70)
    
    practices = {
        "1. Use System Prompts": 
            "Set clear instructions and context for the model",
        
        "2. Temperature Settings":
            "0.0-0.3 for factual tasks, 0.7-1.0 for creative tasks",
        
        "3. Token Management":
            "Monitor tokens to control costs. Use gpt-3.5-turbo when possible",
        
        "4. Error Handling":
            "Implement retry logic with exponential backoff",
        
        "5. Rate Limiting":
            "Respect API rate limits, implement queuing for high volume",
        
        "6. Caching":
            "Cache responses to avoid duplicate API calls",
        
        "7. Input Validation":
            "Validate and sanitize user inputs before sending to API",
        
        "8. Streaming":
            "Use streaming for better user experience on long responses",
    }
    
    for practice, description in practices.items():
        print(f"\n{practice}")
        print(f"  → {description}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM API EXAMPLES - OpenAI")
    print("=" * 70)
    
    print("\nNote: These examples require OPENAI_API_KEY environment variable")
    print("To use these examples:")
    print("1. Install: pip install openai")
    print("2. Set API key: export OPENAI_API_KEY='your-key-here'")
    print("3. Run the examples\n")
    
    # Uncomment to run (requires API key):
    # basic_chat_completion()
    # system_prompt_example()
    # conversation_with_memory()
    # temperature_demonstration()
    # robust_api_example()
    
    calculate_api_costs()
    best_practices()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. System prompts are crucial for directing model behavior
2. Temperature controls creativity (0=deterministic, 1+=creative)
3. Maintain message history for multi-turn conversations
4. Always implement proper error handling and rate limiting
5. Monitor token usage to manage costs
6. Cache responses when possible to reduce API calls
7. Use gpt-3.5-turbo for most tasks (faster and cheaper)
8. Use gpt-4 for complex reasoning and accuracy
    """)

