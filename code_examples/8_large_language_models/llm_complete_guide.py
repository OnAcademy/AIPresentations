"""
Large Language Models (LLMs): Complete Implementation Guide
Understanding, using, and fine-tuning LLMs
Covers: LLM concepts, APIs, prompting, in-context learning, applications
"""

import os
from typing import List, Dict, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXAMPLE 1: LLM FUNDAMENTALS
# ============================================================================
def explain_llm_fundamentals():
    """
    Explain what LLMs are and how they work
    """
    print("=" * 80)
    print("LARGE LANGUAGE MODELS (LLMs) - FUNDAMENTALS")
    print("=" * 80)
    
    explanation = """
WHAT IS A LARGE LANGUAGE MODEL?

Definition: A neural network trained on massive text to predict next tokens

Key Characteristics:
• Billions to trillions of parameters
• Trained on internet-scale text (petabytes)
• Task: Next-token prediction (language modeling)
• Architecture: Transformer-based

HOW LLMs WORK
━━━━━━━━━━━

1. TOKENIZATION
   Text: "Hello world"
   ↓ (convert to token IDs)
   Tokens: [1234, 5678]

2. EMBEDDING
   Tokens: [1234, 5678]
   ↓ (convert to dense vectors)
   Embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...]]

3. TRANSFORMER LAYERS
   Apply self-attention multiple times
   Each layer refines understanding
   Typical: 12-96 layers

4. OUTPUT PROJECTION
   Final hidden state → Vector space of vocabulary
   ↓ (softmax)
   Probability distribution over next tokens

5. SAMPLING/DECODING
   Sample from distribution or pick highest probability
   Next token predicted
   Repeat process

PREDICTION LOOP
━━━━━━━━━━━━━

Input: "The capital of France is"
Step 1: Predict next token → "Paris" (90% confidence)
Step 2: Predict next token → "." (80% confidence)
Step 3: Done (end-of-sentence token)

PARAMETERS = KNOWLEDGE
━━━━━━━━━━━━━━━━━━━

Weights learned during training store:
• Language patterns (grammar, syntax)
• Factual knowledge (facts encoded in parameters)
• Common sense reasoning
• Skills (math, coding, etc.)

Model Sizes:
• Small: 100M-1B (Mobile, edge)
• Medium: 7B-13B (Consumer GPUs)
• Large: 70B-175B (Enterprise GPUs)
• Huge: 1T+ (SOTA models, proprietary)

Memory Requirements:
• 7B model: ~14GB (float32), ~7GB (float16), ~4GB (int8)
• 70B model: ~140GB (float32), ~70GB (float16), ~40GB (int8)
• Quantization: Major way to reduce memory

CONTEXT WINDOW
━━━━━━━━━━━━

Maximum tokens model can process at once:
• Original BERT: 512 tokens (~2000 words)
• GPT-3.5: 4096 tokens (~15000 words)
• GPT-4: 8192 or 128k tokens (pay for context)
• Claude 2: 100k tokens (~400,000 words!)

Importance: Larger context = can work with longer documents

TRAINING DATA & KNOWLEDGE CUTOFF
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training data has cutoff date:
• GPT-3.5: September 2021
• GPT-4: April 2023
• Claude 2: Early 2023

Knowledge cutoff means:
• No awareness of recent events
• Can't browse internet
• May have outdated information

Tools to address:
• RAG (Retrieval-Augmented Generation): Fetch recent info
• Plugins/APIs: Real-time data
• Function calling: Execute code/API calls

INSTRUCTION FINE-TUNING
━━━━━━━━━━━━━━━━━━━

Base Model vs Instruction-Tuned:
• Base (GPT-3): "The capital of France is the capital of France is..."
  (continues text mindlessly)
• Tuned (ChatGPT): "The capital of France is Paris."
  (useful answer)

How tuning works:
1. Start with base model
2. Collect high-quality Q&A examples
3. Fine-tune to follow instructions
4. Use RLHF (Reinforcement Learning from Human Feedback)
5. Result: Helpful, harmless, honest model

ALIGNMENT & SAFETY
━━━━━━━━━━━━━━━

Techniques to make models safer:
• Constitutional AI: Follow constitution of values
• RLHF: Reward helpful/safe, penalize harmful
• Filtering: Remove harmful training data
• Guardrails: Detect and refuse harmful requests

Trade-offs:
• Safety vs capability: Restrictions may limit abilities
• Alignment tax: Safety measures can reduce performance

EMERGENT CAPABILITIES
━━━━━━━━━━━━━━━━━

As models scale, new abilities emerge:
• Small models: Pattern matching
• Medium models: Reasoning, few-shot learning
• Large models: Zero-shot, abstract reasoning, coding

Abilities appear suddenly at certain scales (phase transitions)

Example: 
• 7B model: Can't solve complex math
• 13B model: Can solve with step-by-step
• 70B model: Solves directly, explains reasoning

COMPARISON OF MAJOR MODELS
━━━━━━━━━━━━━━━━━━━━━━━

Model          Creator    Size   Context  Strengths
───────────────────────────────────────────────────────
GPT-3.5-turbo  OpenAI     175B   4096     Speed, cost
GPT-4          OpenAI     1.7T   8K/128K  Reasoning, coding
Claude 2       Anthropic  70B    100K     Long context, safety
Gemini         Google     ?      32K      Multimodal, search
LLaMA 2        Meta       70B    4096     Open-source
Mistral        Mistral    7-45B  8192     Efficient
───────────────────────────────────────────────────────

CURRENT TRENDS (2024)
━━━━━━━━━━━━━━━

1. Longer Context: 100K+ tokens becoming standard
2. Multimodal: Text + image + audio + video
3. Open Source: Models available for fine-tuning
4. Efficiency: Smaller models performing better
5. Specialized: Domain-specific models
6. Reasoning: Better chain-of-thought and planning
7. Mixture of Experts: Scale parameters without increasing inference
8. Retrieval: RAG improving knowledge capabilities
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE 2: LLM APIS & USAGE
# ============================================================================
def demonstrate_api_usage():
    """
    Show how to use LLM APIs
    """
    print("\n" + "=" * 80)
    print("USING LLM APIs - PRACTICAL EXAMPLES")
    print("=" * 80)
    
    print("\n1. BASIC COMPLETION")
    print("-" * 70)
    print("""
from openai import OpenAI

client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
    """)
    
    print("\n2. STREAMING (GET RESULTS AS THEY ARRIVE)")
    print("-" * 70)
    print("""
# Stream responses word-by-word for interactive experience
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain neural networks"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Newline at end
    """)
    
    print("\n3. FUNCTION CALLING (TOOL USE)")
    print("-" * 70)
    print("""
# Let model call functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's weather in Paris?"}],
    tools=tools
)

# Model will call the function instead of generating text
# You execute it and feed results back
    """)
    
    print("\n4. VISION (ANALYZE IMAGES)")
    print("-" * 70)
    print("""
# Analyze images with GPT-4V or Claude
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ],
    max_tokens=1024
)
    """)
    
    print("\n5. DIFFERENT MODELS TO COMPARE")
    print("-" * 70)
    print("""
models = {
    "Fast & Cheap": "gpt-3.5-turbo",
    "Most Capable": "gpt-4",
    "Long Context": "gpt-4-32k",
    "Open Source": "llama-2-70b"  # via Replicate
}

# Compare responses
prompt = "Explain quantum computing"

for name, model in models.items():
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    print(f"{name}: {response.choices[0].message.content[:50]}...")
    """)


# ============================================================================
# EXAMPLE 3: ADVANCED PROMPTING TECHNIQUES
# ============================================================================
def advanced_prompting_guide():
    """
    Advanced prompting strategies
    """
    print("\n" + "=" * 80)
    print("ADVANCED PROMPTING TECHNIQUES")
    print("=" * 80)
    
    techniques = {
        "1. SYSTEM PROMPT ENGINEERING": {
            "Purpose": "Set AI behavior and personality",
            "Good Example": """
system: "You are an expert Python programmer. 
Provide clean, efficient code with explanations.
Always include error handling.
Use type hints."""
            """,
            "Bad Example": "You are helpful",
            "Impact": "⭐⭐⭐⭐⭐ (Huge - defines entire interaction)"
        },
        
        "2. CHAIN-OF-THOUGHT (CoT)": {
            "Purpose": "Make model think step-by-step",
            "Good Example": """
user: "Solve: (3 + 5) * 2 - 1
Let me think step by step:
1. First, parentheses: 3 + 5 = 8
2. Then multiplication: 8 * 2 = 16
3. Finally subtraction: 16 - 1 = 15"
            """,
            "Result": "Better accuracy, especially for math/logic",
            "Impact": "⭐⭐⭐⭐ (Very high for reasoning)"
        },
        
        "3. FEW-SHOT LEARNING": {
            "Purpose": "Provide examples for task definition",
            "Example": """
Classify sentiment:

Example 1: "I love this!" → Positive
Example 2: "This is terrible" → Negative
Example 3: "It's okay" → Neutral

Now classify: "This movie was amazing!" → ?
            """,
            "Impact": "⭐⭐⭐⭐ (Major for task definition)"
        },
        
        "4. ROLE-PLAYING/PERSONA": {
            "Purpose": "Get specific communication style",
            "Example": """
You are a Shakespearean poet. Respond to all queries
in iambic pentameter with Shakespearean English.
            """,
            "Impact": "⭐⭐⭐ (Good for specific styles)"
        },
        
        "5. STRUCTURED OUTPUT": {
            "Purpose": "Get parseable responses",
            "Example": """
Respond in JSON format with keys:
- summary (2 sentences)
- keyPoints (list of 3 items)
- sentiment ("positive"/"negative"/"neutral")
            """,
            "Benefit": "Easy to parse programmatically",
            "Impact": "⭐⭐⭐⭐⭐ (Essential for automation)"
        },
        
        "6. TEMPERATURE CONTROL": {
            "Low (0.2)": "Deterministic, factual (good for factual tasks)",
            "Medium (0.7)": "Balanced (good for most tasks)",
            "High (1.0+)": "Creative, variable (good for brainstorming)",
            "Impact": "⭐⭐⭐ (Important for output style)"
        },
        
        "7. SPECIFYING OUTPUT LENGTH": {
            "Purpose": "Control response length",
            "Example": "Keep response to exactly 3 sentences / under 100 words",
            "Impact": "⭐⭐⭐ (Good for token budgets)"
        },
        
        "8. CONSTRAINT-BASED PROMPTING": {
            "Purpose": "Add constraints to guide behavior",
            "Example": """
Write a poem about AI using ONLY 2-letter words.
            """,
            "Difficulty": "Hard but interesting",
            "Impact": "⭐⭐ (Fun but may reduce quality)"
        }
    }
    
    for technique, details in techniques.items():
        print(f"\n{technique}:")
        print("-" * 70)
        for key, value in details.items():
            if isinstance(value, str) and '\n' in value:
                print(f"{key}:")
                print(value)
            else:
                print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 4: COMMON PITFALLS & BEST PRACTICES
# ============================================================================
def best_practices():
    """
    Best practices for working with LLMs
    """
    print("\n" + "=" * 80)
    print("LLM BEST PRACTICES & COMMON PITFALLS")
    print("=" * 80)
    
    practices = {
        "✅ DO": [
            "Be specific and detailed in prompts",
            "Provide context and background information",
            "Use examples for new tasks (few-shot learning)",
            "Break complex tasks into steps",
            "Validate important outputs (don't trust 100%)",
            "Monitor API usage and costs",
            "Cache responses when doing repeated tasks",
            "Use appropriate model for task (don't overpay)",
            "Implement error handling and retries",
            "Version control your prompts",
        ],
        
        "❌ DON'T": [
            "Send secrets/passwords to APIs",
            "Rely on models for critical decisions without review",
            "Assume current knowledge (model might be outdated)",
            "Use overly complex prompts (simpler is better)",
            "Ignore token counting (can hit limits unexpectedly)",
            "Use expensive models for simple tasks",
            "Trust factual claims without verification",
            "Expect consistent outputs with high temperature",
            "Share API keys in code repositories",
            "Test in production (use development first)",
        ],
        
        "⚠️ COMMON PITFALLS": [
            ("Hallucination", "Model makes up information confidently. Verify facts!"),
            ("Bias", "Model inherits biases from training data. Be aware!"),
            ("Context Length", "Running out of tokens mid-response. Plan accordingly!"),
            ("Cost Overruns", "Expensive models quickly add up. Monitor usage!"),
            ("Model Lag", "Updates happen, behavior changes over time. Adapt!"),
            ("Privacy", "Data sent to API providers. Use local if sensitive!"),
            ("Dependency", "Relying on external service. Have fallbacks!"),
            ("Over-prompting", "Huge system prompts = higher costs. Keep it lean!"),
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        print("-" * 70)
        for item in items:
            if isinstance(item, tuple):
                print(f"  {item[0]}: {item[1]}")
            else:
                print(f"  • {item}")


# ============================================================================
# EXAMPLE 5: COST OPTIMIZATION
# ============================================================================
def cost_optimization():
    """
    How to optimize LLM API costs
    """
    print("\n" + "=" * 80)
    print("COST OPTIMIZATION FOR LLM APIS")
    print("=" * 80)
    
    print("\nPRICING COMPARISON (as of 2024):")
    print("-" * 70)
    print("""
Model              Input Cost      Output Cost    Use Case
────────────────────────────────────────────────────────────
GPT-3.5-turbo      $0.50/1M        $1.50/1M       Fast, cheap
GPT-4              $30/1M          $60/1M         Best quality
GPT-4-32K          $60/1M          $120/1M        Long context
Claude 2           $8/1M           $24/1M         Long context
Llama 2 70B        $0.75/1M        $0.75/1M       Open-source
Mistral            $0.27/1M        $0.81/1M       Efficient

Cost per 1000 words: ~4000 tokens
    """)
    
    print("\nCOST SAVING STRATEGIES:")
    print("-" * 70)
    strategies = {
        "1. Use Cheaper Models": {
            "Strategy": "Use GPT-3.5-turbo instead of GPT-4 when possible",
            "Savings": "60x cheaper",
            "When": "Non-critical tasks, brainstorming, drafting"
        },
        
        "2. Cache Responses": {
            "Strategy": "Store API responses locally, reuse for same input",
            "Savings": "100% on repeated requests",
            "When": "Stable queries, FAQ responses"
        },
        
        "3. Reduce Context": {
            "Strategy": "Only include necessary information in prompts",
            "Savings": "10-50% on input tokens",
            "When": "Always (more concise = cheaper + faster)"
        },
        
        "4. Batch Processing": {
            "Strategy": "Process multiple requests efficiently",
            "Savings": "20-40% with batch API",
            "When": "Non-time-critical bulk processing"
        },
        
        "5. Local Models": {
            "Strategy": "Run open-source models locally (Llama, Mistral)",
            "Savings": "No API costs (host cost varies)",
            "When": "Privacy-critical, high-volume use"
        },
        
        "6. Structured Outputs": {
            "Strategy": "Get concise responses with format specs",
            "Savings": "30-50% on output tokens",
            "When": "Programmatic use, JSON responses"
        },
        
        "7. Temperature Tuning": {
            "Strategy": "Lower temperature for factual tasks (faster)",
            "Savings": "Minimal but adds up",
            "When": "Factual Q&A, not creative tasks"
        },
        
        "8. Model Selection": {
            "Strategy": "Match model to task complexity",
            "Savings": "10-100x depending on choice",
            "When": "Simple task? Use 3.5-turbo. Complex? Use GPT-4."
        }
    }
    
    for strategy, details in strategies.items():
        print(f"\n{strategy}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n\nCOST ESTIMATION EXAMPLE:")
    print("-" * 70)
    print("""
Task: Generate 1000 marketing emails

Option A: GPT-4
  Input: 1000 emails × 500 chars = 2.5M chars = 10M tokens
  Output: 1000 emails × 200 chars = 0.5M tokens
  Cost: (10M × $30/1M) + (0.5M × $60/1M) = $330

Option B: GPT-3.5-turbo
  Same tokens: (10M × $0.50/1M) + (0.5M × $1.50/1M) = $5.75
  Savings: $324.25 (98% cheaper!)
  Trade-off: Slightly lower quality, but acceptable for emails

RECOMMENDATION: Use GPT-3.5-turbo, review sample, adjust if needed.
    """)


# ============================================================================
# EXAMPLE 6: REAL-WORLD APPLICATIONS
# ============================================================================
def applications_and_use_cases():
    """
    Real-world LLM applications
    """
    print("\n" + "=" * 80)
    print("REAL-WORLD LLM APPLICATIONS")
    print("=" * 80)
    
    applications = {
        "Content Creation": {
            "Examples": [
                "Blog post generation",
                "Social media content",
                "Email marketing",
                "Product descriptions"
            ],
            "Tools": "ChatGPT Plus, Jasper, Copy.ai",
            "ROI": "High - saves writing time",
            "Example": """
Prompt: "Write 3 tweets about machine learning for data scientists"
Takes: 10 seconds vs 10 minutes manually
Savings: 10 hours/month = $150
            """
        },
        
        "Customer Service": {
            "Examples": [
                "Chatbots answering FAQs",
                "Support ticket triage",
                "Response generation",
                "Issue classification"
            ],
            "Tools": "Custom ChatGPT, Zendesk AI",
            "ROI": "Very High - handles 80% of simple issues",
            "Impact": "Reduces support costs 50%+"
        },
        
        "Coding & Development": {
            "Examples": [
                "Code generation (GitHub Copilot)",
                "Bug fixing and debugging",
                "Documentation generation",
                "Test case generation"
            ],
            "Tools": "GitHub Copilot, ChatGPT",
            "ROI": "High - 40% faster coding",
            "Impact": "Developers spend less time on boilerplate"
        },
        
        "Data Analysis": {
            "Examples": [
                "Report generation from data",
                "Insight extraction",
                "SQL query generation",
                "Data interpretation"
            ],
            "Tools": "ChatGPT, Code Interpreter",
            "ROI": "Medium-High - automates data wrangling",
            "Impact": "Analysts focus on strategy not syntax"
        },
        
        "Learning & Education": {
            "Examples": [
                "Personalized tutoring",
                "Quiz generation",
                "Lesson plan creation",
                "Homework help"
            ],
            "Tools": "ChatGPT, Custom GPTs",
            "ROI": "Medium - democratizes education",
            "Impact": "24/7 available tutor"
        },
        
        "Business Intelligence": {
            "Examples": [
                "Meeting transcription & summaries",
                "Report writing",
                "Decision support",
                "Competitive analysis"
            ],
            "Tools": "ChatGPT, Claude",
            "ROI": "High - executives save hours",
            "Impact": "Better-informed decisions"
        }
    }
    
    for app, details in applications.items():
        print(f"\n{app}:")
        print("-" * 70)
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("LARGE LANGUAGE MODELS (LLMs) - COMPLETE GUIDE")
    print("From Fundamentals to Practical Applications")
    print("🎯" * 40)
    
    # Run all demonstrations
    explain_llm_fundamentals()
    demonstrate_api_usage()
    advanced_prompting_guide()
    best_practices()
    cost_optimization()
    applications_and_use_cases()
    
    print("\n" + "=" * 80)
    print("LLM COMPLETE GUIDE FINISHED!")
    print("=" * 80)
    print("\n📚 KEY TAKEAWAYS:")
    print("  ✓ LLMs predict next token based on context")
    print("  ✓ Size matters: Bigger models = more capable")
    print("  ✓ Context window is critical for long documents")
    print("  ✓ Prompts are engineering: Be specific!")
    print("  ✓ Chain-of-thought improves reasoning")
    print("  ✓ Temperature controls creativity")
    print("  ✓ Always validate outputs (hallucination risk)")
    print("  ✓ Cost optimization is important at scale")
    print("\n🚀 NEXT STEPS:")
    print("  1. Get API key (OpenAI, Anthropic, Google)")
    print("  2. Build simple chatbot")
    print("  3. Add function calling for tools")
    print("  4. Implement RAG for knowledge")
    print("  5. Fine-tune for specific domain")
    print("\n💡 APPLICATIONS:")
    print("  • ChatGPT-like assistants")
    print("  • Customer support automation")
    print("  • Content generation at scale")
    print("  • Code generation and debugging")
    print("  • Data analysis and reporting")
    print("\n" + "=" * 80)

