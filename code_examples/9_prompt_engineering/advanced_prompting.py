"""
Prompt Engineering: Complete Advanced Guide
Master the art and science of writing effective prompts for AI models
Covers: Techniques, frameworks, evaluation, and advanced strategies
"""

from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXAMPLE 1: PROMPT QUALITY FRAMEWORK
# ============================================================================
def prompt_quality_framework():
    """
    Framework for evaluating and improving prompt quality
    """
    print("=" * 80)
    print("PROMPT ENGINEERING FRAMEWORK")
    print("=" * 80)
    
    explanation = """
PROMPT QUALITY SPECTRUM
━━━━━━━━━━━━━━━━━━━━

⭐ Level 1 (Terrible):
Prompt: "Write about AI"
Problem: No direction, scope undefined
Expected output: Random, potentially irrelevant
Quality score: 1/10

⭐⭐ Level 2 (Poor):
Prompt: "Write about AI for beginners"
Problem: Still vague, no specific angle
Expected output: Generic overview
Quality score: 3/10

⭐⭐⭐ Level 3 (Okay):
Prompt: "Write a 3-paragraph explanation of machine learning for high school students"
Good: Audience, length, topic clear
Missing: Tone, specific focus, format
Quality score: 5/10

⭐⭐⭐⭐ Level 4 (Good):
Prompt: "Write 3 paragraphs explaining machine learning to high school students.
Use analogies they can relate to. Keep language simple. Focus on real-world examples."
Good: Specific, audience-aware, examples
Missing: Output format, specific use case
Quality score: 7/10

⭐⭐⭐⭐⭐ Level 5 (Excellent):
Prompt: "You are an experienced science teacher explaining concepts to teenagers.
Write exactly 3 paragraphs explaining machine learning.
Each paragraph should:
1. Start with a relatable analogy
2. Explain one core concept
3. End with a real-world example

Use vocabulary appropriate for 14-year-olds (9th grade reading level).
Tone: Conversational but informative.
Avoid: Jargon, complex mathematics."

Why excellent:
- Clear role/persona
- Specific requirements per section
- Audience profiling
- Tone guidance
- Constraints clearly stated
Quality score: 9/10

KEY DIMENSIONS OF PROMPT QUALITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CLARITY
   Does the model understand what you want?
   How to improve: Use concrete language, avoid ambiguity

2. SPECIFICITY
   How detailed are requirements?
   How to improve: Add constraints, examples, format specs

3. CONTEXT
   Does model have needed background?
   How to improve: Include relevant information upfront

4. STRUCTURE
   Is prompt well-organized?
   How to improve: Use sections, bullet points, numbered lists

5. EXAMPLES
   Are there demonstrations?
   How to improve: Add 1-3 examples of desired output

6. CONSTRAINTS
   What limits are set?
   How to improve: Specify length, tone, format

7. AUDIENCE
   Is target audience clear?
   How to improve: Specify who will read the output

8. SUCCESS CRITERIA
   How would you know if it's good?
   How to improve: Define what "good" looks like

THE PROMPT ENGINEERING PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: DEFINE THE TASK
   What exactly do you want?
   Brainstorm: Goal, audience, constraints

Step 2: DRAFT PROMPT
   Write initial version
   Include: Role, task, constraints, examples

Step 3: TEST PROMPT
   Try it with AI model
   Evaluate: Quality, relevance, format

Step 4: ANALYZE OUTPUT
   Is output what you wanted?
   If not: Identify what's wrong

Step 5: ITERATE
   Refine prompt based on results
   Repeat until satisfied

Step 6: DOCUMENT
   Save final prompt
   Note: What works, what doesn't

THE ICARUS PRINCIPLE
━━━━━━━━━━━━━━━

DON'T:  Go too far in one direction
DO:     Balance specificity with flexibility

Too Specific:
- Overly rigid constraints
- Prescribes exact wording
- Leaves no room for model creativity
- Result: Awkward, stilted output

Too Vague:
- Barely any constraints
- Model must guess intent
- Result: Hit-or-miss outputs

Sweet Spot:
- Clear requirements
- Room for quality execution
- Examples for format
- Result: Consistent, good quality
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE 2: ADVANCED TECHNIQUES
# ============================================================================
def advanced_techniques():
    """
    Advanced prompting techniques with examples
    """
    print("\n" + "=" * 80)
    print("ADVANCED PROMPTING TECHNIQUES")
    print("=" * 80)
    
    techniques = {
        "1. CHAIN-OF-THOUGHT (CoT)": {
            "What": "Make model show step-by-step reasoning",
            "Why": "Improves accuracy, especially for logic/math",
            "Example": """
Weak: "What is 523 + 487?"
Better: "Let me think through this step by step.
523 + 487
= 500 + 20 + 3 + 400 + 80 + 7
= 900 + 100 + 10
= 1010"
            """,
            "Results": "Accuracy improvement: 5-40% depending on task"
        },
        
        "2. TREE-OF-THOUGHT": {
            "What": "Explore multiple reasoning paths",
            "Why": "Better for complex problems with many steps",
            "Example": """
For decision making:
"Consider this decision from 3 angles:
1. Financial perspective: Cost-benefit analysis
2. Ethical perspective: Who benefits, who loses?
3. Long-term perspective: What are future implications?

Then synthesize a recommendation."
            """,
            "Results": "Better balanced decisions"
        },
        
        "3. FEW-SHOT LEARNING": {
            "What": "Provide examples of desired behavior",
            "Why": "Teaches model pattern without extensive explanation",
            "Example": """
Task: Classify sentiment

Positive example:
Input: "I love this movie!"
Output: Positive (Confidence: 95%)

Negative example:
Input: "This is terrible"
Output: Negative (Confidence: 90%)

Neutral example:
Input: "It's okay"
Output: Neutral (Confidence: 85%)

Now classify: "This was amazing!" → ?
            """,
            "Best practice": "3-5 examples usually optimal"
        },
        
        "4. ZERO-SHOT PROMPTING": {
            "What": "No examples, just instructions",
            "Why": "Works with modern large models",
            "Example": """
"Translate to French: The weather is nice today"
            """,
            "Trade-off": "Simpler but less reliable than few-shot"
        },
        
        "5. ROLE-BASED PROMPTING": {
            "What": "Give model a persona/role",
            "Why": "Gets specific style and expertise level",
            "Example": """
"You are a world-class Python developer with 20 years experience.
Write production-ready code with error handling.
Include type hints and docstrings.
Optimize for readability and performance.
"
            """,
            "Impact": "Huge improvement in relevant expertise"
        },
        
        "6. CONSTRAINT-BASED": {
            "What": "Give specific constraints to follow",
            "Why": "Guides output format and content",
            "Example": """
"Write a haiku about AI
- Exactly 3 lines
- 5-7-5 syllable structure
- No rhyming
- Include at least one technology term"
            """,
            "Difficulty": "Hard constraints can be hard to follow"
        },
        
        "7. COMPARISON PROMPTING": {
            "What": "Ask model to compare/contrast",
            "Why": "Better analysis through comparison",
            "Example": """
"Compare GPT-4 and Claude 2:
- Strengths of each
- Weaknesses of each
- Best use cases
- Cost comparison
- Reasoning abilities"
            """,
            "Result": "More thorough analysis"
        },
        
        "8. SELF-CRITIQUE": {
            "What": "Have model critique own output",
            "Why": "Improves quality through iteration",
            "Example": """
"Write a paragraph about quantum computing.
Then critique your own writing for:
- Technical accuracy
- Clarity for general audience
- Engagement
- Suggest one improvement"
            """,
            "Result": "Better quality with self-awareness"
        },
        
        "9. PROMPT CHAINING": {
            "What": "Use output of one prompt as input to next",
            "Why": "Breaks complex tasks into manageable steps",
            "Example": """
Step 1: "Generate 5 blog post ideas about AI"
Step 2: "Expand the best idea into an outline"
Step 3: "Write the full blog post from outline"
            """,
            "Benefit": "Better results than single massive prompt"
        },
        
        "10. META-PROMPTING": {
            "What": "Have model improve its own prompts",
            "Why": "Can optimize for specific task",
            "Example": """
"Given this task: [task description]
What would be the ideal prompt to solve it?
Create a detailed prompt that:
- Specifies role
- Lists requirements
- Includes examples
- Defines constraints"
            """,
            "Result": "AI-generated prompts can be better than manual"
        }
    }
    
    for technique, details in techniques.items():
        print(f"\n{technique}:")
        print("-" * 70)
        for key, value in details.items():
            if '\n' in str(value):
                print(f"{key}:{value}")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 3: SYSTEM PROMPTS
# ============================================================================
def system_prompt_mastery():
    """
    Master the art of system prompts (for ChatGPT and similar APIs)
    """
    print("\n" + "=" * 80)
    print("SYSTEM PROMPT MASTERY")
    print("=" * 80)
    
    print("\nSYSTEM PROMPT = AI BEHAVIOR DEFINITION")
    print("-" * 70)
    
    print("""
System prompt runs EVERY conversation.
Sets the "personality" and behavior of the model.

COMPONENTS OF GREAT SYSTEM PROMPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRIMARY ROLE
   "You are a helpful customer support agent"
   "You are an expert data scientist"
   "You are a code reviewer"

2. CORE VALUES
   "Be concise but complete"
   "Prioritize user safety"
   "Show your reasoning"

3. SPECIFIC BEHAVIORS
   "Always explain 'why' before giving advice"
   "Use numbered lists for multiple items"
   "Flag uncertain information"

4. CONSTRAINTS
   "Don't provide medical advice"
   "Don't write code without explanation"
   "Don't make assumptions about user intent"

5. RESPONSE FORMAT
   "Structure responses with headers"
   "Use markdown formatting"
   "Include code blocks with syntax highlighting"

6. TONE/STYLE
   "Be professional but friendly"
   "Use simple language for general audiences"
   "Match academic tone for scholarly writing"
""")
    
    print("\nEXAMPLE 1: CUSTOMER SUPPORT AGENT")
    print("-" * 70)
    print("""
You are a helpful and empathetic customer support agent.

YOUR ROLE:
- Solve customer problems efficiently
- Be warm and understanding
- Escalate complex issues appropriately

YOUR VALUES:
- Customer satisfaction is priority #1
- Admit when you don't know something
- Treat each customer with respect

GUIDELINES:
- Read entire issue before responding
- Ask clarifying questions if needed
- Provide step-by-step solutions
- Offer follow-up help

RESPONSE FORMAT:
1. Empathize with their situation
2. State your understanding of the problem
3. Provide solution with clear steps
4. Offer next steps or alternatives
5. Ask if this resolved their issue

CONSTRAINTS:
- Don't make promises you can't keep
- Don't share other customers' information
- Don't dismiss any concern as unimportant
""")
    
    print("\nEXAMPLE 2: TECHNICAL DOCUMENTATION WRITER")
    print("-" * 70)
    print("""
You are a world-class technical documentation writer.

EXPERTISE:
- Explaining complex technical concepts clearly
- Structuring information for different audiences
- Writing for developers (junior and senior)
- Creating examples that actually work

WRITING PRINCIPLES:
- Start simple, build complexity
- Show, don't just tell (examples > theory)
- Include common gotchas and errors
- Anticipate questions users might have

STRUCTURE FOR EACH SECTION:
1. What (brief explanation)
2. Why (motivation, when to use)
3. How (step-by-step with examples)
4. Common mistakes (what to avoid)
5. Related topics (where to go next)

TONE:
- Professional but conversational
- Confident but not arrogant
- Clear and direct
- Helpful and encouraging

FORMAT:
- Use headers for organization
- Code blocks with syntax highlighting
- Tables for comparisons
- Lists for multiple items
- Callouts for warnings/tips
""")
    
    print("\nEXAMPLE 3: RESEARCH ASSISTANT")
    print("-" * 70)
    print("""
You are a research assistant with expertise in synthesizing information.

YOUR ROLE:
- Find relevant information
- Synthesize multiple sources
- Identify gaps and contradictions
- Support researcher's work

RESEARCH METHODOLOGY:
- Distinguish fact from opinion
- Rate confidence in claims
- Note conflicting information
- Cite sources explicitly

OUTPUT STRUCTURE:
1. Summary (key findings)
2. Detailed explanation (organized by topic)
3. Contradictions (where sources disagree)
4. Gaps (what's missing or unclear)
5. Sources (references for further reading)
6. Recommendations (suggested next research)

QUALITY STANDARDS:
- Accuracy is paramount
- Flag uncertainty with confidence levels
- Never present opinion as fact
- Cite everything
- Acknowledge limitations

THINKING PROCESS:
1. Understand the research question
2. Identify what's known
3. Find gaps in knowledge
4. Synthesize information
5. Present with appropriate caveats
""")


# ============================================================================
# EXAMPLE 4: PROMPT TESTING & EVALUATION
# ============================================================================
def prompt_evaluation_framework():
    """
    Framework for testing and evaluating prompts
    """
    print("\n" + "=" * 80)
    print("PROMPT TESTING & EVALUATION")
    print("=" * 80)
    
    print("\nTHE PROMPT EVALUATION FRAMEWORK")
    print("-" * 70)
    
    criteria = {
        "1. CORRECTNESS": {
            "Question": "Is the output factually accurate?",
            "How to Test": "Verify facts, check examples, validate against sources",
            "Scoring": "% of facts that are correct (target: 95%+)"
        },
        
        "2. RELEVANCE": {
            "Question": "Does output address the prompt?",
            "How to Test": "Check if output actually answers the question",
            "Scoring": "Does it stay on topic? (Yes/No)"
        },
        
        "3. COMPLETENESS": {
            "Question": "Are all requested elements present?",
            "How to Test": "Checklist: Did it include X, Y, Z?",
            "Scoring": "# of required elements / total"
        },
        
        "4. CLARITY": {
            "Question": "Is output easy to understand?",
            "How to Test": "Can someone unfamiliar understand it?",
            "Scoring": "Readability score (1-10)"
        },
        
        "5. FORMAT": {
            "Question": "Does output match requested format?",
            "How to Test": "Is it structured as specified?",
            "Scoring": "Perfect match / Partially matches / No match"
        },
        
        "6. TONE": {
            "Question": "Does output match requested tone?",
            "How to Test": "Is it professional/casual/technical as requested?",
            "Scoring": "Perfect / Close / Wrong tone"
        },
        
        "7. LENGTH": {
            "Question": "Is output appropriate length?",
            "How to Test": "Count words/paragraphs",
            "Scoring": "Meets requirement / Close / Way off"
        },
        
        "8. CONSISTENCY": {
            "Question": "Is output consistent across runs?",
            "How to Test": "Run prompt 5 times, compare outputs",
            "Scoring": "% similarity between runs (target: 70%+)"
        }
    }
    
    for criterion, details in criteria.items():
        print(f"\n{criterion}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\n\nPROMPT TESTING PROCESS")
    print("-" * 70)
    print("""
1. CREATE TEST CASES
   - Identify 5-10 representative examples
   - Include edge cases
   - Have expected outputs

2. RUN TESTS
   - Try prompt with each test case
   - Record outputs
   - Note any surprises

3. EVALUATE
   - Score against criteria above
   - Calculate success rate
   - Identify patterns in failures

4. ITERATE
   - Identify why failures occurred
   - Modify prompt to address issues
   - Re-test with same cases

5. BENCHMARK
   - Compare to baseline/previous version
   - Measure improvement
   - Document what works

EXAMPLE TEST SUITE
━━━━━━━━━━━━━━━━

Prompt: "Classify sentiment of restaurant review"

Test Case 1:
Input: "Amazing food, great service! Highly recommend!"
Expected: Positive
Criteria: Accuracy, Classification

Test Case 2:
Input: "The place was okay. Food was cold."
Expected: Neutral/Mixed
Criteria: Handling ambiguous cases

Test Case 3:
Input: "Terrible experience. Never going back."
Expected: Negative
Criteria: Extreme sentiment

Test Case 4:
Input: "It was a place where we ate food."
Expected: Neutral
Criteria: Neutral/no opinion

Test Case 5:
Input: "Good restaurant. The waiter was rude though."
Expected: Mixed
Criteria: Conflicting sentiments
    """)


# ============================================================================
# EXAMPLE 5: COMMON MISTAKES TO AVOID
# ============================================================================
def avoid_common_mistakes():
    """
    Common prompting mistakes and how to avoid them
    """
    print("\n" + "=" * 80)
    print("COMMON PROMPTING MISTAKES")
    print("=" * 80)
    
    mistakes = {
        "❌ MISTAKE 1: UNCLEAR TASK": {
            "Bad": '"Tell me about Python"',
            "Problem": "Too vague - Python language, snake, or Monty Python?",
            "Good": '"Explain how Python lists work for a beginner programmer"',
            "Lesson": "Always specify context and level"
        },
        
        "❌ MISTAKE 2: WALL OF TEXT": {
            "Bad": "One giant paragraph prompt with no structure",
            "Problem": "Hard to read, easy to miss requirements",
            "Good": """ROLE: You are a Python expert
TASK: Explain list comprehensions
AUDIENCE: Beginners
FORMAT: Explanation + 3 examples
LENGTH: 200-300 words
TONE: Friendly""",
            "Lesson": "Structure prompts with sections and bullet points"
        },
        
        "❌ MISTAKE 3: NO EXAMPLES": {
            "Bad": '"Generate similar product descriptions"',
            "Problem": "Model has to guess what similar means",
            "Good": '''Show 1-2 examples of good descriptions
INPUT: A good description
OUTPUT: What makes it good
"Now generate similar for: [new product]"''',
            "Lesson": "Always include examples for new tasks"
        },
        
        "❌ MISTAKE 4: CONTRADICTORY CONSTRAINTS": {
            "Bad": '"Write a detailed summary in 1 sentence"',
            "Problem": "Can't be both detailed AND brief",
            "Good": '"Write a concise 2-3 sentence summary capturing key points"',
            "Lesson": "Ensure constraints are compatible"
        },
        
        "❌ MISTAKE 5: ASSUMING KNOWLEDGE": {
            "Bad": '"Complete this task like we discussed"',
            "Problem": "Model has no context of previous discussion",
            "Good": '"Here is the context: [provide full context]. Complete the task..."',
            "Lesson": "Always provide complete context"
        },
        
        "❌ MISTAKE 6: UNCLEAR SUCCESS CRITERIA": {
            "Bad": '"Make this better"',
            "Problem": "What does better mean?",
            "Good": '"Improve clarity, reduce length by 20%, make it more conversational"',
            "Lesson": "Define exactly what success looks like"
        },
        
        "❌ MISTAKE 7: OVERCOMPLICATING": {
            "Bad": '''Complex multi-part prompt with nested conditions
and special cases and edge cases and...''',
            "Problem": "Model gets confused, outputs suffer",
            "Good": "Break into 2-3 simpler prompts instead",
            "Lesson": "Simpler is usually better"
        },
        
        "❌ MISTAKE 8: IGNORING HALLUCINATIONS": {
            "Bad": "Trust everything model says as truth",
            "Problem": "Models confidently make up things",
            "Good": "Verify important facts independently",
            "Lesson": "Always validate, never blindly trust AI"
        }
    }
    
    for mistake, details in mistakes.items():
        print(f"\n{mistake}:")
        for key, value in details.items():
            print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 6: PROMPT TEMPLATES
# ============================================================================
def prompt_templates():
    """
    Ready-to-use prompt templates
    """
    print("\n" + "=" * 80)
    print("READY-TO-USE PROMPT TEMPLATES")
    print("=" * 80)
    
    templates = {
        "CUSTOMER SUPPORT": '''You are a helpful customer support agent.

Customer Issue: [ISSUE]

Steps:
1. Understand their problem
2. Ask clarifying questions if needed
3. Provide step-by-step solution
4. Offer alternatives if applicable
5. Ask if problem is resolved

Tone: Professional, empathetic, helpful
''',
        
        "CONTENT CREATION": '''You are an expert content creator.

Topic: [TOPIC]
Format: [BLOG POST / SOCIAL / EMAIL]
Target Audience: [AUDIENCE]
Tone: [TONE]
Length: [LENGTH]

Requirements:
- Include key points: [POINTS]
- Call to action: [CTA]
- Avoid: [AVOID]

Begin writing:
''',
        
        "CODE GENERATION": '''You are an expert Python developer.

Task: [TASK DESCRIPTION]

Requirements:
- Python 3.8+
- Include type hints
- Add error handling
- Include docstring
- Optimize for readability

Edge cases to handle: [CASES]

Code:
''',
        
        "ANALYSIS": '''You are an analytical expert.

Question: [QUESTION]

Analyze from perspectives:
1. Data perspective
2. Business perspective
3. Ethical perspective

For each, provide:
- Key insight
- Supporting evidence
- Potential implications

Conclusion: [SYNTHESIZE FINDINGS]
''',
        
        "WRITING": '''You are an excellent writer.

Topic: [TOPIC]
Format: [ARTICLE / ESSAY / STORY]
Audience: [AUDIENCE]
Purpose: [PURPOSE]

Structure:
1. Introduction: [PURPOSE]
2. Main points: [POINTS]
3. Conclusion: [RESOLUTION]

Style: [STYLE]
Length: [LENGTH]

Begin writing:
'''
    }
    
    for template_name, template_text in templates.items():
        print(f"\n{template_name}:")
        print("-" * 70)
        print(template_text)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("PROMPT ENGINEERING: ADVANCED COMPLETE GUIDE")
    print("Master the Art of AI Communication")
    print("🎯" * 40)
    
    # Run all demonstrations
    prompt_quality_framework()
    advanced_techniques()
    system_prompt_mastery()
    prompt_evaluation_framework()
    avoid_common_mistakes()
    prompt_templates()
    
    print("\n" + "=" * 80)
    print("ADVANCED PROMPT ENGINEERING GUIDE COMPLETE!")
    print("=" * 80)
    print("\n📚 KEY TAKEAWAYS:")
    print("  ✓ Clarity > Length: Clear prompts beat long prompts")
    print("  ✓ Structure matters: Organized prompts get better results")
    print("  ✓ Examples work: Few-shot learning is powerful")
    print("  ✓ Show your thinking: Chain-of-thought improves reasoning")
    print("  ✓ Test systematically: Iterate based on results")
    print("  ✓ System prompts control behavior: Set expectations")
    print("  ✓ No two prompts identical: Iteration is key")
    print("  ✓ Always validate: Never trust 100%, verify important outputs")
    print("\n🚀 NEXT STEPS:")
    print("  1. Start with prompt templates")
    print("  2. Test with your own tasks")
    print("  3. Iterate and refine")
    print("  4. Document what works")
    print("  5. Build prompt library for reuse")
    print("  6. Share successful prompts with team")
    print("  7. Measure prompt effectiveness")
    print("\n💡 BEST PRACTICES:")
    print("  • Keep prompts concise but complete")
    print("  • Use structure (bullet points, sections)")
    print("  • Include 2-3 examples for complex tasks")
    print("  • Test with multiple runs for consistency")
    print("  • Version control your prompts")
    print("  • Document context and reasoning")
    print("\n" + "=" * 80)

