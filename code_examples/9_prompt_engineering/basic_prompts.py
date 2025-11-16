"""
Prompt Engineering: Fundamental Techniques
Demonstrating prompt quality levels and techniques
"""

import os
from typing import Dict, List


# ============================================================================
# EXAMPLE 1: PROMPT QUALITY LEVELS
# ============================================================================
def demonstrate_prompt_quality():
    """
    Show progression from bad to excellent prompts
    """
    print("=" * 80)
    print("PROMPT QUALITY COMPARISON")
    print("=" * 80)
    
    prompts = {
        "❌ Level 1: Bad - Too Vague": {
            "prompt": "Write about AI",
            "problems": [
                "No direction or scope",
                "No target audience",
                "No format specified",
                "Unpredictable output"
            ],
            "example_outputs": [
                "1 sentence",
                "100 pages",
                "Poetry",
                "Technical paper"
            ]
        },
        
        "⚠️  Level 2: Better - Some Direction": {
            "prompt": "Write a 3-paragraph explanation of AI for beginners",
            "improvements": [
                "Specified length",
                "Defined audience"
            ],
            "issues": [
                "No specific structure",
                "No content requirements",
                "No tone specification"
            ]
        },
        
        "✓ Level 3: Good - Clear Requirements": {
            "prompt": """Write a 3-paragraph explanation of artificial intelligence 
            for high school students that:
            1. Defines AI clearly
            2. Provides real-world examples
            3. Explains one limitation
            Use simple language.""",
            "benefits": [
                "Clear requirements",
                "Specific audience",
                "Defined structure",
                "Language guidance"
            ],
            "still_missing": [
                "Tone specification",
                "Output format details"
            ]
        },
        
        "★ Level 4: Excellent - Complete": {
            "prompt": """You are an AI educator explaining concepts to high school students.

Task: Write a 3-paragraph explanation of artificial intelligence

Requirements:
1. Paragraph 1: Define AI in simple, conversational terms
2. Paragraph 2: Give 2-3 real-world examples (e.g., Netflix, ChatGPT)
3. Paragraph 3: Explain one significant limitation of current AI

Guidelines:
- Use conversational tone as if explaining to a friend
- Avoid technical jargon
- Each paragraph: 4-6 sentences
- Include one analogy or comparison
- End with a thought-provoking question

Format: Plain text, clear paragraph breaks""",
            "strengths": [
                "Role clearly defined",
                "Task is explicit",
                "All requirements listed",
                "Guidelines specified",
                "Format defined",
                "Length specified",
                "Tone guidance",
                "Examples provided"
            ]
        }
    }
    
    for level, details in prompts.items():
        print(f"\n{level}")
        print("-" * 80)
        print(f"Prompt: {details['prompt'][:100]}...")
        
        if "problems" in details:
            print(f"\nProblems:")
            for p in details["problems"]:
                print(f"  ❌ {p}")
            print(f"\nUnpredictable outputs:")
            for output in details["example_outputs"]:
                print(f"  • {output}")
        
        if "improvements" in details:
            print(f"\nImprovements from Level 1:")
            for imp in details["improvements"]:
                print(f"  ✓ {imp}")
            print(f"\nStill missing:")
            for issue in details["issues"]:
                print(f"  ⚠️  {issue}")
        
        if "benefits" in details:
            print(f"\nBenefits:")
            for benefit in details["benefits"]:
                print(f"  ✓ {benefit}")
        
        if "strengths" in details:
            print(f"\nAll aspects covered:")
            for strength in details["strengths"]:
                print(f"  ★ {strength}")


# ============================================================================
# EXAMPLE 2: PROMPT TEMPLATES
# ============================================================================
class PromptTemplate:
    """Template-based prompt generation"""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format template with variables"""
        return self.template.format(**kwargs)
    
    def display(self):
        """Show template structure"""
        return self.template


def prompt_templates():
    """
    Demonstrate reusable prompt templates for common tasks
    """
    print("\n" + "=" * 80)
    print("REUSABLE PROMPT TEMPLATES")
    print("=" * 80)
    
    # Template 1: Content Writing
    content_template = PromptTemplate("""You are a professional {role}.

Task: Write about "{topic}"

Target Audience: {audience}
Tone: {tone}
Length: {length} words
Format: {format}

Requirements:
{requirements}

Include:
{include_points}

Avoid:
{avoid_points}

Structure:
{structure}""")
    
    # Example usage
    print("\n1. CONTENT WRITING TEMPLATE")
    print("-" * 80)
    
    filled = content_template.format(
        role="professional copywriter",
        topic="benefits of exercise",
        audience="busy professionals aged 30-45",
        tone="motivating but realistic",
        length="500-750",
        format="Blog post with catchy headline",
        requirements="- Evidence-based information\n- Practical tips\n- Personal success story",
        include_points="- Time-efficient workouts\n- Mental health benefits\n- Easy-to-follow plan",
        avoid_points="- Extreme diet talk\n- Unrealistic promises\n- Medical terminology",
        structure="Hook → Problem → Solution → Call-to-Action"
    )
    
    print(filled)
    
    # Template 2: Code Generation
    code_template = PromptTemplate("""Generate {language} code to {task}

Framework/Library: {framework}
Python version: {version}

Requirements:
{requirements}

Code style:
- Include comments for complex sections
- Use meaningful variable names
- Add error handling for: {error_cases}
- Follow {style_guide}

Example input: {input_example}
Expected output: {output_example}""")
    
    print("\n\n2. CODE GENERATION TEMPLATE")
    print("-" * 80)
    
    filled = code_template.format(
        language="Python",
        task="fetch data from an API and save to CSV",
        framework="requests and pandas",
        version="3.8+",
        requirements="- Handle network errors\n- Retry failed requests\n- Log progress",
        error_cases="network timeouts, invalid JSON, file write errors",
        style_guide="PEP 8",
        input_example='url = "https://api.example.com/data"',
        output_example="data.csv with columns: id, name, value"
    )
    
    print(filled)
    
    # Template 3: Analysis & Research
    analysis_template = PromptTemplate("""Analyze the following {subject}: 

{subject_details}

Analysis Type: {analysis_type}
Focus Areas: {focus_areas}

Instructions:
1. Summarize key findings (200-300 words)
2. Identify trends and patterns
3. Highlight surprising insights
4. Consider implications from {perspectives} perspectives

Output Format:
- Executive Summary
- Detailed Findings
- Key Trends
- Recommended Actions

Avoid: {caveats}""")
    
    print("\n\n3. ANALYSIS TEMPLATE")
    print("-" * 80)
    
    filled = analysis_template.format(
        subject="customer feedback",
        subject_details="[Provide survey results or feedback data]",
        analysis_type="Sentiment and trend analysis",
        focus_areas="Customer satisfaction, pain points, feature requests",
        perspectives="3-4",
        caveats="Speculation beyond data, assuming causation without evidence"
    )
    
    print(filled)


# ============================================================================
# EXAMPLE 3: PROMPT COMPONENTS
# ============================================================================
def prompt_components_breakdown():
    """
    Analyze different components of an effective prompt
    """
    print("\n" + "=" * 80)
    print("ANATOMY OF AN EFFECTIVE PROMPT")
    print("=" * 80)
    
    components = {
        "ROLE/CONTEXT": {
            "purpose": "Set the AI's perspective and expertise level",
            "examples": [
                "You are a senior software engineer with 15 years of experience",
                "You are a kindergarten teacher explaining concepts to 5-year-olds",
                "You are a professional writer specializing in science communication"
            ],
            "importance": "HIGH - Dramatically affects output quality and style"
        },
        
        "TASK": {
            "purpose": "Clear action verb and what needs to be done",
            "examples": [
                "Write a comprehensive guide about...",
                "Generate ideas for...",
                "Analyze and compare..."
            ],
            "importance": "CRITICAL - Without this, AI is directionless"
        },
        
        "CONTEXT": {
            "purpose": "Background information and constraints",
            "examples": [
                "This is for a company with 50 employees in the tech industry",
                "Target audience: college students with limited budget",
                "Time-sensitive: needed by end of week"
            ],
            "importance": "HIGH - Helps AI understand real-world constraints"
        },
        
        "REQUIREMENTS": {
            "purpose": "Specific things that MUST be included",
            "examples": [
                "Must include at least 3 examples",
                "Should reference recent studies from 2023-2024",
                "Must not mention competitors by name"
            ],
            "importance": "HIGH - Ensures output meets your needs"
        },
        
        "FORMAT": {
            "purpose": "How the output should be structured",
            "examples": [
                "Format as a JSON object with keys: title, summary, bullets",
                "Use markdown with clear headings",
                "Provide as a numbered list with explanations"
            ],
            "importance": "MEDIUM - Makes parsing/using output easier"
        },
        
        "TONE/STYLE": {
            "purpose": "How the response should sound",
            "examples": [
                "Professional and formal",
                "Casual and humorous",
                "Inspirational and motivational"
            ],
            "importance": "MEDIUM - Affects readability and appropriateness"
        },
        
        "LENGTH": {
            "purpose": "Control output size",
            "examples": [
                "Keep response under 200 words",
                "Provide a comprehensive guide (1000-1500 words)",
                "One sentence summary"
            ],
            "importance": "MEDIUM - Prevents too-long or too-short responses"
        },
        
        "EXAMPLES": {
            "purpose": "Few-shot learning - show what you want",
            "examples": [
                "Example input: 'The sky is blue' → Sentiment: Positive",
                "Example output format: [main_point], [supporting_detail], [conclusion]"
            ],
            "importance": "HIGH (when needed) - Dramatically improves output consistency"
        }
    }
    
    for component, details in components.items():
        print(f"\n{component}")
        print("-" * 80)
        print(f"Purpose: {details['purpose']}")
        print(f"Importance: {details['importance']}")
        print(f"Examples:")
        for example in details['examples']:
            print(f"  • {example}")


# ============================================================================
# EXAMPLE 4: PROMPT OPTIMIZATION CHECKLIST
# ============================================================================
def optimization_checklist():
    """
    Checklist for evaluating and improving prompts
    """
    print("\n" + "=" * 80)
    print("PROMPT OPTIMIZATION CHECKLIST")
    print("=" * 80)
    
    checklist = {
        "CLARITY": [
            "Is the main task clearly stated?",
            "Is there ambiguity that could lead to multiple interpretations?",
            "Would a non-expert understand what's being asked?",
            "Are technical terms defined?"
        ],
        
        "COMPLETENESS": [
            "Is the role/perspective specified?",
            "Are all requirements listed?",
            "Is the target audience clear?",
            "Is the desired output format specified?"
        ],
        
        "SPECIFICITY": [
            "Are vague terms replaced with concrete details?",
            "Are quantitative specifications included (length, count, etc.)?",
            "Are edge cases or special situations addressed?",
            "Are constraints clearly stated?"
        ],
        
        "STRUCTURE": [
            "Is the prompt logically organized?",
            "Are instructions presented in clear sections?",
            "Is the most important information first?",
            "Are examples provided where helpful?"
        ],
        
        "TESTABILITY": [
            "Is it clear what 'good output' looks like?",
            "Can you verify if requirements were met?",
            "Is the output measurable/checkable?",
            "Would different people evaluate similarly?"
        ],
        
        "EFFICIENCY": [
            "Are there unnecessary words?",
            "Could any requirements be combined?",
            "Is there conflicting guidance?",
            "Is the prompt concise but complete?"
        ]
    }
    
    for category, items in checklist.items():
        print(f"\n{category}")
        print("-" * 80)
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")


# ============================================================================
# EXAMPLE 5: COMMON MISTAKES
# ============================================================================
def common_mistakes():
    """
    Show and explain common prompt engineering mistakes
    """
    print("\n" + "=" * 80)
    print("COMMON PROMPT MISTAKES & FIXES")
    print("=" * 80)
    
    mistakes = {
        "1. TOO VAGUE": {
            "bad": "Write about marketing",
            "problems": [
                "No target audience",
                "No format",
                "No length",
                "Unclear scope"
            ],
            "good": "Write a 500-word blog post about email marketing for e-commerce businesses. Include 3 actionable tips and examples.",
            "lesson": "Specificity is your friend"
        },
        
        "2. CONTRADICTORY REQUIREMENTS": {
            "bad": "Write a fun, humorous article that's also very serious and academic",
            "problems": [
                "Can't be both fun AND serious",
                "Model will pick one and ignore other",
                "Output will feel confused"
            ],
            "good": "Write an academic article on AI with a lighthearted introduction. Tone: informative but accessible",
            "lesson": "Ensure requirements don't conflict"
        },
        
        "3. NO FORMAT SPECIFICATION": {
            "bad": "Give me ideas for a marketing campaign",
            "problems": [
                "Output might be prose or bullets",
                "Hard to parse results",
                "Takes more time to use output"
            ],
            "good": "Give me 5 marketing campaign ideas formatted as: [TITLE] - [1 sentence description] - [Target audience]",
            "lesson": "Always specify desired format"
        },
        
        "4. ASSUMING KNOWLEDGE": {
            "bad": "Explain our product features to customers",
            "problems": [
                "AI doesn't know your product",
                "Might make up or generalize features",
                "Tone might be off"
            ],
            "good": "Explain these product features [LIST FEATURES HERE] to small business owners who are tech-savvy but new to this category. Use conversational tone.",
            "lesson": "Provide needed context explicitly"
        },
        
        "5. NO EXAMPLES FOR COMPLEX TASKS": {
            "bad": "Classify these reviews by sentiment",
            "problems": [
                "What's your definition of positive/negative?",
                "Neutral vs mixed vs sarcasm unclear",
                "Output format unknown"
            ],
            "good": """Classify each review as Positive, Neutral, or Negative:
Examples:
- 'Amazing product!' → Positive
- 'It works fine' → Neutral  
- 'Total waste of money' → Negative
- 'Good but expensive' → Mixed (note as such)
[Now classify: ...]""",
            "lesson": "Use few-shot examples for complex tasks"
        }
    }
    
    for mistake, details in mistakes.items():
        print(f"\n{mistake}")
        print("-" * 80)
        print(f"❌ Bad: {details['bad']}")
        print(f"\nProblems:")
        for problem in details['problems']:
            print(f"  • {problem}")
        print(f"\n✓ Better: {details['good']}")
        print(f"\nLesson: {details['lesson']}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PROMPT ENGINEERING FUNDAMENTALS" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    demonstrate_prompt_quality()
    prompt_templates()
    prompt_components_breakdown()
    optimization_checklist()
    common_mistakes()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. CLARITY IS KEY: Vague prompts lead to unpredictable results
2. BE SPECIFIC: Include all relevant details and constraints
3. PROVIDE CONTEXT: Help the AI understand your needs
4. SHOW EXAMPLES: Few-shot learning dramatically improves results
5. SPECIFY FORMAT: Tell AI exactly how you want the output structured
6. DEFINE ROLE: Set appropriate expertise/perspective level
7. TEST ITERATIVELY: Refine based on actual output quality
8. LEARN FROM MISTAKES: Common errors are learning opportunities

GOLDEN RULE: Write prompts the way you'd explain a task to a human expert
    """)

