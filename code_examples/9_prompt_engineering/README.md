# Section 9: Prompt Engineering

## Concepts Covered

1. **Prompt Fundamentals**
   - Clarity and Specificity
   - Context Provision
   - Role Assignment (Persona)

2. **Advanced Techniques**
   - Zero-Shot Learning
   - Few-Shot Learning
   - Chain-of-Thought (CoT) Prompting

3. **Prompt Structure**
   - System Prompts
   - User Messages
   - Examples
   - Output Format Specification

4. **Optimization Strategies**
   - Temperature Adjustment
   - Iterative Refinement
   - Testing and Measurement

## Files in This Section

- `basic_prompts.py` - Fundamental prompting techniques
- `few_shot_learning.py` - Few-shot examples and demonstrations
- `chain_of_thought.py` - Complex reasoning with step-by-step thinking
- `prompt_templates.py` - Reusable prompt templates
- `prompt_optimization.py` - Testing and improving prompts

## Prompt Quality Levels

### ❌ Bad Prompt
```
Write something about AI
```
**Problems**: Vague, no direction, unpredictable output

### ⚠️ Better Prompt
```
Write a 3-paragraph explanation of artificial intelligence for someone without technical background
```
**Better**: Has length, audience, but still lacks structure

### ✅ Excellent Prompt
```
You are an AI educator explaining concepts to high school students.
Write exactly 3 paragraphs about artificial intelligence that:
1. Defines what AI is in simple terms
2. Gives 2-3 real-world examples
3. Explains one limitation of current AI

Use conversational language and avoid technical jargon.
```
**Why it's excellent**: Role, requirements, format, constraints, all specified

## Key Techniques

### 1. **Zero-Shot Prompting**
Ask the model to solve a problem without examples

```
Translate this English text to Spanish: "Hello world"
```

### 2. **Few-Shot Prompting**
Provide examples before the actual task

```
Classify the sentiment of each review:

Review: "Amazing product!" → Sentiment: Positive
Review: "Terrible quality" → Sentiment: Negative
Review: "It's okay, nothing special" → Sentiment: Neutral

Review: "I love this!" → Sentiment: ?
```

### 3. **Chain-of-Thought (CoT)**
Ask the model to think step-by-step

```
Q: If I have 5 apples and give 2 to my friend, then buy 3 more, how many do I have?
A: Let me work through this step by step:
1. I start with 5 apples
2. I give 2 to my friend: 5 - 2 = 3 apples
3. I buy 3 more: 3 + 3 = 6 apples
Final answer: 6 apples
```

### 4. **Role Assignment (Persona)**
Assign a specific role to the model

```
You are an expert copywriter with 20 years of experience.
Write a compelling product description for...
```

## Prompt Structure Template

```
[ROLE]
You are a [specific role with relevant expertise].

[CONTEXT]
Background: [relevant information about the task/domain]
User background: [who is asking]

[TASK]
Your task is to [clear action verb] [object/content]

[REQUIREMENTS]
- Requirement 1
- Requirement 2
- Requirement 3

[OUTPUT FORMAT]
Format your response as: [specific format]

[EXAMPLES]
Example input: ...
Example output: ...

[CONSTRAINTS]
- Keep response under X words
- Use [specific tone/style]
- Avoid [specific topics/language]
```

## Techniques by Use Case

### Code Generation
✓ Use specific language and framework
✓ Request comments/documentation
✓ Ask for error handling
```
Write Python code to [task] using [library]
Include error handling and clear variable names
```

### Content Creation
✓ Specify tone, audience, length
✓ Provide outline/structure
✓ Give style examples
```
Write a blog post for [audience] about [topic]
Tone: [casual/formal/humorous]
Length: [word count]
Include: [key points]
```

### Analysis & Reasoning
✓ Request step-by-step thinking
✓ Ask for multiple perspectives
✓ Request structured output
```
Analyze [document/data]
Think step-by-step
Consider [specific angles]
Provide structured summary
```

### Creative Tasks
✓ Higher temperature (0.7-1.0)
✓ Fewer constraints
✓ Encourage multiple attempts
```
Generate 5 creative ideas for [topic]
Consider unconventional approaches
Be bold and imaginative
```

## Common Mistakes

❌ **Too Vague**: "Write about AI" - unpredictable output
❌ **No Role**: Not setting expectations for writing style
❌ **No Format**: Expecting structured output without specifying format
❌ **No Examples**: Using few-shot without providing examples
❌ **Contradictions**: Asking for both creative AND factual accuracy
❌ **No Constraints**: Allowing unbounded responses

## Testing Your Prompts

1. **Consistency**: Run the same prompt 3-5 times, check variation
2. **Accuracy**: Verify factual correctness of outputs
3. **Format**: Check if output matches specified format
4. **Relevance**: Does output address the actual question?
5. **Length**: Is output within specified constraints?

## Iterative Refinement Process

```
1. Write initial prompt
   ↓
2. Test with sample input
   ↓
3. Analyze output quality
   ↓
4. Identify issues
   ↓
5. Refine prompt
   ↓
6. Re-test
   ↓
7. Repeat until satisfied
```

## Real-World Examples

### Customer Support
```
You are a customer service representative for an e-commerce company.
A customer has written: [CUSTOMER_MESSAGE]
Respond with:
- Acknowledgment of their issue
- 2-3 possible solutions
- Friendly, professional tone
Keep under 200 words.
```

### Data Analysis
```
Analyze this dataset: [DATA]
Step 1: Identify key trends
Step 2: Highlight anomalies
Step 3: Suggest actions
Provide both summary and detailed findings.
```

### Creative Writing
```
Write a short story opening (100-150 words) about: [TOPIC]
Requirements:
- Engaging first sentence
- Create emotional hook
- Establish setting
- Tone: [TONE]
```

## Advanced Tips

- **Anchoring**: Start prompt with similar examples to "anchor" the model's understanding
- **Negative Examples**: Show what NOT to do
- **Thinking Process**: Ask model to explain its reasoning
- **Iteration**: Use model's previous response in next prompt
- **Constraints**: Use specific constraints (word count, format) to guide output

## Next Steps

1. Start with simple prompts, gradually add complexity
2. Keep a library of successful prompts
3. Test systematically and document results
4. Learn from failures and refine
5. Explore advanced techniques (chain-of-thought, function calling)
6. Build prompt templates for recurring tasks

