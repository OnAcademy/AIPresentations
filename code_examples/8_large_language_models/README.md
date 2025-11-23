# Large Language Models (LLMs) - Complete Guide

## Overview

Large Language Models (LLMs) are the foundation of modern AI applications like ChatGPT, Claude, and Gemini. This section covers how to use LLM APIs effectively.

## What is an LLM?

LLMs are neural networks trained on massive amounts of text (billions to trillions of tokens) to predict the next word in a sequence.

### Key Properties

- **Scale**: Billions to trillions of parameters
- **Training Data**: Internet-scale text corpora
- **Task**: Next-token prediction (language modeling)
- **Inference**: Generates text one token at a time

## Famous LLM Models

### OpenAI Series
- **GPT-3.5**: 175B parameters, widely used
- **GPT-4**: More capable, better reasoning
- **GPT-4-Turbo**: Longer context, faster

### Google Series
- **Bard / Gemini**: Multimodal, integrated with Google services
- **PaLM 2**: Base model for Bard

### Meta Series
- **LLaMA**: Open-source, smaller models
- **LLaMA 2**: Improved version

### Anthropic
- **Claude**: Focus on safety and constitution
- **Claude 2**: 100K token context window

## Using LLM APIs

### OpenAI API

```python
import openai

client = openai.OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is AI?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Key Parameters

- **temperature**: Randomness (0=deterministic, 1=creative)
- **max_tokens**: Maximum response length
- **top_p**: Nucleus sampling for diversity
- **frequency_penalty**: Reduce repetition
- **presence_penalty**: Encourage new topics

## Cost Considerations

- Pricing per token (input/output)
- OpenAI GPT-3.5: ~$0.0005-0.001 per 1K tokens
- GPT-4: ~$0.03-0.06 per 1K tokens
- Monitor usage to control costs

## Best Practices

1. **Prompt Engineering**: Clear, specific instructions
2. **Temperature**: Lower for factual, higher for creative
3. **Error Handling**: Handle API failures gracefully
4. **Caching**: Store responses to reduce API calls
5. **Rate Limiting**: Respect API rate limits

## Common Applications

- Chatbots and Q&A systems
- Content generation
- Code completion
- Text summarization
- Sentiment analysis
- Information extraction

## Code Examples

See `openai_api_example.py` for:
- Basic chat completion
- System prompts and personas
- Few-shot learning
- Text generation
- Streaming responses

## Further Reading

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Model Comparison](https://platform.openai.com/docs/models)
- [Papers with Code - LLMs](https://paperswithcode.com/area/language-models)
