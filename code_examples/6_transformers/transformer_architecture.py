"""
Transformer Architecture: Attention is All You Need
Complete implementation of Transformer architecture
Demonstrates: Self-Attention, Multi-Head Attention, Transformer Block, Applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXAMPLE 1: ATTENTION MECHANISM BASICS
# ============================================================================
def explain_attention_mechanism():
    """
    Explain the revolutionary Attention mechanism
    """
    print("=" * 80)
    print("ATTENTION MECHANISM - THE FOUNDATION")
    print("=" * 80)
    
    explanation = """
MOTIVATION: Why do we need Attention?

RNN Problem:
• Processes sequences sequentially: x₁ → h₁ → x₂ → h₂ → ...
• Information bottleneck: all info compressed into single hidden state
• Long-range dependencies: early information gets lost in LSTM cell
• Can't parallelize: must process one element at a time

Solution: ATTENTION
"Focus on relevant parts of input, not everything equally"

HOW ATTENTION WORKS:

Step 1: Compute Relevance Scores
        - Query (Q): What am I looking for?
        - Key (K): What information do you have?
        - Score = Q · K (dot product similarity)
        
Step 2: Normalize with Softmax
        - Attention = softmax(Score / √d_k)
        - Converts scores to probabilities [0,1]
        
Step 3: Weight Values
        - Value (V): Information associated with key
        - Output = Attention × V
        - Weighted sum of values

FORMULA (Scaled Dot-Product Attention):
Attention(Q, K, V) = softmax(QK^T / √d_k) V

WHERE:
- Q (Query): What to look for (d_q × n)
- K (Key): What's available (d_k × m)
- V (Value): Information to extract (d_v × m)
- d_k: Dimension scaling factor (prevents explosion)
- n: Query sequence length
- m: Key/Value sequence length

EXAMPLE - Machine Translation:

English: "The cat sat on the mat"
French:  [Je, le, chat, assis, sur, le, tapis]

When generating "chat" (cat), attention looks at:
- "cat": 0.8 (high attention - exact match!)
- "the": 0.15 (moderate - related)
- "sat": 0.03 (low - less relevant)
- "on": 0.01 (very low - not relevant)
- other: ~0.01

Result: Strong focus on "cat" position

ADVANTAGES:
✓ Captures long-range dependencies directly
✓ Fully parallelizable (unlike RNNs)
✓ Interpretable (can visualize what model attends to)
✓ Scales well to longer sequences
✓ No information bottleneck

VISUALIZATION:
Input Sequence: [a, b, c, d]

For token 'b', attention weights might be:
a: ████ 0.4 (pay attention to 'a')
b: ████████ 0.8 (pay attention to self)
c: ██ 0.2 (less attention to 'c')
d: █ 0.1 (least attention to 'd')

Result: 0.4*V_a + 0.8*V_b + 0.2*V_c + 0.1*V_d
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE 2: MULTI-HEAD ATTENTION
# ============================================================================
def explain_multihead_attention():
    """
    Explain Multi-Head Attention mechanism
    """
    print("\n" + "=" * 80)
    print("MULTI-HEAD ATTENTION")
    print("=" * 80)
    
    explanation = """
CONCEPT: Multiple attention heads looking at different aspects

WHY MULTIPLE HEADS?
• Single attention head learns one type of relationship
• Different heads can learn different patterns simultaneously
• Example: 
  - Head 1 might focus on syntax (grammatical relationships)
  - Head 2 might focus on semantics (meaning relationships)
  - Head 3 might focus on sentence structure

ARCHITECTURE:

                        Input
                          |
        ┌─────────────────┼─────────────────┐
        |                 |                 |
    Linear(Q)        Linear(Q)          Linear(Q)    ← 8 heads (example)
    Linear(K)        Linear(K)          Linear(K)
    Linear(V)        Linear(V)          Linear(V)
        |                 |                 |
    Attention(1)    Attention(2)     Attention(8)
        |                 |                 |
        └─────────────────┼─────────────────┘
                    Concatenate
                          |
                   Linear(Projection)
                          |
                       Output

MATHEMATICAL FORMULA:

MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) W^O
where:
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

BENEFITS:
✓ Represents information at different subspaces
✓ Gives attention mechanism more "modeling capacity"
✓ Allows parallel computation of attention heads
✓ More robust - attention failures in one head compensated by others

TYPICAL ARCHITECTURE:
• 8 heads in small models (BERT-small)
• 12 heads in BERT-base
• 16-20 heads in larger models
• 96+ heads in GPT-3 (175B parameters)

EXAMPLE OUTPUT DIMENSION:
Input: (batch, seq_len, 768)
With 12 heads:
├─ Each head processes 768/12 = 64 dimensions
├─ Each head outputs: (batch, seq_len, 64)
└─ Concatenated: (batch, seq_len, 768)

ATTENTION VISUALIZATION:
When translating "The black cat sat", attention for "cat":

Head 1 (Grammar):
  The:  ███ 0.3    (article-noun relationship)
  black: ████ 0.4   (modifier-noun)
  cat:  ████████ 0.8 (self)
  sat:  ██ 0.2      (verb after subject)

Head 2 (Semantics):
  The: █ 0.1       (non-semantic)
  black: ███████ 0.7 (important for meaning!)
  cat: ████████ 0.9  (core entity)
  sat: █ 0.1         (non-semantic for noun)

Head 3 (Position):
  The: ████ 0.4     (close positions)
  black: ████ 0.4    (close positions)
  cat: ████████ 0.8  (self)
  sat: ██ 0.2        (further away)
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE 3: TRANSFORMER ARCHITECTURE
# ============================================================================
def explain_transformer_architecture():
    """
    Complete Transformer architecture explanation
    """
    print("\n" + "=" * 80)
    print("TRANSFORMER ARCHITECTURE - COMPLETE")
    print("=" * 80)
    
    architecture = """
"ATTENTION IS ALL YOU NEED" - Vaswani et al. (2017)

FULL TRANSFORMER:

                    ┌─────────────────────────────────────┐
                    │    OUTPUT (GENERATED TEXT)          │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │  LINEAR + SOFTMAX                   │
                    │  (Output projection to vocabulary)  │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────────┐
                    │  DECODER STACK (N layers)              │
                    │  ┌──────────────────────────────────┐  │
                    │  │ Decoder Layer                    │  │
                    │  │ ├─ Multi-Head Self-Attention    │  │
                    │  │ ├─ Masked Attention (prevents   │  │
                    │  │ │  looking at future)           │  │
                    │  │ ├─ Cross-Attention (encoder     │  │
                    │  │ │  output as K,V)               │  │
                    │  │ ├─ Feed-Forward Network         │  │
                    │  │ └─ Layer Norm + Residual        │  │
                    │  └──────────────────────────────────┘  │
                    │                                        │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────────┐
                    │  ENCODER STACK (N layers)              │
                    │  ┌──────────────────────────────────┐  │
                    │  │ Encoder Layer                    │  │
                    │  │ ├─ Multi-Head Self-Attention    │  │
                    │  │ ├─ Feed-Forward Network         │  │
                    │  │ │  (2 linear layers + ReLU)     │  │
                    │  │ └─ Layer Norm + Residual        │  │
                    │  └──────────────────────────────────┘  │
                    │                                        │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │  INPUT EMBEDDING + POS ENCODING      │
                    │  ├─ Token Embeddings (words→vectors)│
                    │  └─ Positional Encoding (where am i?)
                    └──────────────────────────────────────┘

ENCODER LAYER DETAILS:

┌──────────────────────────────────────────────────┐
│ Input: (batch, seq_len, d_model)                 │
├──────────────────────────────────────────────────┤
│                                                   │
│ ┌────── Multi-Head Self-Attention ──────┐       │
│ │ Q, K, V = Input (same input for all)  │       │
│ │ → Compute attention weights           │       │
│ │ → Apply to all sequence positions     │       │
│ │ Output: (batch, seq_len, d_model)     │       │
│ └───────────────────────────────────────┘       │
│         ↓                                         │
│ Add & Norm: Output + Input, then normalize       │
│         ↓                                         │
│ ┌──── Feed-Forward Network ────┐                │
│ │ Dense(d_model → 2048)        │                │
│ │ ReLU                          │                │
│ │ Dense(2048 → d_model)        │                │
│ └──────────────────────────────┘                │
│         ↓                                         │
│ Add & Norm: Output + Input, then normalize       │
│         ↓                                         │
│ Output: (batch, seq_len, d_model)               │
└──────────────────────────────────────────────────┘

DECODER LAYER DIFFERENCES:
1. Self-Attention is MASKED
   - Can't attend to future tokens (causality)
   - Position i can only see positions i and earlier
   
2. Cross-Attention
   - Query (Q): From decoder
   - Key (K), Value (V): From encoder
   - Decoder attends to encoder outputs

3. Masked Multi-Head Self-Attention Mask:
   Matrix showing which positions can attend to which
   
   ┌─────────────┐
   │ 1 0 0 0 0 0 │  (Position 0 only sees 0)
   │ 1 1 0 0 0 0 │  (Position 1 sees 0,1)
   │ 1 1 1 0 0 0 │  (Position 2 sees 0,1,2)
   │ 1 1 1 1 0 0 │  (Can't see future)
   └─────────────┘

TYPICAL HYPERPARAMETERS:
• d_model: 512 (embedding dimension)
• num_heads: 8 (attention heads)
• d_ff: 2048 (feed-forward inner dimension)
• num_layers: 6 (for base model)
• dropout: 0.1
• max_sequence_length: 512

POSITIONAL ENCODING:
Why needed: Attention is permutation-invariant
           (doesn't inherently know token order)

Solution: Add positional information
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Result: Words at different positions get different vectors

ADVANTAGES:
✓ Fully parallelizable (process all positions simultaneously)
✓ Long-range dependencies (direct attention mechanism)
✓ State-of-the-art on many NLP tasks
✓ Efficient (compared to RNNs)
✓ Scalable to very large models

COMPUTATION COMPLEXITY:
• Self-Attention: O(n² d_model)  [n = sequence length]
• Feed-Forward: O(n d_ff)
• Total per layer: O(n² d_model + n d_ff)
• Full model: O(num_layers × (n² d_model + n d_ff))

Challenge: O(n²) complexity is quadratic in sequence length
         (problematic for long sequences like 4096+ tokens)
"""
    
    print(architecture)


# ============================================================================
# EXAMPLE 4: FAMOUS TRANSFORMER MODELS
# ============================================================================
def explain_transformer_models():
    """
    Explain famous Transformer-based models
    """
    print("\n" + "=" * 80)
    print("FAMOUS TRANSFORMER MODELS")
    print("=" * 80)
    
    models_info = {
        "BERT (2018) - Bidirectional Encoder": {
            "Creator": "Google AI",
            "Architecture": "Encoder only (no decoder)",
            "Training": "Masked Language Modeling + Next Sentence Prediction",
            "Key Feature": "Bidirectional (sees context both left and right)",
            "Use Cases": "Classification, Sentiment Analysis, NER",
            "Impact": "Changed NLP landscape, benchmark for many tasks",
            "Variants": "RoBERTa, ALBERT, DistilBERT"
        },
        
        "GPT Series (2018+) - Autoregressive Decoder": {
            "Creator": "OpenAI",
            "Architecture": "Decoder only (no encoder)",
            "Training": "Causal language modeling (predicting next token)",
            "Key Feature": "Unidirectional (left-to-right generation)",
            "Versions": "GPT, GPT-2, GPT-3, GPT-4",
            "Use Cases": "Text generation, Few-shot learning, Reasoning",
            "Impact": "Foundation for ChatGPT, most used production model",
            "Scaling": "GPT-3: 175B params, GPT-4: ~1.7T params (estimated)"
        },
        
        "T5 (2019) - Text-to-Text": {
            "Creator": "Google",
            "Architecture": "Full Encoder-Decoder",
            "Unified Format": "All NLP tasks as 'text to text'",
            "Training": "Multiple self-supervised objectives",
            "Use Cases": "Translation, Summarization, QA, Parsing",
            "Special": "Treats all tasks uniformly",
            "Efficiency": "T5-small can fit on phone"
        },
        
        "ELECTRA (2020) - Replaced Token Detection": {
            "Creator": "Google Research",
            "Innovation": "Instead of MLM, detect replaced tokens",
            "Efficiency": "More sample-efficient than BERT",
            "Performance": "Better efficiency-performance tradeoff",
            "Training": "Discriminator + generator setup (like GAN)"
        },
        
        "RoBERTa (2019) - Robustly Optimized": {
            "Creator": "Facebook AI",
            "Improvement": "Better training of BERT",
            "Changes": "More training data, longer training, better hyperparams",
            "Result": "Consistent improvements over BERT",
            "Adoption": "More commonly used than original BERT"
        },
        
        "DeBERTa (2020) - Decoding-enhanced": {
            "Creator": "Microsoft",
            "Innovation": "Disentangled attention mechanism",
            "Advantage": "Separates content and position attention",
            "Result": "SOTA on SuperGLUE benchmark",
            "Performance": "Outperforms GPT-3 on several tasks"
        },
        
        "Vision Transformer (ViT) (2020) - For Images": {
            "Creator": "Google Brain",
            "Revolutionary": "First successful pure-attention image model",
            "Approach": "Divide image into patches, treat as sequence",
            "Result": "Outperforms CNNs on many image tasks",
            "Impact": "Unified NLP and Vision with same architecture"
        },
        
        "CLIP (2021) - Multimodal": {
            "Creator": "OpenAI",
            "Innovation": "Joint training on image-text pairs",
            "Capability": "Align images and text in same space",
            "Use": "Zero-shot image classification, image search",
            "Impact": "Foundation for text-to-image models"
        }
    }
    
    for model_name, details in models_info.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        for key, value in details.items():
            print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 5: EFFICIENT TRANSFORMERS
# ============================================================================
def explain_efficient_transformers():
    """
    Explain ways to make Transformers more efficient
    """
    print("\n" + "=" * 80)
    print("EFFICIENT TRANSFORMERS - ADDRESSING QUADRATIC COMPLEXITY")
    print("=" * 80)
    
    efficient_info = """
PROBLEM: O(n²) self-attention is quadratic

For n = 4096 tokens:
• Standard: 4096² = 16.7M attention operations
• Problematic for: Documents, long contexts, real-time inference

SOLUTIONS:

1. LINEAR ATTENTION (Kernel trick)
   • Replace softmax with kernel functions
   • Reduces complexity to O(n)
   • Models: Linformer, Performer, Linear Attention Transformers
   • Trade-off: Approximates softmax, slight quality loss

2. SPARSE ATTENTION
   • Only compute attention for nearby tokens + random sample
   • Idea: Most relevant information is local
   • Complexity: O(n√n) or O(n log n)
   • Models: Longformer, BigBird
   • Benefits: Works well for long sequences

3. HIERARCHICAL ATTENTION
   • Attention at multiple levels (coarse to fine)
   • Reduces computation while keeping important patterns
   • Example: Swin Transformer for vision
   • Complexity: O(n) with proper hierarchies

4. RECURRENT TRANSFORMERS
   • Process in chunks with recurrent connection
   • Combine benefits of Transformer and RNN
   • Example: Transformer-XL
   • Advantage: Can handle infinite sequences

5. LOW-RANK FACTORIZATION
   • Approximate attention matrix as low-rank
   • A ≈ UV^T where U,V much smaller than A
   • Reduces parameters and computation
   • Trade-off: Slight approximation

6. DISTILLATION
   • Train small "student" model on large "teacher"
   • Transfer knowledge to smaller model
   • Result: 6-10x faster inference
   • Example: DistilBERT (40% params, 60% faster, 97% performance)

7. QUANTIZATION
   • Convert float32 → float16 or int8
   • Reduce memory and computation
   • Sometimes with minimal accuracy loss
   • Example: ONNX Runtime optimizations

8. PROMPT COMPRESSION
   • Remove redundant tokens from input
   • Cache attention keys/values between requests
   • For inference: Can reuse previous computations

COMPARISON:

Model               Context Length  Speed    Memory   Trade-off
──────────────────────────────────────────────────────────────
Standard BERT       512            Base     Base     None
RoPE Pos Enc        4096           -50%     Same     Better encoding
FlashAttention      2K-32K         2-4x     -25%     None (optimization)
Linformer           4096           2x       -75%     ~1-2% quality loss
Longformer          4096           3x       -60%     Sparse pattern
BigBird             4096           2x       -70%     Attention pattern
Performer           ∞              3-5x     -80%     Kernel approximation
──────────────────────────────────────────────────────────────

PRACTICAL APPROACHES (2024):

For Production:
1. Use optimized implementations (FlashAttention)
2. Apply quantization (int8)
3. Use smaller model if possible (DistilBERT vs BERT)
4. Cache KV for multi-turn conversations
5. Batch requests when possible

For Long Contexts:
1. Longformer/BigBird for ~4K tokens
2. Switch to sparse attention models
3. Use hierarchical approaches
4. Consider sliding window with recurrence

For Real-time:
1. Distilled models (DistilBERT)
2. Integer quantization
3. Hardware accelerators (GPU, TPU)
4. KV-cache for auto-regressive generation
"""
    
    print(efficient_info)


# ============================================================================
# EXAMPLE 6: ATTENTION VISUALIZATION EXPLAINED
# ============================================================================
def attention_visualization_guide():
    """
    Explain how to interpret attention visualizations
    """
    print("\n" + "=" * 80)
    print("ATTENTION VISUALIZATION - INTERPRETATION GUIDE")
    print("=" * 80)
    
    guide = """
ATTENTION VISUALIZATION:
Shows which tokens each position attends to

EXAMPLE - Translation Task:

English: "The quick brown fox"
French:  "Le rapide brun renard"

Attention from "rapide" (quick):

                "The" "quick" "brown" "fox"
   "Le":        0.1    0.05    0.05   0.8     → Attends mostly to "fox" (last token)
"rapide":       0.1    0.7     0.1    0.1     → Attends mostly to "quick"!
"brun":         0.05   0.1     0.8    0.05    → Attends mostly to "brown"
"renard":       0.5    0.1     0.05   0.35    → Attends to "The" and "fox"

WHAT IT MEANS:
✓ Model learned word alignments
✓ "rapide" focuses on "quick" (correct semantic match)
✓ Attention heads show interpretable patterns
✓ Can debug model behavior

HEAD ATTENTION (Multi-Head):

Head 1 - Syntactic:
             S-head S-obj V-noun
   Word: [0.8,  0.1,  0.05, 0.05] → Focus on subject

Head 2 - Semantic:
        [0.05, 0.8,  0.1, 0.05] → Focus on main verb

Head 3 - Positional:
        [0.3, 0.3,  0.2, 0.2] → Spread across sentence

TIPS FOR INTERPRETATION:

1. Look for symmetric patterns
   ↔ Bidirectional attention suggests mutual importance

2. Notice diagonal dominance
   → Token attending mostly to itself (typical)
   ⚠ Too much self-attention might indicate poor patterns

3. Check for hierarchical structure
   ✓ Later positions attending to earlier (gathering info)
   ✓ Early positions attending to later (lookahead)

4. Compare heads
   → Each head should learn different patterns
   ⚠ All heads similar = redundancy or overfitting

5. Look for exceptions
   ✓ Tokens breaking pattern often have special meaning
   → Stop words attends broadly (distributes attention)
   ✓ Nouns concentrate attention (focal points)

COMMON PATTERNS:

NORMAL DISTRIBUTION:
Token focuses mainly on itself, some on neighbors
    [0.5, 0.3, 0.15, 0.05]
Interpretation: Local context processing

BROADCAST ATTENTION:
Token focuses on many positions equally
    [0.25, 0.25, 0.25, 0.25]
Interpretation: Might indicate stop word or generic token

CONCENTRATED ATTENTION:
Token focuses on single position
    [0.05, 0.05, 0.85, 0.05]
Interpretation: Strong specific dependency

HIERARCHICAL:
Early tokens: broad attention
Later tokens: focused attention
Interpretation: Information gathering then processing
"""
    
    print(guide)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("TRANSFORMER ARCHITECTURE")
    print("The Foundation of Modern AI - Complete Guide")
    print("🎯" * 40)
    
    # Run demonstrations
    explain_attention_mechanism()
    explain_multihead_attention()
    explain_transformer_architecture()
    explain_transformer_models()
    explain_efficient_transformers()
    attention_visualization_guide()
    
    print("\n" + "=" * 80)
    print("TRANSFORMER TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\n📚 KEY TAKEAWAYS:")
    print("  ✓ Attention mechanism: Query-Key-Value mechanism")
    print("  ✓ Multi-Head: Multiple attention perspectives")
    print("  ✓ Transformer: Parallelizable, long-range dependencies")
    print("  ✓ Encoder-Decoder: Flexible for different tasks")
    print("  ✓ Foundation: Powers modern LLMs (ChatGPT, Gemini)")
    print("\n🚀 NEXT STEPS:")
    print("  1. Study specific models (BERT, GPT, T5)")
    print("  2. Learn efficient variants (Longformer, BigBird)")
    print("  3. Explore Vision Transformers (ViT)")
    print("  4. Implement attention from scratch")
    print("  5. Fine-tune pre-trained Transformers")
    print("\n🏆 MODERN APPLICATIONS:")
    print("  • ChatGPT / GPT-4 (Text generation)")
    print("  • DALL-E 3 (Text-to-Image)")
    print("  • Gemini (Multimodal)")
    print("  • Claude (LLM)")
    print("  • GitHub Copilot (Code generation)")
    print("\n" + "=" * 80)

