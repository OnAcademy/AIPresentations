# Section 6: Transformers

## Concepts Covered

1. **Attention Mechanism**
   - Self-attention
   - Multi-head attention
   - Scaled dot-product attention

2. **Transformer Architecture**
   - Encoder-Decoder structure
   - Positional encoding
   - Feed-forward networks

3. **Popular Models**
   - BERT (Bidirectional)
   - GPT (Unidirectional)
   - RoBERTa, ELECTRA
   - T5, BART

4. **Vision Transformers**
   - ViT (Vision Transformer)
   - Image patches
   - Multimodal models (CLIP)

5. **Fine-tuning & Transfer Learning**
   - Adapter modules
   - LoRA (Low-Rank Adaptation)
   - Domain-specific fine-tuning

## Why Transformers?

✓ **Parallelizable**: Unlike RNNs, can process entire sequence at once
✓ **Long-range dependencies**: Attention can focus on any position
✓ **Transfer learning**: Pre-trained models work well
✓ **Scalable**: Works with very large models and datasets
✓ **Interpretable**: Attention weights show model focus

## Transformer Architecture

```
Input Sequence
    ↓
Positional Encoding (add position information)
    ↓
Multi-Head Self-Attention (attend to all positions)
    ↓
Add & Norm
    ↓
Feed-Forward Network
    ↓
Add & Norm
    ↓
Output
```

## Key Models

| Model | Type | Size | Task |
|-------|------|------|------|
| BERT | Encoder | 110M-340M | Classification, NER, QA |
| GPT-2 | Decoder | 117M-1.5B | Text generation |
| GPT-3 | Decoder | 175B | Few-shot learning |
| T5 | Encoder-Decoder | 60M-11B | Text-to-text |
| BART | Encoder-Decoder | 139M | Summarization, translation |

## Files in This Section

- `transformer_basics.py` - Understanding attention
- `bert_classification.py` - BERT for text classification
- `gpt_generation.py` - GPT for text generation
- `vision_transformer.py` - ViT for image classification
- `fine_tuning.py` - Fine-tuning pre-trained models
- `multi_gpu.py` - Distributed training

## Attention Mechanism Explained

```
Query (Q): What are we looking for?
Key (K): What information do we have?
Value (V): What information should we retrieve?

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

## Quick Example: BERT Classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Tokenize input
inputs = tokenizer("This movie is great!", return_tensors="pt")

# Get predictions
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

## Multi-Head Attention

Benefits of using multiple attention heads:
- Different heads learn different relationships
- Parallel computation
- More expressive than single head
- Typically 8-12 heads per layer

## Vision Transformer (ViT)

Innovation: Apply Transformers to image classification

```
Image (224x224x3)
    ↓
Patch Embedding (16x16 patches → 196 patches)
    ↓
Linear projection to token embeddings
    ↓
Add class token and positional embeddings
    ↓
Transformer Encoder
    ↓
Classification head on [CLS] token
```

## Pre-training vs Fine-tuning

### Pre-training (Expensive)
- Train on large corpus
- Learn general language understanding
- Takes weeks/months on GPU
- Done once per model

### Fine-tuning (Fast)
- Start with pre-trained weights
- Train on task-specific data
- Takes hours to days
- Done for each downstream task

## Transfer Learning with Transformers

```python
# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=10  # Change output for your task
)

# Freeze earlier layers (optional)
for param in model.bert.encoder.layer[:9].parameters():
    param.requires_grad = False

# Fine-tune on your data
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.fit(train_loader, val_loader, epochs=3)
```

## Efficient Fine-tuning

### LoRA (Low-Rank Adaptation)
- Add small trainable matrices
- Freeze pre-trained weights
- 99% parameter reduction
- Minimal accuracy loss

### Adapter Modules
- Small bottleneck layers
- Trained efficiently
- Transfer between tasks
- Only 0.5-1% added parameters

## Common Applications

1. **Text Classification**: Sentiment, toxic content, spam
2. **Named Entity Recognition**: Extract names, dates, locations
3. **Question Answering**: Find answers in passages
4. **Machine Translation**: English to other languages
5. **Text Summarization**: Condense long texts
6. **Image Classification**: ViT for images
7. **Multimodal**: CLIP for image-text matching

## Performance Comparison

```
Task: Sentiment Analysis (Movie Reviews)

Model          Accuracy    Training Time
Baseline       85%         N/A
BERT (frozen)  88%         1 min
BERT (tuned)   92%         10 min
DistilBERT     90%         5 min (faster, smaller)
RoBERTa        93%         15 min
```

## Common Issues & Solutions

❌ **Out of memory**: Use smaller models, gradient accumulation, or mixed precision
❌ **Overfitting**: Add more data, use dropout, regularization
❌ **Slow inference**: Quantize, distill, or use smaller models
❌ **Poor performance**: Try different pre-trained models, more epochs, better hyperparameters

## Optimization Techniques

1. **Knowledge Distillation**: Smaller models learn from larger
2. **Quantization**: Reduce precision (int8 instead of float32)
3. **Pruning**: Remove less important weights
4. **ONNX Export**: Convert to optimized format

## Model Sizes & Trade-offs

| Size | Parameters | Speed | Accuracy | Memory |
|------|-----------|-------|----------|--------|
| Tiny | 4M | ★★★★★ | ★★ | Minimal |
| Small | 30M | ★★★★ | ★★★ | Low |
| Base | 110M | ★★★ | ★★★★ | Medium |
| Large | 340M | ★★ | ★★★★★ | High |

## Next Steps

1. Understand attention mechanism
2. Use pre-trained BERT for classification
3. Fine-tune on custom dataset
4. Explore Vision Transformers
5. Learn about efficient methods
6. Combine with other architectures

