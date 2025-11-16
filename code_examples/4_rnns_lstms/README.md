# Section 4: Recurrent Neural Networks (RNNs) & LSTMs

## Concepts Covered

1. **RNN Fundamentals**
   - Sequential data processing
   - Hidden state and recurrence
   - Vanishing/exploding gradients

2. **LSTM Architecture**
   - Memory cells
   - Gates (input, forget, output)
   - Cell state vs hidden state

3. **GRU (Gated Recurrent Unit)**
   - Simpler than LSTM
   - Similar performance
   - Fewer parameters

4. **Applications**
   - Natural Language Processing
   - Time series prediction
   - Speech recognition
   - Machine translation

5. **Sequence-to-Sequence**
   - Encoder-decoder architecture
   - Attention mechanisms
   - Beam search decoding

## RNN vs LSTM vs GRU

| Aspect | RNN | LSTM | GRU |
|--------|-----|------|-----|
| Memory | Basic hidden state | Cell state + hidden state | Combined state |
| Complexity | Low | High | Medium |
| Parameters | Few | Many | Medium |
| Vanishing Gradient | Susceptible | Robust | Robust |
| Training Speed | Fast | Slow | Medium |
| Best For | Simple sequences | Long sequences | Balanced choice |

## Files in This Section

- `rnn_basics.py` - Simple RNN implementation
- `lstm_sequence.py` - LSTM for sequence modeling
- `time_series.py` - Stock price/weather prediction
- `text_generation.py` - Generating text with RNNs
- `machine_translation.py` - Seq2seq translation models

## LSTM Architecture

```
Input: x_t
    ↓
Forget Gate: Controls what to forget
    ↓
Input Gate: Controls what new info to add
    ↓
Cell State Update: Add new information
    ↓
Output Gate: Decide what to output
    ↓
Hidden State: h_t
```

## Quick Example: Time Series

```python
import tensorflow as tf
import numpy as np

# Create LSTM model for time series
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(timesteps, features), return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train on time series data
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict future values
future = model.predict(X_test)
```

## Key Concepts

### Sequence Problems
- **Many-to-One**: Sentiment analysis, classification
- **One-to-Many**: Image captioning, text generation
- **Many-to-Many**: Machine translation, NER
- **Many-to-Many (aligned)**: Part-of-speech tagging

### Vanishing Gradient Problem
- Gradients become very small in deep networks
- LSTM solves with memory cell and gates
- Allows learning long-term dependencies

### Attention Mechanism
- Let model focus on relevant parts
- Improved translation quality
- Foundation for Transformers

## Applications in Detail

### 1. Stock Price Prediction
- Input: Historical prices
- Output: Future price
- Features: Open, high, low, close, volume

### 2. Language Modeling
- Input: Previous words
- Output: Next word
- Used in: Autocomplete, text generation

### 3. Machine Translation
- Encoder: Convert source sentence to context
- Decoder: Generate target sentence
- Example: English → French

### 4. Speech Recognition
- Input: Audio features (MFCC)
- Output: Text transcription
- Used by: Siri, Google Assistant

## Common Challenges

❌ **Vanishing Gradient**: Solution - LSTM/GRU gates
❌ **Exploding Gradient**: Solution - Gradient clipping
❌ **Slow Training**: Solution - Better architectures (Transformers)
❌ **Limited Context**: Solution - Attention mechanisms

## Advanced Topics

- **Bidirectional RNNs**: Process sequence both directions
- **Multi-layer RNNs**: Stack multiple RNN layers
- **Attention-based Seq2Seq**: Improved machine translation
- **Transformers**: Modern replacement for RNNs

## Learning Path

1. Understand RNN basics and recurrence
2. Study LSTM architecture and gates
3. Implement simple RNN for sequences
4. Build LSTM for time series prediction
5. Learn attention mechanism
6. Explore Transformers (next section)

## Performance Tips

✓ Normalize input data (0-1 range)
✓ Use smaller hidden sizes (64-128)
✓ Gradient clipping (norm 1.0)
✓ Dropout for regularization (0.2-0.5)
✓ Use checkpointing to save best model

## Next Steps

1. Understand sequence types
2. Build time series predictor
3. Implement text generation
4. Learn about attention
5. Move to Transformers for better results

