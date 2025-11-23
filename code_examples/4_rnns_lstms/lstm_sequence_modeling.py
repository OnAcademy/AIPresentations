"""
Recurrent Neural Networks (RNNs) and LSTMs: Sequence Modeling
Complete implementation for sequence processing, time series, and language modeling
Demonstrates: RNN basics, LSTM, GRU, and practical applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXAMPLE 1: RNN ARCHITECTURE EXPLANATION
# ============================================================================
def explain_rnn_basics():
    """
    Explain RNN fundamentals and why they're needed
    """
    print("=" * 80)
    print("RECURRENT NEURAL NETWORKS (RNNs) - FUNDAMENTALS")
    print("=" * 80)
    
    explanation = {
        "Why RNNs?": {
            "Problem": "CNNs work on fixed-size inputs (images)",
            "Challenge": "How to process sequences of variable length?",
            "Examples": [
                "Text (words in sequence)",
                "Time series (stock prices over time)",
                "Audio (sound frames in sequence)",
                "Video (frames in sequence)"
            ],
            "Solution": "Recurrent connections: h_t = f(h_{t-1}, x_t)"
        },
        
        "Key Idea": {
            "Concept": "Process sequence one element at a time",
            "Memory": "Hidden state carries information from past",
            "Formula": "h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)",
            "Output": "y_t = W_hy * h_t + b_y"
        },
        
        "Unrolling in Time": {
            "Concept": "Same network applied multiple times",
            "Training": "Backpropagation Through Time (BPTT)",
            "Gradient": "∂L/∂W = Σ(∂L/∂y_t * ∂y_t/∂h_t * ∂h_t/∂W)"
        },
        
        "Pros": [
            "✓ Handles variable-length sequences",
            "✓ Captures temporal dependencies",
            "✓ Same weights throughout sequence",
            "✓ Can process arbitrary length inputs"
        ],
        
        "Cons": [
            "✗ Vanishing gradient problem (hard to learn long-term dependencies)",
            "✗ Exploding gradient problem",
            "✗ Computationally expensive",
            "✗ Hard to parallelize"
        ]
    }
    
    for section, content in explanation.items():
        print(f"\n{section}:")
        print("-" * 70)
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    {item}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                print(f"  {item}")


# ============================================================================
# EXAMPLE 2: LSTM AND GRU
# ============================================================================
def explain_lstm_gru():
    """
    Explain LSTM and GRU architectures
    """
    print("\n" + "=" * 80)
    print("LSTM vs GRU - SOLVING THE VANISHING GRADIENT PROBLEM")
    print("=" * 80)
    
    lstm_info = {
        "LSTM (Long Short-Term Memory)": {
            "Inventor": "Hochreiter & Schmidhuber (1997)",
            "Problem Solved": "Vanishing gradient problem in RNNs",
            
            "Architecture": {
                "Input Gate": "i_t = σ(W_ii*x_t + W_hi*h_{t-1} + b_i)",
                "Forget Gate": "f_t = σ(W_if*x_t + W_hf*h_{t-1} + b_f)",
                "Cell Gate": "g_t = tanh(W_ig*x_t + W_hg*h_{t-1} + b_g)",
                "Output Gate": "o_t = σ(W_io*x_t + W_ho*h_{t-1} + b_o)",
                
                "Cell State Update": "c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t",
                "Hidden State": "h_t = o_t ⊙ tanh(c_t)"
            },
            
            "Key Innovation": [
                "Cell state (memory) with additive updates",
                "Gates control information flow",
                "Gradient can flow unchanged through cell state"
            ],
            
            "Parameters": "4 gates × 3 weight matrices = 12 matrices",
            "Complexity": "Higher computational cost",
            "Accuracy": "Superior for long sequences"
        },
        
        "GRU (Gated Recurrent Unit)": {
            "Inventor": "Cho et al. (2014)",
            "Simplification": "Simplified LSTM with fewer gates",
            
            "Architecture": {
                "Reset Gate": "r_t = σ(W_ir*x_t + W_hr*h_{t-1} + b_r)",
                "Update Gate": "z_t = σ(W_iz*x_t + W_hz*h_{t-1} + b_z)",
                "Candidate": "h'_t = tanh(W_ih*x_t + W_hh*(r_t ⊙ h_{t-1}) + b_h)",
                "Hidden State": "h_t = (1 - z_t) ⊙ h'_t + z_t ⊙ h_{t-1}"
            },
            
            "Advantages": [
                "Fewer parameters (2/3 of LSTM)",
                "Faster training",
                "Similar performance to LSTM"
            ],
            
            "Use Case": "When training speed matters or data is limited"
        }
    }
    
    for model_name, details in lstm_info.items():
        print(f"\n{model_name}:")
        print("-" * 70)
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    • {k}: {v}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 3: SIMPLE RNN IMPLEMENTATION FROM SCRATCH
# ============================================================================
def simple_rnn_from_scratch():
    """
    Implement simple RNN from scratch
    """
    print("\n" + "=" * 80)
    print("SIMPLE RNN IMPLEMENTATION FROM SCRATCH")
    print("=" * 80)
    
    class SimpleRNN:
        """Minimal RNN implementation"""
        
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            # Initialize weights
            self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
            self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
            self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
            self.bh = np.zeros((hidden_size, 1))                         # hidden bias
            self.by = np.zeros((output_size, 1))                         # output bias
            
            self.hidden_size = hidden_size
        
        def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Forward pass through sequence
            X: (sequence_length, input_size)
            Returns: (outputs, hidden_states)
            """
            seq_len, input_size = X.shape
            h = np.zeros((self.hidden_size, 1))
            
            outputs = []
            hidden_states = [h.copy()]
            
            for t in range(seq_len):
                x_t = X[t].reshape(-1, 1)  # (input_size, 1)
                
                # Update hidden state
                h = np.tanh(self.Wxh @ x_t + self.Whh @ h + self.bh)
                hidden_states.append(h.copy())
                
                # Compute output
                y_t = self.Why @ h + self.by
                outputs.append(y_t)
            
            return np.array(outputs).squeeze(), np.array(hidden_states)
    
    # Example: Character-level sequence
    print("\nExample: Processing sequence of digits")
    print("-" * 70)
    
    # Create simple data: [1, 2, 3, 4, 5]
    sequence = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
    
    rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
    outputs, hidden_states = rnn.forward(sequence)
    
    print(f"Input sequence: {sequence.squeeze()}")
    print(f"Output shape: {outputs.shape}")
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"First hidden state shape: {hidden_states[0].shape}")
    print("\n✓ Simple RNN forward pass completed successfully!")


# ============================================================================
# EXAMPLE 4: LSTM WITH TENSORFLOW
# ============================================================================
def lstm_with_tensorflow():
    """
    Build and train LSTM using TensorFlow
    """
    print("\n" + "=" * 80)
    print("LSTM WITH TENSORFLOW/KERAS")
    print("=" * 80)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        print("✓ TensorFlow imported successfully")
        
        # Example 1: Sequence generation (predicting next number in sequence)
        print("\nExample 1: Sequence Generation")
        print("-" * 70)
        
        # Generate simple sequences: 1,2,3,4,5 → 6
        def generate_sequences(sequence_length=5, num_samples=100):
            X, y = [], []
            for i in range(num_samples):
                start = np.random.randint(0, 100)
                seq = np.arange(start, start + sequence_length)
                X.append(seq.reshape(-1, 1))
                y.append(seq[-1] + 1)  # Next number
            return np.array(X), np.array(y)
        
        X_train, y_train = generate_sequences()
        X_test, y_test = generate_sequences()
        
        print(f"Generated training data: X={X_train.shape}, y={y_train.shape}")
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(32, activation='relu', input_shape=(5, 1), name='lstm1'),
            layers.Dense(16, activation='relu', name='dense1'),
            layers.Dense(1, name='output')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("\nLSTM Model for Sequence Prediction:")
        model.summary()
        
        # Train model
        print("\nTraining LSTM...")
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, 
                          validation_split=0.2, verbose=1)
        
        # Evaluate
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest MAE: {test_mae:.4f}")
        
        # Make predictions
        print("\nSample Predictions:")
        predictions = model.predict(X_test[:5], verbose=0)
        for i in range(5):
            print(f"  Sequence {X_test[i].squeeze()} → Predicted: {predictions[i][0]:.1f}, "
                  f"Actual: {y_test[i]}")
        
        # Example 2: Text generation (character-level)
        print("\n" + "-" * 70)
        print("Example 2: Character-Level Language Modeling")
        print("-" * 70)
        
        # Simple text: repeating pattern
        text = "abcdabcdabcdabcd"
        char_to_idx = {c: i for i, c in enumerate(set(text))}
        idx_to_char = {i: c for c, i in char_to_idx.items()}
        
        print(f"Text: {text}")
        print(f"Character mapping: {char_to_idx}")
        
        # Create sequences
        seq_length = 3
        X_char, y_char = [], []
        for i in range(len(text) - seq_length):
            X_char.append([char_to_idx[c] for c in text[i:i+seq_length]])
            y_char.append(char_to_idx[text[i+seq_length]])
        
        X_char = np.array(X_char)
        y_char = np.array(y_char)
        
        print(f"Sequences: X={X_char.shape}, y={y_char.shape}")
        
        # Build character-level LSTM
        char_model = keras.Sequential([
            layers.Embedding(len(char_to_idx), 8, input_length=seq_length),
            layers.LSTM(16, activation='relu'),
            layers.Dense(len(char_to_idx), activation='softmax')
        ])
        
        char_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                          metrics=['accuracy'])
        
        print("\nCharacter-Level LSTM Model:")
        char_model.summary()
        
        print("\nTraining character model...")
        char_model.fit(X_char, y_char, epochs=5, batch_size=2, verbose=1)
        
        # Generate text
        print("\nGenerating text...")
        seed_sequence = [char_to_idx[c] for c in "abc"]
        generated = "abc"
        
        for _ in range(10):
            prediction = char_model.predict(np.array([seed_sequence]), verbose=0)
            next_idx = np.argmax(prediction[0])
            generated += idx_to_char[next_idx]
            seed_sequence = seed_sequence[1:] + [next_idx]
        
        print(f"Generated: {generated}")
        
        return model, char_model
        
    except ImportError:
        print("⚠ TensorFlow not installed. Install with: pip install tensorflow")
        return None, None


# ============================================================================
# EXAMPLE 5: BIDIRECTIONAL RNNs
# ============================================================================
def explain_bidirectional_rnn():
    """
    Explain bidirectional RNNs
    """
    print("\n" + "=" * 80)
    print("BIDIRECTIONAL RNNs")
    print("=" * 80)
    
    explanation = """
CONCEPT: Process sequence in both directions

WHY BIDIRECTIONAL?
• Forward RNN: left-to-right context
• Backward RNN: right-to-left context
• Combined: full context from both sides

USE CASES:
✓ Machine translation (better context for each word)
✓ Named entity recognition (context from both sides)
✓ Part-of-speech tagging
✓ Sentiment analysis

ARCHITECTURE:
                    Forward LSTM (→)
                          ↓
    Input Sequence → → → → → → → → 
                          ↓
                    Backward LSTM (←)
                          ↓
                    Concatenate outputs
                          ↓
                    Output (bidirectional context)

EXAMPLE IN TENSORFLOW:
    from tensorflow.keras.layers import Bidirectional, LSTM
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(num_classes, activation='softmax')
    ])

OUTPUT SHAPE:
• Forward LSTM: (batch, seq_len, hidden_size)
• Backward LSTM: (batch, seq_len, hidden_size)
• Concatenated: (batch, seq_len, 2*hidden_size)

BENEFITS:
• Better accuracy for many NLP tasks
• 10-20% improvement over unidirectional in many cases
• Especially good for classification tasks

DRAWBACKS:
• 2x more parameters
• Can't be used for real-time generation (need future context)
• Slower training
"""
    
    print(explanation)


# ============================================================================
# EXAMPLE 6: APPLICATIONS
# ============================================================================
def rnn_applications():
    """
    Explain real-world RNN applications
    """
    print("\n" + "=" * 80)
    print("RNN APPLICATIONS IN THE WILD")
    print("=" * 80)
    
    applications = {
        "Machine Translation": {
            "System": "Google Translate (before Transformers)",
            "Technique": "Sequence-to-Sequence with Attention",
            "Input": "English sentence",
            "Output": "French sentence",
            "Accuracy": "~90% on news domain"
        },
        
        "Speech Recognition": {
            "System": "Apple Siri, Google Assistant",
            "Technique": "RNN + CTC (Connectionist Temporal Classification)",
            "Input": "Audio frames",
            "Output": "Text transcription",
            "Accuracy": "~95% on clean speech"
        },
        
        "Stock Price Prediction": {
            "System": "Financial institutions",
            "Technique": "LSTM on time series",
            "Input": "Historical prices + indicators",
            "Output": "Next price prediction",
            "Challenge": "Non-stationary, highly volatile"
        },
        
        "Text Generation": {
            "System": "ChatGPT (initially GPT used RNNs)",
            "Technique": "Character-level or word-level LSTM",
            "Input": "Sequence of tokens",
            "Output": "Next token (generates text)",
            "Applications": "Autocomplete, code generation"
        },
        
        "Medical Monitoring": {
            "System": "Heart rate, ECG monitoring",
            "Technique": "LSTM for anomaly detection",
            "Input": "Time series of vital signs",
            "Output": "Alert if abnormal pattern",
            "Importance": "Early disease detection"
        },
        
        "Video Understanding": {
            "System": "Action recognition in video",
            "Technique": "3D CNN + RNN (two-stream)",
            "Input": "Video frames",
            "Output": "Action label",
            "Example": "Is person 'jumping' or 'walking'?"
        }
    }
    
    for app_name, details in applications.items():
        print(f"\n{app_name}:")
        print("-" * 70)
        for key, value in details.items():
            print(f"  {key}: {value}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("RECURRENT NEURAL NETWORKS & LSTM")
    print("Complete Guide: Sequence Modeling, Time Series, and Language")
    print("🎯" * 40)
    
    # Run demonstrations
    explain_rnn_basics()
    explain_lstm_gru()
    simple_rnn_from_scratch()
    explain_bidirectional_rnn()
    rnn_applications()
    
    # Try TensorFlow if available
    models = lstm_with_tensorflow()
    
    print("\n" + "=" * 80)
    print("RNN TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\n📚 KEY TAKEAWAYS:")
    print("  ✓ RNNs process sequences with hidden state memory")
    print("  ✓ LSTMs solve vanishing gradient with cell state")
    print("  ✓ GRUs are simpler and faster variants")
    print("  ✓ Bidirectional RNNs improve accuracy with full context")
    print("  ✓ RNNs excel at sequence-to-sequence tasks")
    print("\n🚀 NEXT STEPS:")
    print("  1. Study Attention mechanisms")
    print("  2. Learn about Transformers (better for sequences)")
    print("  3. Try seq2seq models with attention")
    print("  4. Build your own language model")
    print("\n" + "=" * 80)

