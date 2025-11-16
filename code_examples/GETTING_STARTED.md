# Getting Started with AI Code Examples

Welcome! This is your complete guide to running the code examples accompanying the IROST AI presentations.

## 📋 What You Have

A comprehensive collection of Python scripts and tutorials covering:
- **Deep Learning fundamentals** (Neural networks, CNNs, RNNs)
- **Modern architectures** (Transformers, Diffusion models)
- **Large Language Models** (ChatGPT, GPT-4, Claude, Gemini)
- **Prompt Engineering** (How to effectively use LLMs)
- **Automation** (Building AI workflows)
- **Production deployment** (Making models production-ready)

## 🚀 5-Minute Quick Start

### 1. Install Python (if needed)
```bash
python --version  # Should be 3.8 or higher
```

### 2. Setup Environment
```bash
# Navigate to code examples
cd /path/to/code_examples

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Your First Example
```bash
# Learn about Machine Learning fundamentals
python 1_fundamentals/supervised_learning.py

# Build and train a neural network
python 2_deep_learning/tensorflow_mnist.py
```

## 📚 Choose Your Path

### Path A: "I want to learn Deep Learning" ⏱️ 2-3 weeks
```
1_fundamentals/supervised_learning.py
    ↓
2_deep_learning/tensorflow_mnist.py
    ↓
3_cnns/README.md (Read about CNNs)
    ↓
4_rnns_lstms/README.md (Read about RNNs)
    ↓
6_transformers/README.md (Modern architecture)
```

### Path B: "I want to work with ChatGPT/LLMs" ⏱️ 1-2 weeks
```
8_large_language_models/openai_api_example.py
    ↓
9_prompt_engineering/basic_prompts.py
    ↓
13_automation_workflows/webhook_automation.py
```

### Path C: "I want to build production AI systems" ⏱️ 3-4 weeks
```
1_fundamentals/ (Understand basics)
    ↓
8_large_language_models/ (Learn LLMs)
    ↓
12_ai_tools_integration/ (Integrate APIs)
    ↓
13_automation_workflows/ (Build workflows)
    ↓
14_deployment_optimization/ (Deploy)
```

### Path D: "I want to generate images" ⏱️ 2-3 weeks
```
2_deep_learning/tensorflow_mnist.py (Neural networks)
    ↓
5_generative_models/README.md (GANs & VAEs)
    ↓
7_diffusion_models/README.md (Modern generation)
```

## 📁 Directory Overview

```
code_examples/
├── 1_fundamentals/           ← Start here for ML basics
├── 2_deep_learning/          ← Neural networks (fully implemented)
├── 3_cnns/                   ← Image processing
├── 4_rnns_lstms/             ← Sequence data
├── 5_generative_models/      ← GANs, VAEs
├── 6_transformers/           ← Modern architecture
├── 7_diffusion_models/       ← Image generation
├── 8_large_language_models/  ← ChatGPT, LLMs (fully implemented)
├── 9_prompt_engineering/     ← How to prompt LLMs (fully implemented)
├── 10_multimodal_models/     ← Vision + Language
├── 11_reinforcement_learning/← Learning from rewards
├── 12_ai_tools_integration/  ← Using APIs
├── 13_automation_workflows/  ← Automating tasks (fully implemented)
└── 14_deployment_optimization/← Making production-ready
```

## 🎯 Example Scenarios

### Scenario 1: "I want to classify images"
1. Run: `python 1_fundamentals/supervised_learning.py`
2. Learn: Read `3_cnns/README.md`
3. Practice: Modify `3_cnns/image_classification.py`

### Scenario 2: "I want to create a chatbot"
1. Run: `python 8_large_language_models/openai_api_example.py`
2. Learn: Read `9_prompt_engineering/README.md`
3. Build: Modify `13_automation_workflows/webhook_automation.py`

### Scenario 3: "I want to predict stock prices"
1. Learn: Read `4_rnns_lstms/README.md`
2. Practice: Run `4_rnns_lstms/time_series.py` (when created)

### Scenario 4: "I want to generate text"
1. Learn: Read `8_large_language_models/README.md`
2. Practice: `python 8_large_language_models/openai_api_example.py`
3. Optimize: Learn from `9_prompt_engineering/basic_prompts.py`

## ⚙️ Environment Setup

### Option 1: Basic Setup (Recommended for beginners)
```bash
# Install TensorFlow and PyTorch
pip install -r requirements.txt
```

### Option 2: GPU Acceleration (For faster training)
```bash
# Install CUDA first (NVIDIA GPU only)
# Then install GPU versions
pip install tensorflow-gpu torch torchvision
```

### Option 3: Specific Tool Setup

**For OpenAI API:**
```bash
export OPENAI_API_KEY="your-key-here"
python 8_large_language_models/openai_api_example.py
```

**For Google Cloud:**
```bash
export GOOGLE_API_KEY="your-key-here"
```

**For Jupyter Notebooks:**
```bash
pip install jupyter
jupyter notebook
```

## 📖 Reading Each Example

Every example follows this structure:

```python
"""
TITLE: What this demonstrates
"""

# ============================================================================
# EXAMPLE 1: First concept
# ============================================================================
def example_1():
    """Explanation of what happens"""
    # Code with detailed comments
    pass

# ============================================================================
# EXAMPLE 2: Second concept
# ============================================================================
def example_2():
    """Explanation"""
    # Code with comments
    pass

if __name__ == "__main__":
    # Shows how to run the examples
    example_1()
    example_2()
```

**How to learn from code:**
1. Read the comments
2. Run the code
3. Modify parameters
4. See what changes
5. Understand the concept

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
Solution: pip install tensorflow
```

### Problem: "ImportError: No module named 'openai'"
```bash
Solution: pip install openai
```

### Problem: "CUDA not available"
```bash
Solution: Use CPU instead or install NVIDIA drivers + CUDA
```

### Problem: "Out of memory"
```bash
Solution: Reduce batch size in code, use smaller models
```

### Problem: "API Key error"
```bash
Solution: Check your API key, set environment variables
```

## 📝 Common Tasks

### Task 1: Run a specific example
```bash
cd code_examples
python [section]/[script].py
```

### Task 2: Modify and experiment
```python
# Open the file in your editor
# Change parameters like:
learning_rate = 0.001  # Try 0.0001 or 0.01
batch_size = 32        # Try 64 or 128
epochs = 10            # Try 20 or 5
# Save and run again
```

### Task 3: Use with Jupyter
```bash
jupyter notebook
# Open code_examples in browser
# Create new notebook or modify existing
```

### Task 4: Check results
```bash
# Most examples save visualizations:
ls *.png      # Look for saved plots
ls *.csv      # Look for saved data
```

## 🎓 Learning Tips

### Tip 1: Read Before Running
- Read the README for each section first
- Understand concepts before code

### Tip 2: Run Small First
- Start with simple examples
- Gradually increase complexity

### Tip 3: Modify and Test
- Change one parameter at a time
- Understand cause and effect

### Tip 4: Take Notes
- Write down what you learn
- Compare results

### Tip 5: Join Communities
- Ask questions on Stack Overflow
- Share progress on social media

## 📚 What Each Section Teaches

| Section | Key Skill | Time | Difficulty |
|---------|-----------|------|------------|
| 1 | ML Fundamentals | 1-2h | Beginner |
| 2 | Neural Networks | 2-3h | Beginner-Int |
| 3 | Image Processing | 2-3h | Intermediate |
| 4 | Sequential Data | 2-3h | Intermediate |
| 5 | Content Generation | 2-3h | Advanced |
| 6 | Modern Architecture | 2-3h | Advanced |
| 7 | Image Generation | 2-3h | Advanced |
| 8 | Large Language Models | 2-3h | Advanced |
| 9 | Prompt Engineering | 1-2h | Advanced |
| 10 | Multimodal AI | 1-2h | Advanced |
| 11 | Reinforcement Learning | 2-3h | Advanced |
| 12 | API Integration | 1-2h | Int-Adv |
| 13 | Automation | 1-2h | Int-Adv |
| 14 | Deployment | 2-3h | Int-Adv |

**Total estimated time: 28-40 hours**

## ✅ Success Checklist

After going through examples, you should be able to:

- ✅ Understand machine learning concepts
- ✅ Build and train neural networks
- ✅ Process images with CNNs
- ✅ Handle sequences with RNNs
- ✅ Use transformer models
- ✅ Work with LLM APIs
- ✅ Write effective prompts
- ✅ Integrate AI into applications
- ✅ Automate workflows
- ✅ Deploy models to production

## 🚀 Next Level

After completing the basics:
1. **Build Projects**: Apply to real data
2. **Research Papers**: Read about latest techniques
3. **Contribute**: Share your improvements
4. **Teach Others**: Solidify your knowledge

## 📞 Getting Help

1. **Check README**: Each section has detailed explanations
2. **Read Comments**: Code has extensive comments
3. **Google It**: Most issues have solutions online
4. **Forums**: Stack Overflow, Reddit r/MachineLearning
5. **Documentation**: TensorFlow, PyTorch docs

## 🎯 Your First 30 Minutes

```
Minute 0-5:    Read this file and README.md
Minute 5-10:   Install dependencies
Minute 10-15:  Run first example
Minute 15-25:  Read output and understand
Minute 25-30:  Modify one parameter and re-run
```

## 💡 Key Reminders

- **Start Simple**: Don't skip fundamentals
- **Practice**: Code along, don't just read
- **Experiment**: Modify and test variations
- **Understand**: Know "why" not just "how"
- **Build**: Create your own projects
- **Share**: Help others learn

## 🎓 Learning Outcome

After completing these examples, you'll be able to:
- Understand how AI works fundamentally
- Build, train, and optimize models
- Use modern LLMs effectively
- Automate tasks with AI
- Deploy solutions professionally

## 📚 Additional Resources

- **TensorFlow**: https://tensorflow.org
- **PyTorch**: https://pytorch.org
- **Hugging Face**: https://huggingface.co
- **OpenAI**: https://openai.com/api
- **Papers with Code**: https://paperswithcode.com

---

**Ready to start? Pick your path and run your first example!**

Questions? Check the README in each section or the documentation links.

Good luck! 🚀

