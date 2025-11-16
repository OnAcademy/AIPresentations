# Installation & Setup Guide

## 📦 What's Included

Complete code examples and tutorials for both presentations:
- **application.html**: Deep Learning Revolution (DNNs, CNNs, RNNs, GANs, Transformers, Diffusion)
- **presentation.html**: Generative AI (LLMs, Prompt Engineering, Automation)

**Location**: `code_examples/` directory

## ✅ Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package manager (usually included with Python)
- **4GB RAM**: Minimum (8GB+ recommended)
- **Disk Space**: 2GB+ for dependencies

### Check Your System

```bash
# Check Python version
python --version

# Check pip
pip --version

# Check disk space
df -h  # Linux/Mac
dir    # Windows
```

## 🚀 Installation Steps

### Step 1: Navigate to Code Examples

```bash
cd /home/hossein/Universe/nebulas/OnAcadamy/IROST_presentation/code_examples
```

### Step 2: Create Virtual Environment

**Why?** Virtual environments keep project dependencies isolated.

```bash
# Create
python -m venv venv

# Activate (choose one based on your OS):

# Linux/Mac
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

You should see `(venv)` in your terminal prompt when activated.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Deep Learning: TensorFlow, PyTorch
- NLP: Transformers, Hugging Face
- APIs: OpenAI, Google, Anthropic
- Utilities: NumPy, Pandas, Matplotlib, etc.

**Installation time**: 5-10 minutes (depending on internet speed)

## ✨ Verify Installation

```bash
# Test imports
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import transformers; print('Transformers OK')"
```

## 🎯 First Run

```bash
# Try a simple example
python 1_fundamentals/supervised_learning.py

# You should see output about ML fundamentals
```

## 📚 Documentation Files

After installation, read these in order:

1. **GETTING_STARTED.md** - Beginner-friendly guide
2. **README.md** - Project overview
3. **INDEX.md** - Complete index & learning paths
4. **QUICK_START.md** - Quick tutorials

## 🔧 Optional: GPU Support

For faster training with NVIDIA GPU:

```bash
# Install CUDA Toolkit first (from NVIDIA website)
# Then install GPU versions

pip install tensorflow-gpu
# OR
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note**: GPU requires NVIDIA graphics card and CUDA installation.

## 🌐 Optional: API Keys Setup

For working with AI APIs:

### OpenAI (ChatGPT, GPT-4)
```bash
export OPENAI_API_KEY="your-key-here"
```

### Google (Gemini)
```bash
export GOOGLE_API_KEY="your-key-here"
```

### Anthropic (Claude)
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Or create `.env` file:**
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

## 📊 Verify All Sections Available

```bash
ls -la code_examples/

# Should show:
# 1_fundamentals/
# 2_deep_learning/
# 3_cnns/
# ... (14 sections total)
```

## 🚪 Exit Virtual Environment

When done:
```bash
deactivate
```

## ⚡ Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt

# Run example
python 1_fundamentals/supervised_learning.py

# Run Jupyter
jupyter notebook

# Deactivate
deactivate

# Remove environment (cleanup)
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows
```

## 🐛 Troubleshooting

### Issue: "Python not found"
```bash
# Install Python 3.8+ from python.org
# Or use package manager:
# Mac: brew install python
# Ubuntu: sudo apt-get install python3
```

### Issue: "No module named 'tensorflow'"
```bash
# Install missing package
pip install tensorflow
```

### Issue: "Permission denied" (Linux/Mac)
```bash
# Add execution permission
chmod +x venv/bin/python
```

### Issue: "Out of memory"
```bash
# Run smaller examples or reduce batch size in code
# Check available RAM:
free -h  # Linux
vm_stat  # Mac
```

### Issue: "CUDA/GPU not working"
```bash
# Check if CUDA available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Use CPU instead if GPU not available
```

## 📖 Directory Structure

```
code_examples/
├── 1_fundamentals/           ← Start here
├── 2_deep_learning/          ← Neural networks (main example)
├── 3_cnns/                   ← Images
├── 4_rnns_lstms/             ← Sequences
├── 5_generative_models/      ← GANs, VAEs
├── 6_transformers/           ← Modern architecture
├── 7_diffusion_models/       ← Image generation
├── 8_large_language_models/  ← LLMs (main example)
├── 9_prompt_engineering/     ← Prompting (main example)
├── 10_multimodal_models/     ← Vision + Language
├── 11_reinforcement_learning/← Learning from rewards
├── 12_ai_tools_integration/  ← API integration
├── 13_automation_workflows/  ← Automation (main example)
├── 14_deployment_optimization/← Production
├── README.md                 ← Project overview
├── GETTING_STARTED.md        ← Beginner guide
├── QUICK_START.md            ← Quick tutorials
├── INDEX.md                  ← Complete index
└── requirements.txt          ← Dependencies
```

## 🎓 Next Steps After Installation

1. **Read GETTING_STARTED.md**
2. **Run first example**: `python 1_fundamentals/supervised_learning.py`
3. **Choose learning path** from INDEX.md
4. **Follow tutorials** in each section
5. **Experiment** with modifying code
6. **Build projects** using learned concepts

## 📞 Common Setup Issues & Solutions

| Issue | Solution |
|-------|----------|
| Python not found | Install from python.org |
| pip not found | Install Python with pip |
| Permission denied | Use `sudo` or check permissions |
| Module not found | Run `pip install` for that module |
| Out of memory | Reduce batch size, use smaller models |
| GPU not working | Install CUDA, use CPU instead |
| API key error | Set environment variables correctly |

## ✅ Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed without errors
- [ ] Can import TensorFlow
- [ ] Can import PyTorch
- [ ] Can import transformers
- [ ] First example runs successfully
- [ ] Documentation files readable
- [ ] 14 section directories exist
- [ ] Ready to start learning!

## 🎯 Success Indicators

Installation is successful when:

```bash
# These should all work without errors:
python -c "import tensorflow"
python -c "import torch"
python -c "import transformers"
python -c "import openai"
python -c "import pandas"

# And running an example works:
python 1_fundamentals/supervised_learning.py
# Output: [Info about ML fundamentals]
```

## 📱 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.9-3.11 |
| RAM | 4GB | 8GB+ |
| Disk | 2GB | 5GB+ |
| GPU | Optional | NVIDIA RTX series |
| CUDA | - | 11.8+ (if using GPU) |

## 🌍 Offline Usage

Most examples work offline. For online features:
- API examples require internet
- Model downloads need internet first time
- After first download, models cache locally

## 🔄 Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Or specific package
pip install --upgrade tensorflow
```

## 🗑️ Cleanup

Remove virtual environment when done:

```bash
# Linux/Mac
rm -rf venv

# Windows
rmdir /s venv
```

## 📚 Where to Go From Here

After successful installation:

1. **Beginner Path**: Start with `GETTING_STARTED.md`
2. **Full Overview**: Read `README.md`
3. **Choose Topic**: See `INDEX.md` for learning paths
4. **First Example**: Run `python 1_fundamentals/supervised_learning.py`
5. **Explore Sections**: Check each section's README

---

**Installation complete? Start learning!** 🚀

Go to `GETTING_STARTED.md` for your first steps.

