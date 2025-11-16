# Quick Start Guide

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- 4GB+ RAM
- GPU (recommended for deep learning, optional)

## Installation

### 1. Clone or Download Code Examples

```bash
cd /path/to/code_examples
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you have GPU support:
```bash
# For NVIDIA GPU with CUDA
pip install tensorflow-gpu torch torchvision
```

## Running Examples by Section

### 1️⃣ Fundamentals (AI & ML Basics)
```bash
cd 1_fundamentals

# Supervised Learning
python supervised_learning.py

# Unsupervised Learning  
python unsupervised_learning.py
```

### 2️⃣ Deep Learning Fundamentals
```bash
cd 2_deep_learning

# MNIST with TensorFlow
python tensorflow_mnist.py

# MNIST with PyTorch
python pytorch_mnist.py
```

### 3️⃣ Convolutional Neural Networks
```bash
cd 3_cnns

# Image Classification
python image_classification.py

# Transfer Learning
python transfer_learning.py
```

### 4️⃣ RNNs & LSTMs
```bash
cd 4_rnns_lstms

# Sequence Modeling
python sequence_modeling.py

# Time Series
python time_series.py
```

### 5️⃣ Generative Models
```bash
cd 5_generative_models

# GANs (Generative Adversarial Networks)
python gan_example.py

# VAE (Variational Autoencoders)
python vae_example.py
```

### 6️⃣ Transformers
```bash
cd 6_transformers

# Transformer Architecture
python transformer_basics.py

# BERT & GPT
python language_models.py
```

### 7️⃣ Diffusion Models
```bash
cd 7_diffusion_models

# Stable Diffusion
python diffusion_basics.py
```

### 8️⃣ Large Language Models
```bash
cd 8_large_language_models

# OpenAI API (requires API key)
export OPENAI_API_KEY="your-key-here"
python openai_api_example.py

# Hugging Face
python huggingface_example.py
```

### 9️⃣ Prompt Engineering
```bash
cd 9_prompt_engineering

# Basic Techniques
python basic_prompts.py

# Few-Shot Learning
python few_shot_learning.py

# Chain-of-Thought
python chain_of_thought.py
```

### 🔟 Multimodal Models
```bash
cd 10_multimodal_models

# Vision & Language
python clip_example.py

# Image Captioning
python image_captioning.py
```

### 1️⃣1️⃣ Reinforcement Learning
```bash
cd 11_reinforcement_learning

# Q-Learning
python q_learning.py

# Policy Gradient
python policy_gradient.py
```

### 1️⃣2️⃣ AI Tools Integration
```bash
cd 12_ai_tools_integration

# API Examples
python openai_integration.py
python google_integration.py
python huggingface_integration.py
```

### 1️⃣3️⃣ Automation Workflows
```bash
cd 13_automation_workflows

# Webhook Example
python webhook_automation.py

# Airtable Integration
python airtable_automation.py

# Notion Integration
python notion_automation.py
```

### 1️⃣4️⃣ Deployment & Optimization
```bash
cd 14_deployment_optimization

# Model Quantization
python quantization.py

# ONNX Export
python onnx_export.py

# Docker Deployment
docker build -t ai-app .
docker run ai-app
```

## Common Tasks

### Task 1: Train Your First Neural Network
```bash
cd 2_deep_learning
python tensorflow_mnist.py
# Output: Trained model, accuracy metrics, visualizations
```

### Task 2: Use ChatGPT API
```bash
cd 8_large_language_models
export OPENAI_API_KEY="sk-..."
python openai_api_example.py
```

### Task 3: Learn Prompt Engineering
```bash
cd 9_prompt_engineering
python basic_prompts.py
# Output: Best practices, examples, improvements
```

### Task 4: Build an Automation Workflow
```bash
cd 13_automation_workflows
python webhook_automation.py
# Output: Working workflow examples
```

### Task 5: Fine-tune a Pre-trained Model
```bash
cd 6_transformers
python fine_tuning.py
```

## Troubleshooting

### ImportError: No module named 'tensorflow'
```bash
pip install tensorflow
```

### ImportError: No module named 'openai'
```bash
pip install openai
```

### Out of Memory Error
- Use smaller batch sizes
- Use smaller model variants
- Enable GPU acceleration
- Close other applications

### API Rate Limits
- Add delays between requests
- Implement exponential backoff
- Use batch processing
- Upgrade to paid tier

### CUDA/GPU Issues
- Check NVIDIA driver: `nvidia-smi`
- Install CUDA toolkit
- Use CPU instead: `tf.config.list_physical_devices('GPU')`

## Environment Variables

Create `.env` file in project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Google
GOOGLE_API_KEY=...

# Anthropic
ANTHROPIC_API_KEY=...

# Database
DATABASE_URL=postgresql://user:pass@localhost/dbname

# Airtable
AIRTABLE_API_KEY=...
AIRTABLE_BASE_ID=...

# Notion
NOTION_API_KEY=...
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

## Running Jupyter Notebooks

```bash
jupyter notebook
# Opens browser at http://localhost:8888
```

## Docker Usage

Build image:
```bash
docker build -t ai-examples .
```

Run container:
```bash
docker run -it -p 8888:8888 ai-examples
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Run a simple example
3. ✅ Modify example with your own data
4. ✅ Explore related examples
5. ✅ Build your own project

## Learning Path

**Beginner:**
1. Fundamentals (1)
2. Deep Learning Basics (2)
3. CNNs (3)

**Intermediate:**
1. RNNs (4)
2. Transformers (6)
3. Large Language Models (8)
4. Prompt Engineering (9)

**Advanced:**
1. Generative Models (5)
2. Diffusion Models (7)
3. Multimodal (10)
4. Reinforcement Learning (11)
5. Deployment (14)

**Practical:**
1. AI Tools (12)
2. Automation (13)
3. Build Your Own

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org)
- [PyTorch Documentation](https://pytorch.org)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Papers with Code](https://paperswithcode.com)

## Getting Help

1. Check README.md in specific section
2. Review example code comments
3. Check error messages carefully
4. Look for similar issues online
5. Consult documentation

## Contributing

Found a bug? Have an improvement?
1. Test thoroughly
2. Document changes
3. Submit example

## License

Educational purposes - IROST Presentation Materials

---

**Happy Learning! 🚀**

