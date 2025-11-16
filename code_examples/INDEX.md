# Complete Index of Code Examples

All code examples from IROST AI presentations organized by topic and difficulty level.

## 📚 Complete Table of Contents

### Level 1: Fundamentals (Beginner)

#### 1. **Fundamentals of AI & Machine Learning** [`1_fundamentals/`]
   - Supervised Learning (Classification & Regression)
   - Unsupervised Learning (Clustering, Dimensionality Reduction)
   - Reinforcement Learning Basics
   - Train/Test Split & Evaluation Metrics
   
   📄 **Key Files:**
   - `supervised_learning.py` - Classification examples with iris dataset
   - `unsupervised_learning.py` - K-Means clustering demonstration
   - `evaluation_metrics.py` - Understanding accuracy, precision, recall, F1
   - `train_test_split.py` - Data splitting techniques

#### 2. **Deep Learning Fundamentals** [`2_deep_learning/`]
   - Neural Network Basics (Perceptrons, Layers)
   - Activation Functions (ReLU, Sigmoid, Tanh)
   - Forward & Backward Propagation
   - Loss Functions & Optimization (SGD, Adam)
   - Training Fundamentals
   
   📄 **Key Files:**
   - `tensorflow_mnist.py` - Complete MNIST tutorial with TensorFlow
   - `pytorch_mnist.py` - MNIST with PyTorch
   - `activation_functions.py` - Visualizing all activation functions
   - `regularization.py` - Dropout, L1/L2, Early Stopping
   - `hyperparameter_tuning.py` - Finding optimal parameters

---

### Level 2: Specialized Architectures (Intermediate)

#### 3. **Convolutional Neural Networks (CNNs)** [`3_cnns/`]
   - Convolution Operation & Filters
   - Pooling Layers
   - CNN Architecture Design
   - Classic Models (LeNet, AlexNet, VGG, ResNet, EfficientNet)
   - Transfer Learning with Pre-trained Models
   
   📄 **Key Files:**
   - `cnn_basics.py` - Building CNNs from scratch
   - `image_classification.py` - CIFAR-10 classification
   - `transfer_learning.py` - Using pre-trained ImageNet models
   - `object_detection.py` - YOLO and R-CNN examples
   - `visualization.py` - Understanding learned features

#### 4. **Recurrent Neural Networks & LSTMs** [`4_rnns_lstms/`]
   - RNN Fundamentals
   - LSTM Architecture (Forget Gate, Input Gate, Output Gate)
   - GRU (Gated Recurrent Unit)
   - Sequence-to-Sequence Models
   - Attention Mechanisms
   
   📄 **Key Files:**
   - `rnn_basics.py` - Simple RNN implementation
   - `lstm_sequence.py` - LSTM for sequence modeling
   - `time_series.py` - Stock price / weather prediction
   - `text_generation.py` - Generating text with RNNs
   - `machine_translation.py` - Seq2seq translation

#### 5. **Generative Models** [`5_generative_models/`]
   - GANs (Generative Adversarial Networks)
   - VAE (Variational Autoencoders)
   - DCGAN, StyleGAN, CycleGAN
   - Image Generation & Style Transfer
   
   📄 **Key Files:**
   - `gan_basics.py` - Simple GAN on MNIST
   - `dcgan.py` - Deep Convolutional GAN
   - `conditional_gan.py` - Class-conditioned generation
   - `vae_basics.py` - Variational Autoencoders
   - `style_transfer.py` - CycleGAN, Pix2Pix
   - `evaluation.py` - FID, Inception Score

---

### Level 3: Modern Architectures (Advanced)

#### 6. **Transformers** [`6_transformers/`]
   - Attention Mechanism (Scaled Dot-Product)
   - Multi-Head Attention
   - Self-Attention & Cross-Attention
   - Transformer Encoder-Decoder
   - BERT, GPT, RoBERTa
   - Vision Transformers (ViT)
   
   📄 **Key Files:**
   - `transformer_basics.py` - Understanding attention from scratch
   - `bert_classification.py` - BERT for text classification
   - `gpt_generation.py` - GPT for text generation
   - `vision_transformer.py` - ViT for image classification
   - `fine_tuning.py` - Fine-tuning pre-trained models
   - `multi_gpu.py` - Distributed training

#### 7. **Diffusion Models** [`7_diffusion_models/`]
   - Diffusion Process (Forward & Reverse)
   - Noise Scheduling
   - Denoising Networks
   - Stable Diffusion, DALL-E
   - Text-to-Image Generation
   
   📄 **Key Files:**
   - `diffusion_basics.py` - Diffusion fundamentals
   - `ddpm.py` - Denoising Diffusion Probabilistic Model
   - `stable_diffusion.py` - Using Stable Diffusion API
   - `text_to_image.py` - Generating images from text
   - `image_editing.py` - Inpainting and editing

---

### Level 4: LLMs & Real-World Applications (Advanced)

#### 8. **Large Language Models** [`8_large_language_models/`]
   - Tokenization & Embeddings
   - Context Windows
   - Next Token Prediction
   - GPT Architecture
   - LLM APIs (OpenAI, Google, Anthropic)
   - Streaming & Rate Limiting
   
   📄 **Key Files:**
   - `openai_api_example.py` - ChatGPT & GPT-4 API usage
   - `google_gemini.py` - Google Gemini API
   - `anthropic_claude.py` - Anthropic Claude API
   - `huggingface_example.py` - Open-source models
   - `token_analysis.py` - Understanding tokenization
   - `streaming_responses.py` - Real-time text generation
   - `function_calling.py` - Structured outputs

#### 9. **Prompt Engineering** [`9_prompt_engineering/`]
   - Zero-Shot Learning
   - Few-Shot Learning
   - Chain-of-Thought (CoT) Prompting
   - Role Assignment (Persona)
   - Prompt Optimization & Testing
   - Few-Shot Examples
   
   📄 **Key Files:**
   - `basic_prompts.py` - Prompt quality levels and improvement
   - `few_shot_learning.py` - Few-shot demonstrations
   - `chain_of_thought.py` - Step-by-step reasoning
   - `prompt_templates.py` - Reusable templates
   - `prompt_optimization.py` - Testing and measuring

#### 10. **Multimodal Models** [`10_multimodal_models/`]
   - Vision-Language Models (CLIP)
   - Image Captioning
   - Visual Question Answering (VQA)
   - Cross-Modal Retrieval
   
   📄 **Key Files:**
   - `clip_example.py` - Image-text matching
   - `image_captioning.py` - Generating captions
   - `visual_qa.py` - Answering questions about images
   - `cross_modal_retrieval.py` - Finding similar images/text

#### 11. **Reinforcement Learning** [`11_reinforcement_learning/`]
   - Markov Decision Process
   - Q-Learning & Deep Q-Network (DQN)
   - Policy Gradient Methods
   - Actor-Critic Algorithms
   
   📄 **Key Files:**
   - `qlearning.py` - Classic Q-Learning
   - `dqn.py` - Deep Q-Network
   - `policy_gradient.py` - REINFORCE algorithm
   - `actor_critic.py` - A2C/A3C algorithms
   - `gym_environment.py` - Using OpenAI Gym

---

### Level 5: Integration & Deployment (Practical)

#### 12. **AI Tools Integration** [`12_ai_tools_integration/`]
   - OpenAI API Integration
   - Google Cloud AI Services
   - Hugging Face Hub
   - Error Handling & Rate Limiting
   - Cost Optimization
   
   📄 **Key Files:**
   - `openai_integration.py` - Complete ChatGPT integration
   - `google_integration.py` - Google Vertex AI, Gemini
   - `huggingface_integration.py` - Model hub usage
   - `error_handling.py` - Robust error management
   - `cost_tracking.py` - Monitor API costs

#### 13. **Automation & Workflows** [`13_automation_workflows/`]
   - Webhook Integration
   - Workflow Automation (Zapier, Make)
   - AI in Workflows
   - Database Automation (Airtable, Notion)
   - Error Handling & Monitoring
   
   📄 **Key Files:**
   - `webhook_automation.py` - Building custom workflows
   - `airtable_automation.py` - Airtable API integration
   - `notion_automation.py` - Notion API integration
   - `workflow_examples.json` - Pre-built workflows
   - `zapier_integration.py` - Zapier API usage

#### 14. **Deployment & Optimization** [`14_deployment_optimization/`]
   - Model Quantization & Compression
   - ONNX Export
   - TensorFlow Lite (On-Device)
   - Docker Containerization
   - API Deployment
   - Monitoring & Logging
   
   📄 **Key Files:**
   - `quantization.py` - Model quantization techniques
   - `onnx_export.py` - Converting to ONNX format
   - `tflite_conversion.py` - Mobile model conversion
   - `docker_deployment.py` - Containerizing models
   - `api_serving.py` - FastAPI/Flask endpoints
   - `monitoring.py` - Performance tracking

---

## 🎯 Learning Paths

### Path 1: Deep Learning Fundamentals (2-3 weeks)
1. Fundamentals (1) - Understanding ML concepts
2. Deep Learning (2) - Neural networks basics
3. CNNs (3) - Image understanding
4. RNNs (4) - Sequence processing
5. Transformers (6) - Modern architecture

### Path 2: Large Language Models (2-3 weeks)
1. Fundamentals (1) - ML basics
2. Deep Learning (2) - Neural networks
3. Transformers (6) - Attention mechanism
4. LLMs (8) - Working with language models
5. Prompt Engineering (9) - Effective prompting

### Path 3: Generative AI (3-4 weeks)
1. Fundamentals (1)
2. Deep Learning (2)
3. Generative Models (5) - GANs & VAEs
4. Diffusion Models (7) - Modern generation
5. LLMs (8) + Prompt Engineering (9)

### Path 4: Full Stack AI Engineer (8-10 weeks)
1. All sections 1-7 (Core concepts)
2. Sections 8-11 (Applications)
3. Sections 12-14 (Production)

### Path 5: Practical AI Developer (4-6 weeks)
1. Fundamentals (1)
2. LLMs (8)
3. Prompt Engineering (9)
4. AI Tools Integration (12)
5. Automation (13)

---

## 📊 Concept Difficulty Matrix

```
             Beginner    Intermediate   Advanced    Expert
Deep Learn    1-2           3-4          5-6         7
NLP           2            6             8-9         -
Vision        2            3-4           10          -
Automation    -            12            13-14       -
Production    -            -             14          -
```

---

## 🚀 Quick Start by Goal

### "I want to build a chatbot"
→ Start with: Fundamentals (1) → LLMs (8) → Prompt Engineering (9) → Automation (13)

### "I want to generate images"
→ Start with: Deep Learning (2) → Generative Models (5) → Diffusion Models (7)

### "I want to classify images"
→ Start with: Fundamentals (1) → Deep Learning (2) → CNNs (3) → Transfer Learning (3)

### "I want to deploy ML models"
→ Start with: Any foundation → Deployment & Optimization (14)

### "I want to automate workflows with AI"
→ Start with: LLMs (8) → AI Tools (12) → Automation (13)

---

## 📦 Dependencies

All dependencies are in `requirements.txt`:
- Deep Learning: TensorFlow, PyTorch
- NLP: Transformers, Hugging Face
- APIs: OpenAI, Google, Anthropic
- Utilities: Pandas, NumPy, Requests

Install all:
```bash
pip install -r requirements.txt
```

---

## 🔗 Cross-References

### GANs mentioned in:
- Section 5: Full implementation
- Section 7: Comparison with diffusion
- Section 13: Workflow for generation

### Transfer Learning mentioned in:
- Section 3: CNNs (ResNet, VGG)
- Section 6: Transformers (BERT, GPT)
- Section 8: LLMs (Fine-tuning)

### Attention Mechanism mentioned in:
- Section 4: RNNs (Seq2seq)
- Section 6: Transformers (Core concept)
- Section 10: Multimodal (Cross-modal)

---

## 📝 Code Quality Standards

All code includes:
- ✅ Clear comments and docstrings
- ✅ Type hints where applicable
- ✅ Error handling
- ✅ Example usage
- ✅ Performance notes

---

## 🎓 Learning Resources Embedded

Each section includes:
- Concept explanations
- Working code examples
- Practical exercises
- Common mistakes & fixes
- Next steps guidance

---

## 💡 Tips for Success

1. **Run examples in order** - Build foundational understanding
2. **Modify code** - Change parameters, test variations
3. **Read comments** - Understand the "why" not just "how"
4. **Check documentation** - Link to official docs provided
5. **Practice** - Build your own variants
6. **Monitor performance** - Track metrics and improvements

---

## 🤝 Contributing

Found an issue or have improvements?
1. Test your changes
2. Document updates
3. Submit with clear description

---

## 📄 License

Educational purposes - IROST Presentation Materials

---

**Last Updated**: 2024
**Total Examples**: 50+
**Total Lines of Code**: 10,000+
**Coverage**: All topics from presentations

Happy Learning! 🚀

