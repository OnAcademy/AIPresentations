# Section 5: Generative Models (GANs & VAEs)

## Concepts Covered

1. **Generative Adversarial Networks (GANs)**
   - Generator vs Discriminator
   - Adversarial loss
   - Training dynamics

2. **VAE (Variational Autoencoders)**
   - Encoder-decoder architecture
   - Latent space
   - Reconstruction and KL divergence

3. **Advanced GANs**
   - DCGAN (Deep Convolutional)
   - StyleGAN
   - Conditional GANs

4. **Applications**
   - Image generation
   - Style transfer
   - Image-to-image translation
   - Super-resolution
   - Anomaly detection

5. **Challenges**
   - Mode collapse
   - Training instability
   - Evaluation metrics

## GAN Architecture

```
Noise Vector (z)
    ↓
Generator (G)
    ↓
Fake Image
    ├─→ Discriminator (D)
    │      ↓
    │  Real or Fake?
    │
Real Image ─→ Discriminator (D)
                  ↓
              Real or Fake?
```

## Files in This Section

- `gan_basics.py` - Building a simple GAN
- `dcgan.py` - Deep Convolutional GAN
- `conditional_gan.py` - Controlled generation
- `vae_basics.py` - Variational Autoencoders
- `style_transfer.py` - CycleGAN and pix2pix
- `evaluation.py` - FID, Inception Score

## GANs vs VAEs vs Flow Models

| Aspect | GAN | VAE | Flow |
|--------|-----|-----|------|
| Training | Adversarial | Variational | Maximum Likelihood |
| Quality | High (sharp) | Lower (blurry) | High |
| Speed | Fast | Fast | Slower |
| Stability | Difficult | Stable | Stable |
| Likelihood | Unknown | Tractable | Tractable |

## GAN Training Objective

```
min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Generator: Minimize discriminator's confidence
Discriminator: Maximize confidence on real/fake
```

## Quick Example: Simple GAN

```python
import tensorflow as tf

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(28*28*1, activation='tanh'),
    tf.keras.layers.Reshape((28, 28, 1))
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Training involves alternating updates...
```

## VAE Concept

Unlike GANs, VAEs have tractable likelihood:

```
Encoder: x → μ, σ → z (latent code)
         Learns distribution of data
         
Decoder: z → x̂ (reconstruct)
         z is sampled from N(μ, σ)

Loss = Reconstruction Loss + KL Divergence
     = MSE(x, x̂) + KL(N(μ,σ) || N(0,1))
```

## Popular GAN Variants

### 1. DCGAN (2015)
- Uses convolutional layers
- Stable training techniques
- Generates high-quality images

### 2. Conditional GAN (cGAN)
- Controlled generation with class labels
- Can generate specific digits, faces, etc.
- Used in text-to-image generation

### 3. StyleGAN (2018)
- Fine-grained control over generated images
- Separates style from content
- State-of-the-art face generation

### 4. CycleGAN (2017)
- Unpaired image-to-image translation
- No paired training data needed
- Horse ↔ Zebra, Summer ↔ Winter

### 5. Pix2Pix (2016)
- Paired image translation
- Requires aligned training data
- Sketch → Photo, Satellite → Map

## Training Challenges

### Mode Collapse
Problem: Generator learns to fool discriminator with limited patterns
Solutions:
- Minibatch discrimination
- Feature matching
- Training on harder tasks

### Vanishing Gradients
Problem: Discriminator too good, generator gradient disappears
Solutions:
- Wasserstein loss
- Spectral normalization
- Progressive growing

### Training Instability
Problem: Oscillating loss, divergence
Solutions:
- Better learning rates
- Batch normalization
- Label smoothing

## Evaluation Metrics

### Inception Score (IS)
- Higher is better
- Inception network confidence + diversity
- Range: 1-1000 (typically 5-50 for good models)

### Fréchet Inception Distance (FID)
- Lower is better
- Distance between generated and real distributions
- Typical: <50 is good, <20 is excellent

### Precision and Recall
- Precision: Quality of generated samples
- Recall: Diversity of generated samples

## Applications in Detail

### 1. Image Generation
- Generate new faces, objects, scenes
- Used in design tools, games
- Quality depends on training data

### 2. Image Inpainting
- Fill missing/corrupted regions
- Used in photo editing, restoration
- Leverages learned image priors

### 3. Super-Resolution
- Increase image resolution
- Upscale 64x64 → 256x256
- Used in satellite imagery, medical imaging

### 4. Anomaly Detection with VAE
- Train on normal data
- High reconstruction error = anomaly
- Used in fraud detection, manufacturing

### 5. Style Transfer
- Extract style from one image
- Apply to another image
- CycleGAN enables unpaired transfer

## Best Practices

✓ **Use Batch Normalization**: Stabilizes training
✓ **Monitor Both Losses**: D and G should be balanced
✓ **Use Label Smoothing**: Soft targets (0.9, 0.1) instead of (1, 0)
✓ **Progressive Growing**: Start small, gradually increase resolution
✓ **Spectral Normalization**: Normalize weights to improve stability
✓ **Validation Set**: Monitor FID/IS on held-out data

## Hyperparameter Tuning

| Parameter | Impact | Typical Range |
|-----------|--------|----------------|
| Learning Rate | High | 0.0002 - 0.0005 |
| Batch Size | High | 32 - 128 |
| Noise Dim | Medium | 100 - 512 |
| D Updates | High | 1 or 5 per G |
| Dropout | Medium | 0.3 - 0.5 |

## Advanced Techniques

1. **Two Time-Scale Update Rule**: Different learning rates for G and D
2. **Spectral Normalization**: Control Lipschitz constant
3. **Gradient Penalty**: Enforce 1-Lipschitz constraint
4. **Feature Matching**: Match intermediate features
5. **Self-Attention**: Add attention layers (SA-GAN)

## Real-World Examples

- **Face Generation**: StyleGAN, StarGAN
- **Music Generation**: MuseGAN
- **Video Generation**: MoCoGAN, StyleGAN-V
- **Text-to-Image**: DALL-E (Diffusion), AttnGAN
- **Anomaly Detection**: VAE on manufacturing data

## Common Mistakes

❌ Using same learning rate for G and D (use 2x for G)
❌ Not monitoring both generator and discriminator loss
❌ Training until convergence (might indicate mode collapse)
❌ Using unbalanced architectures
❌ Poor hyperparameter initialization

## Generative Models Comparison

| Model | Type | Quality | Stability | Speed |
|-------|------|---------|-----------|-------|
| GAN | Adversarial | Excellent | Difficult | Fast |
| VAE | Variational | Good | Stable | Fast |
| Flow | Likelihood | Excellent | Stable | Slower |
| Diffusion | Iterative | Excellent | Very stable | Slower |

## Next Steps

1. Understand GAN objective
2. Build simple GAN on MNIST
3. Implement DCGAN
4. Try conditional generation
5. Explore VAE for interpolation
6. Learn about Diffusion models (next generation)

