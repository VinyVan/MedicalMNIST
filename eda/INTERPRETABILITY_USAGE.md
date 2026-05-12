# Interpretability Methods Usage Guide

This guide explains how to use the advanced interpretability methods in the Medical MNIST analysis notebook.

## Installation

The interpretability methods are in `eda/interpretability_methods.py`. To use them in your notebook, add this cell:

```python
# Import interpretability methods
import sys
sys.path.append('eda')
from interpretability_methods import (
    GuidedBackprop,
    GuidedGradCAM,
    OcclusionSensitivity,
    LimeExplainer,
    ShapExplainer,
    PermutationChannelImportance,
    invert_grayscale_image,
    create_guided_backprop_visualization,
    create_guided_gradcam_visualization,
    create_occlusion_visualization,
    create_lime_visualization,
    create_shap_visualization,
    create_pci_visualization,
    create_inversion_visualization,
    run_comprehensive_interpretability
)

print("Interpretability methods imported successfully!")
```

## Individual Method Usage

### 1. Guided Backpropagation

```python
# Select a sample image
sample_idx = wrong_indices[0]  # or any index
image_path = test_paths[sample_idx]
true_label = class_names[true_labels[sample_idx]]
pred_label = class_names[predictions[sample_idx]]
confidence = np.max(probabilities[sample_idx])

# Generate visualization
fig = create_guided_backprop_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 2. Guided Grad-CAM

```python
fig = create_guided_gradcam_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 3. Occlusion Sensitivity

```python
fig = create_occlusion_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 4. LIME

```python
fig = create_lime_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 5. SHAP

```python
fig = create_shap_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 6. Permutation Channel Importance (PCI)

```python
fig = create_pci_visualization(
    trained_models[0], image_path, true_label, pred_label, confidence, config
)
plt.show()
```

### 7. Grayscale Image Inversion

```python
fig = create_inversion_visualization(
    image_path, true_label, pred_label, confidence
)
plt.show()
```

## Comprehensive Analysis

To run all methods on a single image and save results:

```python
# Run all interpretability methods on wrong predictions
results = run_comprehensive_interpretability(
    model=trained_models[0],
    image_path=test_paths[wrong_indices[0]],
    true_label=class_names[true_labels[wrong_indices[0]]],
    predicted_label=class_names[predictions[wrong_indices[0]]],
    confidence=np.max(probabilities[wrong_indices[0]]),
    config=config,
    save_dir="interpretability_results"
)

print(f"Generated {len(results)} interpretability visualizations")
```

## Batch Analysis on Multiple Images

```python
# Analyze all wrong predictions with all methods
import os
save_dir = "interpretability_results"
os.makedirs(save_dir, exist_ok=True)

for idx in wrong_indices:
    results = run_comprehensive_interpretability(
        model=trained_models[0],
        image_path=test_paths[idx],
        true_label=class_names[true_labels[idx]],
        predicted_label=class_names[predictions[idx]],
        confidence=np.max(probabilities[idx]),
        config=config,
        save_dir=save_dir
    )
```

## Method Descriptions

### Guided Backpropagation
- Shows pixel-level gradients that contribute to the prediction
- Only positive gradients are propagated through ReLU layers
- Provides fine-grained localization

### Guided Grad-CAM
- Combines Grad-CAM with Guided Backpropagation
- Provides both class-discriminative localization and pixel-level gradients
- Better localization than Grad-CAM alone

### Occlusion Sensitivity
- Systematically occludes parts of the image
- Shows which regions are most important for the prediction
- Red regions indicate high importance

### LIME (Local Interpretable Model-Agnostic Explanations)
- Approximates the model locally with an interpretable linear model
- Shows which features (pixels) are most important
- Model-agnostic approach

### SHAP (SHapley Additive exPlanations)
- Uses game theory to assign importance values
- Provides consistent and fair feature attribution
- Gradient-based approximation for efficiency

### Permutation Channel Importance (PCI)
- For grayscale images: inverts the image (255 - pixel) and measures prediction impact
- Shows how sensitive the model is to pixel intensity changes
- Displays probability drop when image is inverted
- Useful for understanding if model relies on absolute pixel values

### Grayscale Image Inversion
- Simple transformation: inverted_pixel = 255 - original_pixel
- Converts black to white and white to black
- Helps test model robustness to intensity inversion
- Useful for medical imaging where intensity interpretation matters

## Notes

- All methods require the model to be in eval mode
- Images are automatically resized to 64x64 for model input
- Visualizations are saved at 256x256 for better viewing
- Results are saved as PNG files in the specified directory
- Some methods (LIME, SHAP) may be slower due to sampling
- PCI and inversion are particularly useful for grayscale medical images

## Performance Tips

- For faster analysis, reduce `num_samples` in LIMEExplainer
- Occlusion sensitivity can be sped up by increasing `stride` parameter
- Use GPU for faster gradient computations
- Process images in batches for large-scale analysis
- PCI and inversion are fast methods suitable for quick analysis
