"""
Advanced Interpretability Methods for Medical MNIST
Provides multiple visualization techniques for model interpretability:
- Guided Backpropagation
- Guided Grad-CAM
- LIME (Local Interpretable Model-Agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Occlusion Sensitivity
- Permutation Channel Importance (PCI)
- Grayscale Image Inversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# GUIDED BACKPROPAGATION
# ============================================================

class GuidedBackprop:
    """
    Guided Backpropagation visualization.
    Only backpropagates positive gradients through ReLU layers.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            # Store input for backward pass
            module.input = input
        
        def backward_hook(module, grad_input, grad_output):
            # Guided backprop: only positive gradients
            # Clone to avoid in-place modification error
            guided_grad = torch.clamp(grad_input[0].clone(), min=0.0)
            return (guided_grad,)
        
        # Register hooks on all ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
    
    def generate_gradients(self, input_tensor, class_idx):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients
        gradients = input_tensor.grad
        
        # Convert to numpy
        gradients = gradients.squeeze().cpu().detach().numpy()
        
        # Normalize for visualization
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        return gradients


def create_guided_backprop_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create guided backpropagation visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Generate guided backprop
    guided_bp = GuidedBackprop(model)
    class_idx = config.class_names.index(predicted_label)
    gradients = guided_bp.generate_gradients(input_tensor, class_idx)
    
    # Resize gradients to display size
    gradients_resized = cv2.resize(gradients, (256, 256))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    ax2.imshow(gradients_resized, cmap='jet')
    ax2.set_title('Guided Backpropagation')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# GUIDED GRAD-CAM
# ============================================================

class GuidedGradCAM:
    """
    Combines Grad-CAM with Guided Backpropagation for pixel-level gradients.
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def relu_forward_hook(module, input, output):
            module.input = input
        
        def relu_backward_hook(module, grad_input, grad_output):
            # Clone to avoid in-place modification error
            guided_grad = torch.clamp(grad_input[0].clone(), min=0.0)
            return (guided_grad,)
        
        # Register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        
        # Register hooks on ReLU layers for guided backprop
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(relu_forward_hook)
                module.register_full_backward_hook(relu_backward_hook)
    
    def generate_guided_gradcam(self, input_tensor, class_idx):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get guided backprop gradients
        guided_grads = input_tensor.grad.squeeze().cpu().detach().numpy()
        
        # Get Grad-CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * activations, dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.detach().cpu().numpy()
        
        # Resize CAM to input size
        cam_resized = cv2.resize(cam, (64, 64))
        
        # Combine guided backprop with Grad-CAM
        guided_gradcam = guided_grads * cam_resized
        
        # Normalize
        guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
        
        return guided_gradcam, cam, guided_grads


def create_guided_gradcam_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create guided Grad-CAM visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Auto-detect last conv layer
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = name
    
    # Generate guided Grad-CAM
    guided_gradcam = GuidedGradCAM(model, last_conv_layer)
    class_idx = config.class_names.index(predicted_label)
    guided_gradcam_result, cam, guided_grads = guided_gradcam.generate_guided_gradcam(input_tensor, class_idx)
    
    # Resize to display size
    guided_gradcam_resized = cv2.resize(guided_gradcam_result, (256, 256))
    cam_resized = cv2.resize(cam, (256, 256))
    guided_grads_resized = cv2.resize(guided_grads, (256, 256))
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Original\nTrue: {true_label}\nPred: {predicted_label}')
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    
    axes[2].imshow(guided_grads_resized, cmap='jet')
    axes[2].set_title('Guided Backprop')
    axes[2].axis('off')
    
    axes[3].imshow(guided_gradcam_resized, cmap='jet')
    axes[3].set_title('Guided Grad-CAM')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# OCCLUSION SENSITIVITY
# ============================================================

class OcclusionSensitivity:
    """
    Occlusion sensitivity analysis.
    Systematically occludes parts of the image to see prediction changes.
    """
    def __init__(self, model):
        self.model = model
    
    def generate_occlusion_map(self, input_tensor, class_idx, patch_size=8, stride=4):
        self.model.eval()
        
        with torch.no_grad():
            # Get original prediction
            original_output = self.model(input_tensor)
            original_prob = F.softmax(original_output, dim=1)[0, class_idx].item()
        
        # Get input dimensions
        _, _, H, W = input_tensor.shape
        
        # Initialize occlusion map
        occlusion_map = np.zeros((H, W))
        
        # Iterate over patches
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                # Create occluded input
                occluded_input = input_tensor.clone()
                occluded_input[:, :, i:i+patch_size, j:j+patch_size] = 0
                
                # Get prediction with occlusion
                with torch.no_grad():
                    output = self.model(occluded_input)
                    prob = F.softmax(output, dim=1)[0, class_idx].item()
                
                # Store probability drop
                prob_drop = original_prob - prob
                occlusion_map[i:i+patch_size, j:j+patch_size] = prob_drop
        
        # Normalize
        occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min() + 1e-8)
        
        return occlusion_map


def create_occlusion_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create occlusion sensitivity visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Generate occlusion map
    occlusion = OcclusionSensitivity(model)
    class_idx = config.class_names.index(predicted_label)
    occlusion_map = occlusion.generate_occlusion_map(input_tensor, class_idx)
    
    # Resize to display size
    occlusion_resized = cv2.resize(occlusion_map, (256, 256))
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    im = ax2.imshow(occlusion_resized, cmap='hot')
    ax2.set_title('Occlusion Sensitivity\n(Red = Important Regions)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    original_rgb = np.stack([np.array(original_img)]*3, axis=-1).astype(np.float32) / 255.0
    heatmap_color = cv2.applyColorMap(np.uint8(255 * occlusion_resized), cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * original_rgb + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    
    ax3.imshow(overlay)
    ax3.set_title('Occlusion Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# LIME IMPLEMENTATION
# ============================================================

class LimeExplainer:
    """
    LIME (Local Interpretable Model-Agnostic Explanations)
    Explains predictions by approximating locally with interpretable model.
    """
    def __init__(self, model, num_samples=1000, kernel_width=25):
        self.model = model
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def _generate_perturbations(self, image, num_samples):
        """Generate perturbed images by randomly masking superpixels"""
        H, W = image.shape
        perturbations = []
        
        for _ in range(num_samples):
            # Randomly mask patches
            perturbed = image.copy()
            mask = np.random.rand(H, W) > 0.5
            perturbed[mask] = 0
            perturbations.append(perturbed)
        
        return np.array(perturbations)
    
    def _kernel_distance(self, perturbation, original):
        """Compute distance between perturbation and original"""
        return np.exp(-np.sum((perturbation - original) ** 2) / (2 * self.kernel_width ** 2))
    
    def explain(self, input_tensor, class_idx):
        self.model.eval()
        
        # Convert to numpy
        image_np = input_tensor.squeeze().cpu().detach().numpy()
        
        # Generate perturbations
        perturbations = self._generate_perturbations(image_np, self.num_samples)
        
        # Get predictions for perturbations
        predictions = []
        weights = []
        
        for perturbation in perturbations:
            # Convert back to tensor
            pert_tensor = torch.FloatTensor(perturbation).unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(pert_tensor)
                prob = F.softmax(output, dim=1)[0, class_idx].item()
            
            predictions.append(prob)
            
            # Compute weight
            weight = self._kernel_distance(perturbation, image_np)
            weights.append(weight)
        
        # Fit linear model (simplified - using feature importance)
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Compute feature importance based on correlation
        feature_importance = np.zeros_like(image_np)
        
        for i in range(image_np.shape[0]):
            for j in range(image_np.shape[1]):
                # Correlation between pixel value and prediction
                pixel_values = perturbations[:, i, j]
                if np.std(pixel_values) > 0:
                    correlation = np.corrcoef(pixel_values, predictions)[0, 1]
                    feature_importance[i, j] = correlation if not np.isnan(correlation) else 0
        
        # Normalize
        feature_importance = np.abs(feature_importance)
        feature_importance = (feature_importance - feature_importance.min()) / (feature_importance.max() - feature_importance.min() + 1e-8)
        
        return feature_importance


def create_lime_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create LIME visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Generate LIME explanation
    lime = LimeExplainer(model, num_samples=500)
    class_idx = config.class_names.index(predicted_label)
    lime_map = lime.explain(input_tensor, class_idx)
    
    # Resize to display size
    lime_resized = cv2.resize(lime_map, (256, 256))
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    im = ax2.imshow(lime_resized, cmap='RdBu_r')
    ax2.set_title('LIME Explanation\n(Blue = Negative, Red = Positive)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    original_rgb = np.stack([np.array(original_img)]*3, axis=-1).astype(np.float32) / 255.0
    heatmap_color = cv2.applyColorMap(np.uint8(255 * lime_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * original_rgb + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    
    ax3.imshow(overlay)
    ax3.set_title('LIME Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# SHAP IMPLEMENTATION
# ============================================================

class ShapExplainer:
    """
    SHAP (SHapley Additive exPlanations)
    Uses game theory to assign importance values to features.
    Simplified implementation for image classification.
    """
    def __init__(self, model, background_samples=50):
        self.model = model
        self.background_samples = background_samples
    
    def explain(self, input_tensor, class_idx):
        self.model.eval()
        
        # Convert to numpy
        image_np = input_tensor.squeeze().cpu().detach().numpy()
        H, W = image_np.shape
        
        # Generate background samples (random noise)
        background = np.random.randn(self.background_samples, H, W)
        
        # Compute SHAP values using gradient-based approximation
        input_tensor_grad = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor_grad)
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients
        gradients = input_tensor_grad.grad.squeeze().cpu().detach().numpy()
        
        # SHAP values are gradients * (input - background)
        # Simplified: use gradients as SHAP values
        shap_values = gradients * image_np
        
        # Normalize
        shap_values = np.abs(shap_values)
        shap_values = (shap_values - shap_values.min()) / (shap_values.max() - shap_values.min() + 1e-8)
        
        return shap_values


def create_shap_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create SHAP visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Generate SHAP explanation
    shap = ShapExplainer(model)
    class_idx = config.class_names.index(predicted_label)
    shap_map = shap.explain(input_tensor, class_idx)
    
    # Resize to display size
    shap_resized = cv2.resize(shap_map, (256, 256))
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    im = ax2.imshow(shap_resized, cmap='coolwarm')
    ax2.set_title('SHAP Values\n(Blue = Negative, Red = Positive)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    original_rgb = np.stack([np.array(original_img)]*3, axis=-1).astype(np.float32) / 255.0
    heatmap_color = cv2.applyColorMap(np.uint8(255 * shap_resized), cv2.COLORMAP_COOLWARM)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * original_rgb + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    
    ax3.imshow(overlay)
    ax3.set_title('SHAP Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# PERMUTATION CHANNEL IMPORTANCE (PCI)
# ============================================================

class PermutationChannelImportance:
    """
    Permutation Channel Importance (PCI)
    For grayscale images: inverts the image (255 - pixel) and measures prediction impact
    For RGB images: permutes channels and measures prediction impact
    """
    def __init__(self, model):
        self.model = model
    
    def invert_grayscale_image(self, image_np):
        """Invert grayscale image: 255 - pixel_value"""
        return 255 - image_np
    
    def analyze_importance(self, input_tensor, class_idx):
        """
        Analyze importance by inverting the image (for grayscale)
        Returns the probability drop when image is inverted
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get original prediction
            original_output = self.model(input_tensor)
            original_prob = F.softmax(original_output, dim=1)[0, class_idx].item()
        
        # Convert to numpy and invert (for grayscale)
        image_np = input_tensor.squeeze().cpu().detach().numpy()
        inverted_np = self.invert_grayscale_image(image_np)
        
        # Convert back to tensor
        inverted_tensor = torch.FloatTensor(inverted_np).unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            # Get prediction with inverted image
            inverted_output = self.model(inverted_tensor)
            inverted_prob = F.softmax(inverted_output, dim=1)[0, class_idx].item()
        
        # Calculate importance as probability drop
        importance = original_prob - inverted_prob
        
        return {
            'original_prob': original_prob,
            'inverted_prob': inverted_prob,
            'importance': importance,
            'inverted_image': inverted_np
        }


def create_pci_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization=True, show_visualization=False):
    """Create Permutation Channel Importance visualization"""
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(next(model.parameters()).device)
    
    # Generate PCI analysis
    pci = PermutationChannelImportance(model)
    class_idx = config.class_names.index(predicted_label)
    results = pci.analyze_importance(input_tensor, class_idx)
    
    # Resize inverted image to display size
    inverted_resized = cv2.resize(results['inverted_image'], (256, 256))
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    ax2.imshow(inverted_resized, cmap='gray')
    ax2.set_title(f'Inverted Image\n(255 - pixel)')
    ax2.axis('off')
    
    # Show importance as a bar
    importance = results['importance']
    colors = ['green' if importance > 0 else 'red']
    ax3.bar(['Original', 'Inverted'], [results['original_prob'], results['inverted_prob']], 
            color=['blue', 'orange'], alpha=0.7)
    ax3.set_title(f'PCI Analysis\nImportance: {importance:.4f}')
    ax3.set_ylabel('Probability')
    ax3.set_ylim(0, 1)
    
    # Add text annotation
    ax3.text(0.5, 0.95, f'Prob Drop: {abs(importance):.4f}', 
             transform=ax3.transAxes, ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# GRAYSCALE IMAGE INVERSION
# ============================================================

def invert_grayscale_image(image_path, display_size=(256, 256)):
    """
    Invert grayscale image: 255 - pixel_value
    Converts black to white and white to black
    """
    # Load image
    img = Image.open(image_path).convert('L')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert: 255 - pixel_value
    inverted_array = 255 - img_array
    
    # Convert back to PIL Image
    inverted_img = Image.fromarray(inverted_array.astype(np.uint8))
    
    # Resize for display
    inverted_img = inverted_img.resize(display_size)
    
    return inverted_img


def create_inversion_visualization(image_path, true_label, predicted_label, confidence, save_visualization=True, show_visualization=False):
    """Create visualization showing original and inverted grayscale image"""
    # Load original image
    img = Image.open(image_path).convert('L')
    original_img = img.resize((256, 256))
    
    # Invert image
    inverted_img = invert_grayscale_image(image_path, display_size=(256, 256))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title(f'Original Image\nTrue: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}')
    ax1.axis('off')
    
    ax2.imshow(inverted_img, cmap='gray')
    ax2.set_title('Inverted Image\n(255 - pixel)\nBlack ↔ White')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if show_visualization:
        plt.show()
    
    if save_visualization:
        return fig
    else:
        plt.close(fig)
        return None


# ============================================================
# COMPREHENSIVE INTERPRETABILITY ANALYSIS
# ============================================================

def run_comprehensive_interpretability(model, image_path, true_label, predicted_label, confidence, config, save_dir="interpretability_results", save_visualization=True, show_visualization=False):
    """Run all interpretability methods on a single image"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE INTERPRETABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Image: {Path(image_path).name}")
    print(f"True: {true_label}, Predicted: {predicted_label}, Confidence: {confidence:.3f}")
    
    results = {}
    
    # 1. Guided Backpropagation
    try:
        print("\n1. Generating Guided Backpropagation...")
        fig = create_guided_backprop_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"guided_backprop_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['guided_backprop'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['guided_backprop'] = 'shown'
        else:
            results['guided_backprop'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 2. Guided Grad-CAM
    try:
        print("\n2. Generating Guided Grad-CAM...")
        fig = create_guided_gradcam_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"guided_gradcam_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['guided_gradcam'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['guided_gradcam'] = 'shown'
        else:
            results['guided_gradcam'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 3. Occlusion Sensitivity
    try:
        print("\n3. Generating Occlusion Sensitivity...")
        fig = create_occlusion_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"occlusion_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['occlusion'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['occlusion'] = 'shown'
        else:
            results['occlusion'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 4. LIME
    try:
        print("\n4. Generating LIME explanation...")
        fig = create_lime_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"lime_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['lime'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['lime'] = 'shown'
        else:
            results['lime'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 5. SHAP
    try:
        print("\n5. Generating SHAP explanation...")
        fig = create_shap_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"shap_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['shap'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['shap'] = 'shown'
        else:
            results['shap'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 6. Permutation Channel Importance (PCI)
    try:
        print("\n6. Generating Permutation Channel Importance...")
        fig = create_pci_visualization(model, image_path, true_label, predicted_label, confidence, config, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"pci_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['pci'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['pci'] = 'shown'
        else:
            results['pci'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # 7. Grayscale Inversion
    try:
        print("\n7. Generating Grayscale Inversion...")
        fig = create_inversion_visualization(image_path, true_label, predicted_label, confidence, save_visualization, show_visualization)
        if fig is not None and save_visualization:
            save_path = os.path.join(save_dir, f"inversion_{Path(image_path).stem}.png")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"   ✅ Saved to: {save_path}")
            results['inversion'] = save_path
        elif fig is not None and show_visualization:
            plt.close(fig)
            results['inversion'] = 'shown'
        else:
            results['inversion'] = 'generated'
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Completed {len(results)}/7 methods")
    print(f"{'='*60}")
    
    return results
