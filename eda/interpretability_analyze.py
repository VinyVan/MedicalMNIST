"""
Analyze functions for interpretability methods in Medical MNIST notebook
Each function categorizes predictions and generates visualizations for different cases
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_guided_backprop(models, test_paths, true_labels, predictions, probabilities, 
                            class_names, config, save_dir="guided_backprop_results", 
                            seed=42, n_samples_per_category=3):
    """
    Analyze predictions using Guided Backpropagation
    """
    from interpretability_methods import create_guided_backprop_visualization
    
    print("=" * 60)
    print("GUIDED BACKPROPAGATION ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_guided_backprop_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Guided Backpropagation analysis complete. Results saved to: {save_dir}/")


def analyze_guided_gradcam(models, test_paths, true_labels, predictions, probabilities,
                          class_names, config, save_dir="guided_gradcam_results",
                          seed=42, n_samples_per_category=3):
    """
    Analyze predictions using Guided Grad-CAM
    """
    from interpretability_methods import create_guided_gradcam_visualization
    
    print("=" * 60)
    print("GUIDED GRAD-CAM ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_guided_gradcam_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Guided Grad-CAM analysis complete. Results saved to: {save_dir}/")


def analyze_occlusion(models, test_paths, true_labels, predictions, probabilities,
                     class_names, config, save_dir="occlusion_results",
                     seed=42, n_samples_per_category=3):
    """
    Analyze predictions using Occlusion Sensitivity
    """
    from interpretability_methods import create_occlusion_visualization
    
    print("=" * 60)
    print("OCCLUSION SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_occlusion_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Occlusion Sensitivity analysis complete. Results saved to: {save_dir}/")


def analyze_lime(models, test_paths, true_labels, predictions, probabilities,
               class_names, config, save_dir="lime_results",
               seed=42, n_samples_per_category=3):
    """
    Analyze predictions using LIME
    """
    from interpretability_methods import create_lime_visualization
    
    print("=" * 60)
    print("LIME ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_lime_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ LIME analysis complete. Results saved to: {save_dir}/")


def analyze_shap(models, test_paths, true_labels, predictions, probabilities,
              class_names, config, save_dir="shap_results",
              seed=42, n_samples_per_category=3):
    """
    Analyze predictions using SHAP
    """
    from interpretability_methods import create_shap_visualization
    
    print("=" * 60)
    print("SHAP ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_shap_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ SHAP analysis complete. Results saved to: {save_dir}/")


def analyze_pci(models, test_paths, true_labels, predictions, probabilities,
              class_names, config, save_dir="pci_results",
              seed=42, n_samples_per_category=3):
    """
    Analyze predictions using Permutation Channel Importance (PCI)
    """
    from interpretability_methods import create_pci_visualization
    
    print("=" * 60)
    print("PERMUTATION CHANNEL IMPORTANCE (PCI) ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    model = models[0]
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_pci_visualization(
                    model, test_paths[idx], true_label, pred_label, confidence, config
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ PCI analysis complete. Results saved to: {save_dir}/")


def analyze_inversion(test_paths, true_labels, predictions, probabilities,
                     class_names, save_dir="inversion_results",
                     seed=42, n_samples_per_category=3):
    """
    Analyze predictions using Grayscale Image Inversion
    """
    from interpretability_methods import create_inversion_visualization
    
    print("=" * 60)
    print("GRAYSCALE INVERSION ANALYSIS")
    print("=" * 60)
    
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # Categorize cases
    cases = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'wrong_high_conf': [],
        'wrong_low_conf': []
    }
    
    for i in range(len(predictions)):
        confidence = np.max(probabilities[i])
        is_correct = predictions[i] == true_labels[i]
        
        if is_correct and confidence > 0.9:
            cases['correct_high_conf'].append(i)
        elif is_correct and confidence < 0.7:
            cases['correct_low_conf'].append(i)
        elif not is_correct and confidence > 0.9:
            cases['wrong_high_conf'].append(i)
        elif not is_correct and confidence < 0.7:
            cases['wrong_low_conf'].append(i)
    
    print(f"\n📊 Case Distribution:")
    for case_type, indices in cases.items():
        print(f"  {case_type}: {len(indices)} cases")
    
    for case_type, indices in cases.items():
        if len(indices) == 0:
            print(f"\n⚠️  No {case_type} cases found")
            continue
        
        n_samples = min(n_samples_per_category, len(indices))
        print(f"\n{'='*50}")
        print(f"🔍 {case_type.upper()} — showing {n_samples} examples")
        print(f"{'='*50}")
        
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        for idx in sample_indices:
            true_label = class_names[true_labels[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = np.max(probabilities[idx])
            
            print(f"\n  📁 Image: {Path(test_paths[idx]).name}")
            print(f"  ✅ True: {true_label}, 🔮 Predicted: {pred_label}, 📈 Confidence: {confidence:.3f}")
            
            try:
                fig = create_inversion_visualization(
                    test_paths[idx], true_label, pred_label, confidence
                )
                save_path = os.path.join(save_dir, f"{case_type}_{Path(test_paths[idx]).stem}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  💾 Saved to: {save_path}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Grayscale Inversion analysis complete. Results saved to: {save_dir}/")


def run_all_interpretability_analyses(models, test_paths, true_labels, predictions, 
                                      probabilities, class_names, config,
                                      base_save_dir="interpretability_analyses",
                                      seed=42, n_samples_per_category=3):
    """
    Run all interpretability analyses with categorization
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL INTERPRETABILITY ANALYSES")
    print("=" * 70)
    
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Run each analysis
    analyze_guided_backprop(models, test_paths, true_labels, predictions, probabilities,
                           class_names, config, 
                           save_dir=os.path.join(base_save_dir, "guided_backprop"),
                           seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_guided_gradcam(models, test_paths, true_labels, predictions, probabilities,
                          class_names, config,
                          save_dir=os.path.join(base_save_dir, "guided_gradcam"),
                          seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_occlusion(models, test_paths, true_labels, predictions, probabilities,
                     class_names, config,
                     save_dir=os.path.join(base_save_dir, "occlusion"),
                     seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_lime(models, test_paths, true_labels, predictions, probabilities,
               class_names, config,
               save_dir=os.path.join(base_save_dir, "lime"),
               seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_shap(models, test_paths, true_labels, predictions, probabilities,
              class_names, config,
              save_dir=os.path.join(base_save_dir, "shap"),
              seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_pci(models, test_paths, true_labels, predictions, probabilities,
              class_names, config,
              save_dir=os.path.join(base_save_dir, "pci"),
              seed=seed, n_samples_per_category=n_samples_per_category)
    
    analyze_inversion(test_paths, true_labels, predictions, probabilities,
                     class_names,
                     save_dir=os.path.join(base_save_dir, "inversion"),
                     seed=seed, n_samples_per_category=n_samples_per_category)
    
    print("\n" + "=" * 70)
    print("✅ ALL INTERPRETABILITY ANALYSES COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {base_save_dir}/")
