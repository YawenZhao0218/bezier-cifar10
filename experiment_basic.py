"""
experiment_basic.py - Basic Bézier curve experiments
Implements the core experiments comparing Settings A, B, and C
Modified to display results in mean±std format
Enhanced version with 5 experiments using different class combinations
Fixed Setting B to efficiently select images from the same class
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import random

from utils import PGDAttack, normalize_cifar10
from bezier_core import BezierAdversarialUnconstrained

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seeds at module level
set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model():
    """Load pretrained ResNet-18 model"""
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    
    if os.path.exists('resnet18_cifar10_best.pth'):
        checkpoint = torch.load('resnet18_cifar10_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
    else:
        print("ERROR: No pretrained model found! Please run train_model.py first.")
        exit(1)
    
    return model.to(device).eval()

def organize_images_by_class(dataloader, model, max_per_class=100):
    """Organize images by class"""
    images_by_class = defaultdict(list)
    
    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(device)
        label_tensor = label.to(device)
        
        with torch.no_grad():
            pred = model(normalize_cifar10(img_tensor)).argmax(dim=1)
            if pred == label_tensor:
                images_by_class[label.item()].append((img_tensor, idx))
                
                # Need at least max_per_class images per class
                if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == 10:
                    break
    
    return images_by_class

def evaluate_bezier_path(model, bezier_obj, delta1, theta, delta2, x1, x2, y1, y2, 
                        setting_type='A', num_points=50):
    """Evaluate Bézier path at multiple points (excluding endpoints)"""
    # Exclude endpoints to avoid evaluation bias
    t_values = torch.linspace(0.02, 0.98, num_points).to(device)
    
    if setting_type == 'A':
        # Setting A: Single image - both x1 and x2 are the same
        x2 = x1
        y2 = y1
    
    # Track success for each image
    success_x1 = 0
    success_x2 = 0
    success_both = 0
    
    with torch.no_grad():
        for t in t_values:
            delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
            delta_t = bezier_obj.project_norm_ball(delta_t)
            
            # Test on first image
            x1_adv = torch.clamp(x1 + delta_t, 0, 1)
            outputs1 = model(normalize_cifar10(x1_adv))
            pred1 = outputs1.argmax(dim=1).item()
            s1 = pred1 != y1.item()
            
            if setting_type == 'A':
                # For Setting A, only one image to test
                if s1:
                    success_x1 += 1
                    success_x2 += 1  # Same as x1 for Setting A
                    success_both += 1
            else:
                # Test on second image for Settings B and C
                x2_adv = torch.clamp(x2 + delta_t, 0, 1)
                outputs2 = model(normalize_cifar10(x2_adv))
                pred2 = outputs2.argmax(dim=1).item()
                s2 = pred2 != y2.item()
                
                if s1:
                    success_x1 += 1
                if s2:
                    success_x2 += 1
                if s1 and s2:
                    success_both += 1
    
    return {
        'success_rate_x1': success_x1 / num_points,
        'success_rate_x2': success_x2 / num_points,
        'success_rate_both': success_both / num_points,
        'success_rate_avg': (success_x1 + success_x2) / (2 * num_points)
    }

def print_comprehensive_results_table(results):
    """Print results in mean±std format"""
    print("\n" + "="*120)
    print("BASIC EXPERIMENTS - COMPREHENSIVE RESULTS (mean ± std)")
    print("="*120)
    
    # Detailed results for each norm and setting
    for norm in ['linf', 'l2', 'l1']:
        if not any(norm in results[setting] for setting in ['setting_A', 'setting_B', 'setting_C']):
            continue
            
        print(f"\n{'='*100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'='*100}")
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            if norm not in results[setting]:
                continue
                
            setting_name = {
                'setting_A': 'Setting A (Single Image)',
                'setting_B': 'Setting B (Same Class)',
                'setting_C': 'Setting C (Different Classes)'
            }[setting]
            
            print(f"\n{setting_name}:")
            data = results[setting][norm]
            
            if data['num_samples'] == 0:
                print("  No valid samples collected")
                continue
            
            # For Setting A
            if setting == 'setting_A':
                success_rates = data['success_rates']
                theta_norms = data['theta_norms']
                
                avg_success = np.mean(success_rates) * 100
                std_success = np.std(success_rates) * 100
                avg_theta = np.mean(theta_norms)
                std_theta = np.std(theta_norms)
                
                print(f"  Number of samples:     {data['num_samples']}")
                print(f"  Path success rate:     {avg_success:>6.1f} ± {std_success:<5.1f}%")
                print(f"  Control point θ/ε:     {avg_theta:>6.2f} ± {std_theta:<5.2f}")
            
            # For Settings B and C
            else:
                if data['detailed_results']:
                    # Extract individual metrics
                    x1_rates = [d['success_rate_x1'] for d in data['detailed_results']]
                    x2_rates = [d['success_rate_x2'] for d in data['detailed_results']]
                    both_rates = [d['success_rate_both'] for d in data['detailed_results']]
                    avg_rates = [d['success_rate_avg'] for d in data['detailed_results']]
                    theta_norms = data['theta_norms']
                    
                    # Calculate statistics
                    avg_x1 = np.mean(x1_rates) * 100
                    std_x1 = np.std(x1_rates) * 100
                    avg_x2 = np.mean(x2_rates) * 100
                    std_x2 = np.std(x2_rates) * 100
                    avg_both = np.mean(both_rates) * 100
                    std_both = np.std(both_rates) * 100
                    avg_avg = np.mean(avg_rates) * 100
                    std_avg = np.std(avg_rates) * 100
                    avg_theta = np.mean(theta_norms)
                    std_theta = np.std(theta_norms)
                    
                    print(f"  Number of samples:     {data['num_samples']}")
                    print(f"  Image 1 success rate:  {avg_x1:>6.1f} ± {std_x1:<5.1f}%")
                    print(f"  Image 2 success rate:  {avg_x2:>6.1f} ± {std_x2:<5.1f}%")
                    print(f"  Both images success:   {avg_both:>6.1f} ± {std_both:<5.1f}%")
                    print(f"  Average success rate:  {avg_avg:>6.1f} ± {std_avg:<5.1f}%")
                    print(f"  Control point θ/ε:     {avg_theta:>6.2f} ± {std_theta:<5.2f}")
    
    # Summary table
    print("\n" + "="*120)
    print("SUMMARY TABLE")
    print("="*120)
    print(f"\n{'Setting':<30} {'Norm':<8} {'Samples':<10} {'Img1':<18} {'Img2':<18} {'Both':<18} {'Average':<18}")
    print("-" * 110)
    
    for setting in ['setting_A', 'setting_B', 'setting_C']:
        setting_display = {
            'setting_A': 'Setting A (Single)',
            'setting_B': 'Setting B (Same Class)',
            'setting_C': 'Setting C (Diff Class)'
        }[setting]
        
        for norm in ['linf', 'l2', 'l1']:
            if norm not in results[setting]:
                continue
                
            norm_symbol = {'linf': 'ℓ∞', 'l2': 'ℓ₂', 'l1': 'ℓ₁'}[norm]
            data = results[setting][norm]
            
            if data['num_samples'] == 0:
                continue
            
            if setting == 'setting_A':
                success_rates = data['success_rates']
                avg_success = np.mean(success_rates) * 100
                std_success = np.std(success_rates) * 100
                
                print(f"{setting_display:<30} {norm_symbol:<8} {data['num_samples']:<10} "
                      f"{'N/A':<18} {'N/A':<18} "
                      f"{'N/A':<18} {avg_success:>6.1f}±{std_success:<5.1f}%")
            else:
                if data['detailed_results']:
                    x1_rates = [d['success_rate_x1'] for d in data['detailed_results']]
                    x2_rates = [d['success_rate_x2'] for d in data['detailed_results']]
                    both_rates = [d['success_rate_both'] for d in data['detailed_results']]
                    avg_rates = [d['success_rate_avg'] for d in data['detailed_results']]
                    
                    avg_x1 = np.mean(x1_rates) * 100
                    std_x1 = np.std(x1_rates) * 100
                    avg_x2 = np.mean(x2_rates) * 100
                    std_x2 = np.std(x2_rates) * 100
                    avg_both = np.mean(both_rates) * 100
                    std_both = np.std(both_rates) * 100
                    avg_avg = np.mean(avg_rates) * 100
                    std_avg = np.std(avg_rates) * 100
                    
                    img1_str = f"{avg_x1:.1f}±{std_x1:.1f}%"
                    img2_str = f"{avg_x2:.1f}±{std_x2:.1f}%"
                    both_str = f"{avg_both:.1f}±{std_both:.1f}%"
                    avg_str = f"{avg_avg:.1f}±{std_avg:.1f}%"
                    
                    print(f"{setting_display:<30} {norm_symbol:<8} {data['num_samples']:<10} "
                          f"{img1_str:<18} {img2_str:<18} "
                          f"{both_str:<18} {avg_str:<18}")
    
    print("\nLegend:")
    print("• Img1 / Img2: Success rate on individual images")
    print("• Both: Success rate when both images are fooled simultaneously")
    print("• Average: Average success rate across images")
    print("• Path evaluation excludes endpoints (t ∈ [0.02, 0.98])")

def print_additional_insights(results):
    """Print additional insights with statistical analysis"""
    print("\n" + "="*120)
    print("ADDITIONAL INSIGHTS")
    print("="*120)
    
    # Compare norms with statistics
    print("\n1. Norm Comparison (Average Success Rates):")
    for setting in ['setting_A', 'setting_B', 'setting_C']:
        setting_name = {
            'setting_A': 'Setting A',
            'setting_B': 'Setting B',
            'setting_C': 'Setting C'
        }[setting]
        
        print(f"\n   {setting_name}:")
        norm_performances = []
        
        for norm in ['linf', 'l2', 'l1']:
            if norm in results[setting] and results[setting][norm]['num_samples'] > 0:
                if setting == 'setting_A':
                    rates = results[setting][norm]['success_rates']
                else:
                    rates = [d['success_rate_avg'] for d in results[setting][norm]['detailed_results']]
                
                avg_rate = np.mean(rates) * 100
                std_rate = np.std(rates) * 100
                norm_performances.append((norm, avg_rate, std_rate))
                print(f"   - {norm.upper()}: {avg_rate:.1f} ± {std_rate:.1f}%")
        
        if norm_performances:
            best_norm = max(norm_performances, key=lambda x: x[1])
            print(f"   → Best: {best_norm[0].upper()} ({best_norm[1]:.1f} ± {best_norm[2]:.1f}%)")
    
    # Theta norm analysis with statistics
    print("\n2. Control Point (θ) Magnitude Analysis:")
    for norm in ['linf', 'l2', 'l1']:
        print(f"\n   {norm.upper()} norm:")
        theta_data = []
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            if norm in results[setting] and results[setting][norm]['num_samples'] > 0:
                theta_norms = results[setting][norm]['theta_norms']
                avg_theta = np.mean(theta_norms)
                std_theta = np.std(theta_norms)
                setting_name = setting.replace('_', ' ').title()
                theta_data.append((setting_name, avg_theta, std_theta))
                print(f"   - {setting_name}: θ/ε = {avg_theta:.2f} ± {std_theta:.2f}")
        
        if theta_data:
            avg_across_settings = np.mean([d[1] for d in theta_data])
            print(f"   → Average across settings: {avg_across_settings:.2f}×ε")
    
    # Statistical summary
    print("\n3. Statistical Summary:")
    total_samples = 0
    for setting in ['setting_A', 'setting_B', 'setting_C']:
        for norm in ['linf', 'l2', 'l1']:
            if norm in results[setting]:
                total_samples += results[setting][norm]['num_samples']
    
    print(f"   Total samples collected: {total_samples}")
    print(f"   Settings tested: A (Single Image), B (Same Class), C (Different Classes)")
    print(f"   Norms evaluated: L∞, L₂, L₁")
    print(f"   Path points sampled: 50 (excluding endpoints)")
    print(f"   PGD attack iterations: 40")
    print(f"   Number of experiments: 5 (with different class combinations)")
    print(f"   Bézier optimization learning rate: 0.01")

def run_single_experiment(experiment_id, images_by_class, model, num_samples_per_setting=25):
    """Run a single experiment with specific class selections"""
    norms = ['linf', 'l2', 'l1']
    epsilons = {
        'linf': 8/255,
        'l2': 0.5,
        'l1': 10.0
    }
    
    pgd_steps = 40  # Changed from 20 to 40
    
    # Community standard alpha factors for 40-step PGD
    pgd_alpha_factors = {
        'linf': 4.0,    # α = ε/4 (community standard for 40 steps)
        'l2': 5.0,      # α = ε/5 (moderate attack)
        'l1': 10.0      # α = ε/10 (stable optimization)
    }
    
    results = {
        'setting_A': {},
        'setting_B': {},
        'setting_C': {}
    }
    
    # Get class IDs
    class_ids = list(images_by_class.keys())
    
    # Rotate class selection based on experiment_id to ensure different combinations
    class_offset = experiment_id * 2
    
    print(f"\n  Experiment {experiment_id + 1}/5 - Using class offset {class_offset}")
    
    for norm in norms:
        print(f"\n  Testing {norm.upper()} norm (ε={epsilons[norm]})...")
        
        eps = epsilons[norm]
        alpha = eps / pgd_alpha_factors[norm]  # Use community standard formula
        
        pgd_attack = PGDAttack(model, eps=eps, alpha=alpha, 
                              num_iter=pgd_steps, norm=norm)
        
        bezier = BezierAdversarialUnconstrained(model, norm=norm, eps=eps, 
                                               lr=0.01, num_iter=30)  # lr=0.01
        
        # Initialize results for this norm
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            results[setting][norm] = {
                'success_rates': [],
                'detailed_results': [],
                'theta_norms': [],
                'num_samples': 0,
                'avg_success_rate': 0,
                'avg_theta_norm': 0
            }
        
        samples_A = 0
        samples_B = 0
        samples_C = 0
        
        pbar = tqdm(total=num_samples_per_setting * 3, 
                   desc=f"  Exp{experiment_id+1} {norm}")
        
        attempt_count = 0
        max_attempts = 1000  # Prevent infinite loop
        
        while (samples_A < num_samples_per_setting or 
               samples_B < num_samples_per_setting or 
               samples_C < num_samples_per_setting) and attempt_count < max_attempts:
            
            attempt_count += 1
            
            # Setting A: Single Image
            if samples_A < num_samples_per_setting:
                # Choose a class for Setting A
                class_a = class_ids[(attempt_count + class_offset) % len(class_ids)]
                
                if len(images_by_class[class_a]) >= 1:
                    # Select one image
                    img_idx = attempt_count % len(images_by_class[class_a])
                    x = images_by_class[class_a][img_idx][0]
                    y = torch.tensor([class_a]).to(device)
                    
                    # Verify clean accuracy
                    with torch.no_grad():
                        pred = model(normalize_cifar10(x)).argmax(dim=1)
                        if pred != y:
                            continue
                    
                    # Generate two perturbations for the same image
                    delta1_A = pgd_attack.perturb(x, y)
                    delta2_A = pgd_attack.perturb(x, y)
                    
                    # Verify both endpoints work
                    with torch.no_grad():
                        x_adv_d1 = torch.clamp(x + delta1_A, 0, 1)
                        x_adv_d2 = torch.clamp(x + delta2_A, 0, 1)
                        pred_d1 = model(normalize_cifar10(x_adv_d1)).argmax(dim=1)
                        pred_d2 = model(normalize_cifar10(x_adv_d2)).argmax(dim=1)
                        
                        if pred_d1 == y or pred_d2 == y:
                            continue
                    
                    # Optimize Bézier path
                    theta_A, _, _, theta_norms = bezier.optimize_setting_A(x, y, delta1_A, delta2_A)
                    
                    # Evaluate path
                    eval_results = evaluate_bezier_path(
                        model, bezier, delta1_A, theta_A, delta2_A, 
                        x, x, y, y, setting_type='A'
                    )
                    
                    results['setting_A'][norm]['success_rates'].append(eval_results['success_rate_avg'])
                    results['setting_A'][norm]['detailed_results'].append(eval_results)
                    results['setting_A'][norm]['theta_norms'].append(theta_norms[-1])
                    samples_A += 1
                    pbar.update(1)
            
            # Setting B: Same Class - FIXED LOGIC
            if samples_B < num_samples_per_setting:
                # Choose a class for Setting B
                class_b = class_ids[(attempt_count + class_offset + 1) % len(class_ids)]
                
                if len(images_by_class[class_b]) >= 2:
                    # Select two different images from the same class
                    img1_idx = (attempt_count * 2) % len(images_by_class[class_b])
                    img2_idx = (attempt_count * 2 + 1) % len(images_by_class[class_b])
                    
                    # Ensure different images
                    if img1_idx == img2_idx:
                        img2_idx = (img2_idx + 1) % len(images_by_class[class_b])
                    
                    x1 = images_by_class[class_b][img1_idx][0]
                    x2 = images_by_class[class_b][img2_idx][0]
                    y = torch.tensor([class_b]).to(device)
                    
                    # Verify clean accuracy
                    with torch.no_grad():
                        pred1 = model(normalize_cifar10(x1)).argmax(dim=1)
                        pred2 = model(normalize_cifar10(x2)).argmax(dim=1)
                        
                        if pred1 != y or pred2 != y:
                            continue
                    
                    # Generate perturbations for each image
                    delta1 = pgd_attack.perturb(x1, y)
                    delta2 = pgd_attack.perturb(x2, y)
                    
                    # Verify endpoints work
                    with torch.no_grad():
                        pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
                        pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
                        
                        if pred1 == y or pred2 == y:
                            continue
                    
                    # Optimize Bézier path
                    theta_B, _, _, theta_norms = bezier.optimize_setting_B(x1, x2, y, delta1, delta2)
                    
                    # Evaluate path
                    eval_results = evaluate_bezier_path(
                        model, bezier, delta1, theta_B, delta2, 
                        x1, x2, y, y, setting_type='B'
                    )
                    
                    results['setting_B'][norm]['success_rates'].append(eval_results['success_rate_both'])
                    results['setting_B'][norm]['detailed_results'].append(eval_results)
                    results['setting_B'][norm]['theta_norms'].append(theta_norms[-1])
                    samples_B += 1
                    pbar.update(1)
            
            # Setting C: Different Classes
            if samples_C < num_samples_per_setting:
                # Choose two different classes
                class_c1 = class_ids[(attempt_count + class_offset) % len(class_ids)]
                class_c2 = class_ids[(attempt_count + class_offset + 1) % len(class_ids)]
                
                if class_c1 != class_c2 and len(images_by_class[class_c1]) >= 1 and len(images_by_class[class_c2]) >= 1:
                    # Select one image from each class
                    img1_idx = attempt_count % len(images_by_class[class_c1])
                    img2_idx = attempt_count % len(images_by_class[class_c2])
                    
                    x1 = images_by_class[class_c1][img1_idx][0]
                    x2 = images_by_class[class_c2][img2_idx][0]
                    y1 = torch.tensor([class_c1]).to(device)
                    y2 = torch.tensor([class_c2]).to(device)
                    
                    # Verify clean accuracy
                    with torch.no_grad():
                        pred1 = model(normalize_cifar10(x1)).argmax(dim=1)
                        pred2 = model(normalize_cifar10(x2)).argmax(dim=1)
                        
                        if pred1 != y1 or pred2 != y2:
                            continue
                    
                    # Generate perturbations for each image
                    delta1 = pgd_attack.perturb(x1, y1)
                    delta2 = pgd_attack.perturb(x2, y2)
                    
                    # Verify endpoints work
                    with torch.no_grad():
                        pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
                        pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
                        
                        if pred1 == y1 or pred2 == y2:
                            continue
                    
                    # Optimize Bézier path
                    theta_C, _, _, theta_norms = bezier.optimize_setting_C(x1, x2, y1, y2, delta1, delta2)
                    
                    # Evaluate path
                    eval_results = evaluate_bezier_path(
                        model, bezier, delta1, theta_C, delta2, 
                        x1, x2, y1, y2, setting_type='C'
                    )
                    
                    results['setting_C'][norm]['success_rates'].append(eval_results['success_rate_both'])
                    results['setting_C'][norm]['detailed_results'].append(eval_results)
                    results['setting_C'][norm]['theta_norms'].append(theta_norms[-1])
                    samples_C += 1
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate averages
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            if results[setting][norm]['success_rates']:
                results[setting][norm]['avg_success_rate'] = \
                    np.mean(results[setting][norm]['success_rates'])
                results[setting][norm]['num_samples'] = \
                    len(results[setting][norm]['success_rates'])
                results[setting][norm]['avg_theta_norm'] = \
                    np.mean(results[setting][norm]['theta_norms'])
        
        # Print progress summary
        print(f"    Collected samples - A: {samples_A}, B: {samples_B}, C: {samples_C}")
    
    return results

def aggregate_experiment_results(all_experiment_results):
    """Aggregate results from multiple experiments"""
    aggregated = {
        'setting_A': {},
        'setting_B': {},
        'setting_C': {}
    }
    
    norms = ['linf', 'l2', 'l1']
    
    for setting in ['setting_A', 'setting_B', 'setting_C']:
        for norm in norms:
            # Collect all results for this setting and norm
            all_success_rates = []
            all_detailed_results = []
            all_theta_norms = []
            
            for exp_results in all_experiment_results:
                if norm in exp_results[setting] and exp_results[setting][norm]['num_samples'] > 0:
                    all_success_rates.extend(exp_results[setting][norm]['success_rates'])
                    all_detailed_results.extend(exp_results[setting][norm]['detailed_results'])
                    all_theta_norms.extend(exp_results[setting][norm]['theta_norms'])
            
            if all_success_rates:
                aggregated[setting][norm] = {
                    'success_rates': all_success_rates,
                    'detailed_results': all_detailed_results,
                    'theta_norms': all_theta_norms,
                    'num_samples': len(all_success_rates),
                    'avg_success_rate': np.mean(all_success_rates),
                    'avg_theta_norm': np.mean(all_theta_norms)
                }
            else:
                aggregated[setting][norm] = {
                    'success_rates': [],
                    'detailed_results': [],
                    'theta_norms': [],
                    'num_samples': 0,
                    'avg_success_rate': 0,
                    'avg_theta_norm': 0
                }
    
    return aggregated

def run_basic_experiments():
    """Run basic Bézier curve experiments with 5 repetitions"""
    model = load_model()
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    print("\nOrganizing images by class...")
    images_by_class = organize_images_by_class(testloader, model, max_per_class=100)
    
    print(f"Found images for {len(images_by_class)} classes")
    for class_id, images in images_by_class.items():
        print(f"  Class {class_id}: {len(images)} images")
    
    num_samples_per_setting = 25
    num_experiments = 5
    
    print("\nStarting experiments...")
    print(f"Target: {num_samples_per_setting} samples per setting per norm")
    print(f"Number of experiments: {num_experiments}")
    print(f"PGD attack iterations: 40 (with community standard α = ε/4 for L∞)")
    print(f"Bézier optimization: 30 iterations with lr=0.01")
    print("="*80)
    
    all_experiment_results = []
    
    for exp_id in range(num_experiments):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_id + 1} / {num_experiments}")
        print(f"{'='*80}")
        
        exp_results = run_single_experiment(exp_id, images_by_class, model, num_samples_per_setting)
        all_experiment_results.append(exp_results)
    
    # Aggregate results from all experiments
    print("\nAggregating results from all experiments...")
    aggregated_results = aggregate_experiment_results(all_experiment_results)
    
    return aggregated_results

def save_results(results):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'bezier_basic_results_{timestamp}.json'
    
    # Convert to serializable format
    results_serializable = {}
    for setting in results:
        results_serializable[setting] = {}
        for norm in results[setting]:
            results_serializable[setting][norm] = {
                'avg_success_rate': float(results[setting][norm]['avg_success_rate']),
                'num_samples': int(results[setting][norm]['num_samples']),
                'success_rates': [float(x) for x in results[setting][norm]['success_rates']],
                'avg_theta_norm': float(results[setting][norm].get('avg_theta_norm', 0)),
                'theta_norms': [float(x) for x in results[setting][norm]['theta_norms']],
                'detailed_results': results[setting][norm]['detailed_results']
            }
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"\nResults saved to {filename}")
    return filename

if __name__ == "__main__":
    print("Bézier Adversarial Curves - Basic Experiments")
    print("="*80)
    print("\nThree Experimental Settings:")
    print("1. Setting A (Single Image)")
    print("   • Generate two different adversarial perturbations for the same image")
    print("   • Optimize the Bézier path connecting them")
    print("\n2. Setting B (Same Class)")
    print("   • Select two images from the same class")
    print("   • Find a perturbation path effective for both images")
    print("\n3. Setting C (Different Classes)")
    print("   • Select two images from different classes")
    print("   • Find a perturbation path effective for both images")
    print("\nExperiment Configuration:")
    print("   • 5 experiments with different class combinations")
    print("   • 25 samples per setting per experiment")
    print("   • PGD attack with 40 iterations")
    print("   • Community standard α values: L∞=ε/4, L₂=ε/5, L₁=ε/10")
    print("   • Bézier optimization: 30 iterations with lr=0.01")
    print("   • Results displayed in mean ± std format")
    print("="*80)
    
    if not os.path.exists('resnet18_cifar10_best.pth'):
        print("\nERROR: No trained model found!")
        print("Please run 'python train_model.py' first.")
        exit(1)
    
    # Run experiments
    results = run_basic_experiments()
    
    # Print results
    print_comprehensive_results_table(results)
    print_additional_insights(results)
    
    # Save results
    results_file = save_results(results)
    
    print(f"\nExperiments completed!")
    print(f"Results saved to: {results_file}")
    print("\nKey features:")
    print("• Results displayed in mean ± std format")
    print("• Comprehensive statistical analysis")
    print("• Reproducible with random seed")
    print("• 5 experiments × 25 samples per setting per norm")
    print("• PGD attack with 40 iterations and community standard α values")
    print("• Bézier optimization with learning rate 0.01")
    print("• Fixed Setting B to efficiently select from same class")