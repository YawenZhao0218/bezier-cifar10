"""
experiment_multi_image.py - Multi-image optimization experiments
Tests the effect of auxiliary images on optimization
Modified to use 0,5,10,15,20,25 auxiliary images
Aligned with experiment_basic.py: 40 PGD iterations, community standard alpha, 5 experiments
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
import matplotlib.pyplot as plt
import random

from utils import PGDAttack, normalize_cifar10
from bezier_core import BezierAdversarialMultiImage

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
    """Load pretrained ResNet-18"""
    model = resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    
    checkpoint = torch.load('resnet18_cifar10_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model with accuracy: {checkpoint['acc']:.2f}%")
    
    return model.to(device).eval()

def organize_images_by_class(dataloader, model, max_per_class=300):
    """Organize images by class - increased to 300 for more auxiliary images"""
    images_by_class = defaultdict(list)
    
    for idx, (img, label) in enumerate(dataloader):
        img_tensor = img.to(device)
        label_tensor = label.to(device)
        
        with torch.no_grad():
            pred = model(normalize_cifar10(img_tensor)).argmax(dim=1)
            if pred == label_tensor:
                images_by_class[label.item()].append((img_tensor, idx))
                
                if all(len(imgs) >= max_per_class for imgs in images_by_class.values()) and len(images_by_class) == 10:
                    break
    
    return images_by_class

def get_fixed_test_set_for_setting(images_by_class, setting, class_ids, num_test_images=100):
    """
    Get fixed test set for a specific setting
    Reserve first 30 images for training (main + up to 25 auxiliary)
    Use next num_test_images for testing
    """
    train_reserve = 30  # Increased for up to 25 auxiliary images
    
    if setting == 'A' or setting == 'B':
        # Single class for settings A and B
        class_id = class_ids[0]
        test_images = []
        test_labels = []
        
        # Skip first train_reserve images and take next num_test_images
        for i in range(train_reserve, min(train_reserve + num_test_images, len(images_by_class[class_id]))):
            test_images.append(images_by_class[class_id][i][0])
            test_labels.append(torch.tensor([class_id]).to(device))
        
        return test_images, test_labels
    
    elif setting == 'C':
        # Two classes for setting C
        class_id1, class_id2 = class_ids
        test_images = []
        test_labels = []
        
        # Get num_test_images/2 from each class
        images_per_class = num_test_images // 2
        
        # From first class
        for i in range(train_reserve, min(train_reserve + images_per_class, len(images_by_class[class_id1]))):
            test_images.append(images_by_class[class_id1][i][0])
            test_labels.append(torch.tensor([class_id1]).to(device))
        
        # From second class
        for i in range(train_reserve, min(train_reserve + images_per_class, len(images_by_class[class_id2]))):
            test_images.append(images_by_class[class_id2][i][0])
            test_labels.append(torch.tensor([class_id2]).to(device))
        
        return test_images, test_labels

def evaluate_transferability_multi(model, bezier_obj, delta1, theta, delta2, 
                                  transfer_images, transfer_labels, num_path_points=50):
    """Evaluate transferability with fixed 50 points"""
    t_values = torch.linspace(0.01, 0.99, num_path_points).to(device)
    
    results = {
        'delta1_success': [],
        'delta2_success': [],
        'path_success': [],
        'best_path_success': [],
        'any_path_success': []
    }
    
    with torch.no_grad():
        for x, y in zip(transfer_images, transfer_labels):
            # Test endpoints
            x_adv = torch.clamp(x + delta1, 0, 1)
            pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
            delta1_success = (pred != y).item()
            results['delta1_success'].append(delta1_success)
            
            x_adv = torch.clamp(x + delta2, 0, 1)
            pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
            delta2_success = (pred != y).item()
            results['delta2_success'].append(delta2_success)
            
            # Test path
            path_success_count = 0
            
            for t in t_values:
                delta_t = bezier_obj.bezier_curve(delta1, theta, delta2, t)
                delta_t = bezier_obj.project_norm_ball(delta_t)
                
                x_adv = torch.clamp(x + delta_t, 0, 1)
                pred = model(normalize_cifar10(x_adv)).argmax(dim=1)
                
                if pred != y:
                    path_success_count += 1
            
            results['best_path_success'].append(path_success_count)
            results['any_path_success'].append(path_success_count > 0)
    
    stats = {
        'delta1_transfer_rate': np.mean(results['delta1_success']),
        'delta2_transfer_rate': np.mean(results['delta2_success']),
        'endpoints_avg_transfer_rate': np.mean(results['delta1_success'] + results['delta2_success']),
        'any_path_point_transfer_rate': np.mean(results['any_path_success']),
        'avg_successful_path_points': np.mean(results['best_path_success']),
    }
    
    rescued_by_path = []
    for i in range(len(transfer_images)):
        if not results['delta1_success'][i] and not results['delta2_success'][i] and results['any_path_success'][i]:
            rescued_by_path.append(i)
    
    stats['rescue_rate'] = len(rescued_by_path) / len(transfer_images) if len(transfer_images) > 0 else 0
    
    return stats

def generate_valid_endpoints_setting_A(x_main, y_main, pgd_attack, model, max_attempts=50):
    """Generate valid endpoints for Setting A (single image)"""
    for attempt in range(max_attempts):
        delta1 = pgd_attack.perturb(x_main, y_main)
        delta2 = pgd_attack.perturb(x_main, y_main)
        
        # Verify both endpoints work on the same image
        with torch.no_grad():
            x_adv_d1 = torch.clamp(x_main + delta1, 0, 1)
            x_adv_d2 = torch.clamp(x_main + delta2, 0, 1)
            pred_d1 = model(normalize_cifar10(x_adv_d1)).argmax(dim=1)
            pred_d2 = model(normalize_cifar10(x_adv_d2)).argmax(dim=1)
            
            if pred_d1 != y_main and pred_d2 != y_main:
                return delta1, delta2, True
    
    return None, None, False

def generate_valid_endpoints_setting_BC(x1, x2, y1, y2, pgd_attack, model, max_attempts=50):
    """Generate valid endpoints for Settings B and C"""
    for attempt in range(max_attempts):
        delta1 = pgd_attack.perturb(x1, y1)
        delta2 = pgd_attack.perturb(x2, y2)
        
        # Verify endpoints
        with torch.no_grad():
            pred1 = model(normalize_cifar10(torch.clamp(x1 + delta1, 0, 1))).argmax(1)
            pred2 = model(normalize_cifar10(torch.clamp(x2 + delta2, 0, 1))).argmax(1)
            
            if pred1 != y1 and pred2 != y2:
                return delta1, delta2, True
    
    return None, None, False

def print_multi_image_results_extended(results):
    """Print results for extended auxiliary image configurations"""
    print("\n" + "="*120)
    print("MULTI-IMAGE OPTIMIZATION RESULTS - EXTENDED (0,5,10,15,20,25 auxiliary images)")
    print("="*120)
    
    for norm in ['linf', 'l2', 'l1']:
        if norm not in results:
            continue
        
        print(f"\n{'='*100}")
        print(f"{norm.upper()} NORM RESULTS")
        print(f"{'='*100}")
        
        for setting in ['setting_A', 'setting_B', 'setting_C']:
            setting_name = {
                'setting_A': 'Setting A (Single Image)',
                'setting_B': 'Setting B (Same Class)',
                'setting_C': 'Setting C (Different Classes)'
            }[setting]
            
            print(f"\n{setting_name}:")
            
            if not results[norm][setting] or all(not v for v in results[norm][setting].values()):
                print("  No valid results obtained for this setting.")
                continue
            
            print(f"{'Num Auxiliary':<15} {'Endpoint Avg':<20} {'Path Success':<20} "
                  f"{'Improvement':<20} {'Rescue Rate':<15}")
            print("-" * 90)
            
            setting_results = results[norm][setting]
            
            for num_additional in sorted(setting_results.keys()):
                if setting_results[num_additional]:
                    stats_list = setting_results[num_additional]
                    
                    # Calculate mean and std
                    avg_endpoints = np.mean([s['endpoints_avg_transfer_rate'] for s in stats_list]) * 100
                    std_endpoints = np.std([s['endpoints_avg_transfer_rate'] for s in stats_list]) * 100
                    
                    avg_path = np.mean([s['any_path_point_transfer_rate'] for s in stats_list]) * 100
                    std_path = np.std([s['any_path_point_transfer_rate'] for s in stats_list]) * 100
                    
                    avg_rescue = np.mean([s['rescue_rate'] for s in stats_list]) * 100
                    std_rescue = np.std([s['rescue_rate'] for s in stats_list]) * 100
                    
                    improvement = avg_path - avg_endpoints
                    
                    # Format output with mean ± std
                    endpoint_str = f"{avg_endpoints:.1f}±{std_endpoints:.1f}%"
                    path_str = f"{avg_path:.1f}±{std_path:.1f}%"
                    
                    if improvement > 0:
                        imp_str = f"\033[92m+{improvement:.1f}%\033[0m"
                    else:
                        imp_str = f"\033[91m{improvement:.1f}%\033[0m"
                    
                    rescue_str = f"{avg_rescue:.1f}±{std_rescue:.1f}%"
                    
                    print(f"{num_additional:<15} {endpoint_str:<20} {path_str:<20} "
                          f"{imp_str:<29} {rescue_str:<15}")

def run_multi_image_experiments_extended():
    """Run multi-image experiments with 0,5,10,15,20,25 auxiliary images"""
    # Set random seeds again to ensure reproducibility
    set_random_seeds(42)
    
    norms = ['linf', 'l2', 'l1']
    epsilons = {
        'linf': 8/255,
        'l2': 0.5,
        'l1': 10.0
    }
    
    # Modified: Use same parameters as experiment_basic.py
    pgd_steps = 40  # Changed from 20 to 40
    
    # Community standard alpha factors for 40-step PGD (same as experiment_basic)
    pgd_alpha_factors = {
        'linf': 4.0,    # α = ε/4 (community standard for 40 steps)
        'l2': 5.0,      # α = ε/5 (moderate attack)
        'l1': 10.0      # α = ε/10 (stable optimization)
    }
    
    # Extended auxiliary image configurations
    auxiliary_configs = [0, 5, 10, 15, 20, 25]
    
    model = load_model()
    
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    print("Organizing images by class...")
    images_by_class = organize_images_by_class(testloader, model, max_per_class=300)
    
    # Check availability
    print("\nImage availability per class:")
    for class_id, images in images_by_class.items():
        print(f"  Class {class_id}: {len(images)} images")
    
    num_test_images = 100
    print(f"\nUsing {num_test_images} images for testing (fixed across all experiments)")
    print(f"Using auxiliary images: {auxiliary_configs}")
    
    all_results = {
        'generation_stats': {}
    }
    
    for norm in norms:
        print(f"\n{'='*80}")
        print(f"Testing {norm.upper()} norm with epsilon={epsilons[norm]}")
        print(f"{'='*80}")
        
        eps = epsilons[norm]
        
        norm_results = {
            'setting_A': {i: [] for i in auxiliary_configs},
            'setting_B': {i: [] for i in auxiliary_configs},
            'setting_C': {i: [] for i in auxiliary_configs}
        }
        
        # Generation statistics
        generation_stats = {
            'setting_A': {'attempted': 0, 'successful': 0},
            'setting_B': {'attempted': 0, 'successful': 0},
            'setting_C': {'attempted': 0, 'successful': 0}
        }
        
        # Modified: Use community standard alpha values (same as experiment_basic)
        alpha = eps / pgd_alpha_factors[norm]
        
        # Create PGD attack with same parameters as experiment_basic
        pgd_attack = PGDAttack(
            model, 
            eps=eps, 
            alpha=alpha,
            num_iter=pgd_steps,  # 40 iterations
            norm=norm
        )
        
        # Modified: Changed from 3 to 5 experiments
        num_experiments = 5  # Changed from 3 to 5 (same as experiment_basic)
        max_attempts_per_setting = 30
        
        # Setting A: Single image
        print("\n  Testing Setting A...")
        successful_runs_A = 0
        attempt_count_A = 0
        
        while successful_runs_A < num_experiments and attempt_count_A < max_attempts_per_setting:
            attempt_count_A += 1
            generation_stats['setting_A']['attempted'] += 1
            
            # Try different classes to find valid samples
            class_a = list(images_by_class.keys())[attempt_count_A % len(images_by_class)]
            
            # Check if enough images
            required_images = 30 + num_test_images  # 130 total
            if len(images_by_class[class_a]) < required_images:
                continue
            
            x_main = images_by_class[class_a][0][0]
            y_main = torch.tensor([class_a]).to(device)
            
            # Generate valid endpoints
            delta1, delta2, valid = generate_valid_endpoints_setting_A(
                x_main, y_main, pgd_attack, model, max_attempts=50
            )
            
            if not valid:
                print(f"    Failed to generate valid endpoints for class {class_a}")
                continue
            
            generation_stats['setting_A']['successful'] += 1
            successful_runs_A += 1
            
            print(f"    Running Setting A experiment {successful_runs_A}/{num_experiments}...")
            
            # Get fixed test set
            test_images, test_labels = get_fixed_test_set_for_setting(
                images_by_class, 'A', [class_a], num_test_images
            )
            print(f"    Using {len(test_images)} test images")
            
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_A(x_main, y_main, delta1, delta2)
                else:
                    # Additional images from same class
                    additional_images = []
                    additional_labels = []
                    
                    for i in range(1, num_additional + 1):
                        if i < len(images_by_class[class_a]):
                            additional_images.append(images_by_class[class_a][i][0])
                            additional_labels.append(torch.tensor([class_a]).to(device))
                    
                    if len(additional_images) < num_additional:
                        print(f"    Warning: Only {len(additional_images)} auxiliary images available")
                    
                    theta, _, _, _ = bezier.optimize_setting_A_multi(
                        x_main, y_main, delta1, delta2,
                        additional_images, additional_labels
                    )
                
                # Evaluate on fixed test set
                stats = evaluate_transferability_multi(
                    model, bezier, delta1, theta, delta2,
                    test_images, test_labels
                )
                norm_results['setting_A'][num_additional].append(stats)
        
        # Setting B: Same class (similar structure)
        print("\n  Testing Setting B...")
        successful_runs_B = 0
        attempt_count_B = 0
        
        while successful_runs_B < num_experiments and attempt_count_B < max_attempts_per_setting:
            attempt_count_B += 1
            generation_stats['setting_B']['attempted'] += 1
            
            class_b = list(images_by_class.keys())[(attempt_count_B + 1) % len(images_by_class)]
            
            required_images = 30 + num_test_images
            if len(images_by_class[class_b]) < required_images:
                continue
            
            x1 = images_by_class[class_b][0][0]
            x2 = images_by_class[class_b][1][0]
            y = torch.tensor([class_b]).to(device)
            
            delta1, delta2, valid = generate_valid_endpoints_setting_BC(
                x1, x2, y, y, pgd_attack, model, max_attempts=50
            )
            
            if not valid:
                continue
            
            generation_stats['setting_B']['successful'] += 1
            successful_runs_B += 1
            
            print(f"    Running Setting B experiment {successful_runs_B}/{num_experiments}...")
            
            test_images, test_labels = get_fixed_test_set_for_setting(
                images_by_class, 'B', [class_b], num_test_images
            )
            
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_B(x1, x2, y, delta1, delta2)
                else:
                    additional_images = []
                    additional_labels = []
                    
                    for i in range(2, 2 + num_additional):
                        if i < len(images_by_class[class_b]):
                            additional_images.append(images_by_class[class_b][i][0])
                            additional_labels.append(torch.tensor([class_b]).to(device))
                    
                    theta, _, _, _ = bezier.optimize_setting_B_multi(
                        x1, x2, y, delta1, delta2,
                        additional_images, additional_labels
                    )
                
                stats = evaluate_transferability_multi(
                    model, bezier, delta1, theta, delta2,
                    test_images, test_labels
                )
                norm_results['setting_B'][num_additional].append(stats)
        
        # Setting C: Different classes (similar structure)
        print("\n  Testing Setting C...")
        successful_runs_C = 0
        attempt_count_C = 0
        
        while successful_runs_C < num_experiments and attempt_count_C < max_attempts_per_setting:
            attempt_count_C += 1
            generation_stats['setting_C']['attempted'] += 1
            
            class_ids = list(images_by_class.keys())
            class_c1 = class_ids[(attempt_count_C + 2) % len(class_ids)]
            class_c2 = class_ids[(attempt_count_C + 3) % len(class_ids)]
            
            if class_c1 == class_c2:
                continue
            
            required_images = 30 + num_test_images // 2
            if len(images_by_class[class_c1]) < required_images or len(images_by_class[class_c2]) < required_images:
                continue
            
            x1 = images_by_class[class_c1][0][0]
            x2 = images_by_class[class_c2][0][0]
            y1 = torch.tensor([class_c1]).to(device)
            y2 = torch.tensor([class_c2]).to(device)
            
            delta1, delta2, valid = generate_valid_endpoints_setting_BC(
                x1, x2, y1, y2, pgd_attack, model, max_attempts=50
            )
            
            if not valid:
                continue
            
            generation_stats['setting_C']['successful'] += 1
            successful_runs_C += 1
            
            print(f"    Running Setting C experiment {successful_runs_C}/{num_experiments}...")
            
            test_images, test_labels = get_fixed_test_set_for_setting(
                images_by_class, 'C', [class_c1, class_c2], num_test_images
            )
            
            for num_additional in auxiliary_configs:
                bezier = BezierAdversarialMultiImage(model, norm=norm, eps=eps, lr=0.01, num_iter=30)
                
                if num_additional == 0:
                    theta, _, _, _ = bezier.optimize_setting_C(x1, x2, y1, y2, delta1, delta2)
                else:
                    additional_images = []
                    additional_labels = []
                    
                    # Split auxiliary images between classes
                    num_from_class1 = (num_additional + 1) // 2
                    num_from_class2 = num_additional // 2
                    
                    for i in range(1, 1 + num_from_class1):
                        if i < len(images_by_class[class_c1]):
                            additional_images.append(images_by_class[class_c1][i][0])
                            additional_labels.append(torch.tensor([class_c1]).to(device))
                    
                    for i in range(1, 1 + num_from_class2):
                        if i < len(images_by_class[class_c2]):
                            additional_images.append(images_by_class[class_c2][i][0])
                            additional_labels.append(torch.tensor([class_c2]).to(device))
                    
                    theta, _, _, _ = bezier.optimize_setting_C_multi(
                        x1, x2, y1, y2, delta1, delta2,
                        additional_images, additional_labels
                    )
                
                stats = evaluate_transferability_multi(
                    model, bezier, delta1, theta, delta2,
                    test_images, test_labels
                )
                norm_results['setting_C'][num_additional].append(stats)
        
        # Report final statistics
        print(f"\n  {norm.upper()} Summary:")
        print(f"    Setting A: {successful_runs_A}/{num_experiments} successful runs")
        print(f"    Setting B: {successful_runs_B}/{num_experiments} successful runs")
        print(f"    Setting C: {successful_runs_C}/{num_experiments} successful runs")
        
        all_results[norm] = norm_results
        all_results['generation_stats'][norm] = generation_stats
    
    return all_results

if __name__ == "__main__":
    print("Bézier Adversarial Curves - Multi-Image Optimization")
    print("Extended version: Testing with 0,5,10,15,20,25 auxiliary images")
    print("Aligned with experiment_basic.py settings:")
    print("- PGD: 40 iterations with community standard α values")
    print("- 5 experiments per setting for better statistics")
    print("- Bézier optimization: 30 iterations with lr=0.01")
    print("="*80)
    
    results = run_multi_image_experiments_extended()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'bezier_multi_image_extended_{timestamp}.json'
    
    # Save results
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print_multi_image_results_extended(results)
    
    print(f"\nResults saved to {filename}")
    print("\nKey features:")
    print("1. Extended auxiliary images: 0,5,10,15,20,25")
    print("2. Fixed test sets of 100 images")
    print("3. Mean ± std reporting")
    print("4. 50 fixed sampling points for path evaluation")
    print("5. Aligned with experiment_basic.py:")
    print("   • PGD: 40 iterations")
    print("   • Alpha: L∞=ε/4, L₂=ε/5, L₁=ε/10")
    print("   • 5 experiments per setting")