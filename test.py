"""
Model Evaluation Script for SignSpeak ISL Recognition

This script evaluates the trained Keras model and generates:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Per-class metrics

Usage:
    python test.py
    python test.py --model path/to/model.h5
    python test.py --dataset path/to/Indian
"""

import os
import sys
import argparse
import warnings
import time
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

# -----------------------------
#   CONFIGURATION
# -----------------------------
DEFAULT_MODEL_PATHS = [
    './model/indian_sign_model.h5',
    './checkpoints/best_model_20251228_173824.h5',
    r'D:\sign-to-text-and-speech\model\indian_sign_model.h5'
]

DEFAULT_DATASET_PATHS = [
    './Indian',
    r'D:\sign-to-text-and-speech\Indian',
    '../Indian'
]

# ISL Alphabet (35 classes: 1-9 + A-Z)
ISL_ALPHABET = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# -----------------------------
#   MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands

# Must match train_indian_model.py (USE_PREPROCESSED_LANDMARKS=True) and app.py Keras path.
SUPPORT_TWO_HANDS = True


def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to pixel coordinates (same as training / app)."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    """Wrist-relative + scale by max abs value — same as train_indian_model.pre_process_landmark."""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list)) or 1.0
    return [n / max_value for n in temp_landmark_list]


def combine_two_hands(per_hand_list):
    if not per_hand_list:
        return None
    if SUPPORT_TWO_HANDS:
        if len(per_hand_list) == 1:
            return per_hand_list[0] + [0.0] * len(per_hand_list[0])
        return per_hand_list[0] + per_hand_list[1]
    return per_hand_list[0]


def extract_keras_landmarks(image, hands_detector):
    """
    Extract the same feature vector used during training (preprocessed landmarks, 84-dim if two-hand mode).
    Previously test.py used bbox-normalized coords, which does not match the trained model.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    pre_list = []
    for hand_landmarks in results.multi_hand_landmarks:
        landmark_list = calc_landmark_list(image, hand_landmarks)
        pre_list.append(pre_process_landmark(landmark_list))
    features = combine_two_hands(pre_list)
    return features

def load_dataset(dataset_path, max_samples_per_class=None):
    """Load dataset and extract features."""
    print(f"\n📂 Loading dataset from: {dataset_path}")
    
    X, y = [], []
    class_counts = defaultdict(int)
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    
    total_images = 0
    processed = 0
    skipped = 0
    
    # Count total images first
    for class_name in ISL_ALPHABET:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(files) if max_samples_per_class is None else min(len(files), max_samples_per_class)
    
    print(f"   Total images to process: {total_images}")
    
    for class_idx, class_name in enumerate(ISL_ALPHABET):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):
            print(f"   ⚠️  Class folder not found: {class_name}")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_samples_per_class:
            image_files = image_files[:max_samples_per_class]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                skipped += 1
                continue
            
            features = extract_keras_landmarks(image, hands)
            
            if features is not None:
                X.append(features)
                y.append(class_idx)
                class_counts[class_name] += 1
                processed += 1
            else:
                skipped += 1
            
            # Progress update
            if (processed + skipped) % 500 == 0:
                print(f"   Progress: {processed + skipped}/{total_images} ({100*(processed+skipped)/total_images:.1f}%)")
    
    hands.close()
    
    print(f"\n✅ Dataset loaded:")
    print(f"   Processed: {processed} images")
    print(f"   Skipped (no hand detected): {skipped} images")
    print(f"   Classes with data: {len(class_counts)}/{len(ISL_ALPHABET)}")
    
    return np.array(X), np.array(y), class_counts

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model and return all metrics."""
    print("\n🔍 Evaluating model...")

    # Warm-up pass for more stable latency measurement
    _ = model.predict(X_test[: min(16, len(X_test))], verbose=0)

    # Timed inference pass
    t0 = time.perf_counter()
    y_pred_proba = model.predict(X_test, verbose=0)
    t1 = time.perf_counter()

    y_pred = np.argmax(y_pred_proba, axis=1)

    total_latency_sec = t1 - t0
    avg_latency_ms = (total_latency_sec / max(len(X_test), 1)) * 1000.0
    throughput_fps = len(X_test) / max(total_latency_sec, 1e-9)

    # Calculate metrics
    metrics = {
        'top_1_accuracy': accuracy_score(y_test, y_pred),  # Top-1 equals standard accuracy
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'latency_total_sec': total_latency_sec,
        'latency_per_sample_ms': avg_latency_ms,
        'throughput_samples_per_sec': throughput_fps,
    }

    # Top-k accuracy (if more than 1 class)
    if len(np.unique(y_test)) > 1:
        try:
            metrics['top_3_accuracy'] = top_k_accuracy_score(y_test, y_pred_proba, k=3)
            metrics['top_5_accuracy'] = top_k_accuracy_score(y_test, y_pred_proba, k=5)
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    
    return metrics, cm, report, y_pred, y_pred_proba

def print_metrics(metrics):
    """Print metrics in a formatted table."""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*45)
    print(f"{'Accuracy':<30} {metrics['accuracy']*100:>14.2f}%")
    print(f"{'Top-1 Accuracy':<30} {metrics['top_1_accuracy']*100:>14.2f}%")
    print(f"{'Precision (Macro)':<30} {metrics['precision_macro']*100:>14.2f}%")
    print(f"{'Precision (Weighted)':<30} {metrics['precision_weighted']*100:>14.2f}%")
    print(f"{'Recall (Macro)':<30} {metrics['recall_macro']*100:>14.2f}%")
    print(f"{'Recall (Weighted)':<30} {metrics['recall_weighted']*100:>14.2f}%")
    print(f"{'F1-Score (Macro)':<30} {metrics['f1_macro']*100:>14.2f}%")
    print(f"{'F1-Score (Weighted)':<30} {metrics['f1_weighted']*100:>14.2f}%")
    
    if 'top_3_accuracy' in metrics:
        print(f"{'Top-3 Accuracy':<30} {metrics['top_3_accuracy']*100:>14.2f}%")
    if 'top_5_accuracy' in metrics:
        print(f"{'Top-5 Accuracy':<30} {metrics['top_5_accuracy']*100:>14.2f}%")
    print(f"{'Total Inference Time':<30} {metrics['latency_total_sec']:>14.4f}s")
    print(f"{'Latency per Sample':<30} {metrics['latency_per_sample_ms']:>14.3f} ms")
    print(f"{'Throughput':<30} {metrics['throughput_samples_per_sec']:>14.2f} samples/s")
    
    print("-"*45)


def print_topk_explanation():
    """Explain how Top-1 and Top-3 accuracy are calculated."""
    print("\n" + "="*60)
    print("🧮 TOP-1 / TOP-3 CALCULATION")
    print("="*60)
    print("Top-1 Accuracy:")
    print("  - For each sample, take the class with highest probability (argmax).")
    print("  - If that class equals the true label, it counts as correct.")
    print("  - Formula: correct_top1 / total_samples")
    print("")
    print("Top-3 Accuracy:")
    print("  - For each sample, take the top 3 highest-probability classes.")
    print("  - If true label appears anywhere in those 3 classes, it counts as correct.")
    print("  - Formula: correct_top3 / total_samples")


def print_loss_explanation(model):
    """Print loss function explanation."""
    print("\n" + "="*60)
    print("📉 LOSS FUNCTION EXPLANATION")
    print("="*60)

    # If model was loaded compiled, this is often available.
    model_loss = getattr(model, "loss", None)
    if model_loss:
        print(f"Model loss configured in Keras: {model_loss}")
    else:
        print("Model loss not exposed (model may be loaded without compile metadata).")

    print("")
    print("For this multiclass ISL task, training typically uses Categorical Crossentropy:")
    print("  Loss = -sum(y_true * log(y_pred)) across classes")
    print("  - y_true: one-hot encoded true class")
    print("  - y_pred: predicted class probabilities (softmax output)")
    print("Lower loss means predicted probabilities are closer to true labels.")


def print_latency_explanation(metrics):
    """Explain how latency values are calculated."""
    print("\n" + "="*60)
    print("⏱️ LATENCY CALCULATION")
    print("="*60)
    print("Measured by timing model.predict(X_test) with time.perf_counter().")
    print(f"Total inference time: {metrics['latency_total_sec']:.4f} seconds")
    print(f"Per-sample latency: {metrics['latency_per_sample_ms']:.3f} ms")
    print(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/second")
    print("")
    print("Formula:")
    print("  per_sample_latency_ms = (total_time_sec / num_samples) * 1000")

def print_classification_report(report, class_names):
    """Print per-class metrics."""
    print("\n" + "="*60)
    print("📋 PER-CLASS METRICS")
    print("="*60)
    
    print(f"\n{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*56)
    
    for class_name in class_names:
        if class_name in report:
            r = report[class_name]
            print(f"{class_name:<10} {r['precision']*100:>11.2f}% {r['recall']*100:>11.2f}% {r['f1-score']*100:>11.2f}% {r['support']:>10.0f}")
    
    print("-"*56)
    
    # Macro/Weighted averages
    if 'macro avg' in report:
        r = report['macro avg']
        print(f"{'Macro Avg':<10} {r['precision']*100:>11.2f}% {r['recall']*100:>11.2f}% {r['f1-score']*100:>11.2f}% {r['support']:>10.0f}")
    
    if 'weighted avg' in report:
        r = report['weighted avg']
        print(f"{'Wt. Avg':<10} {r['precision']*100:>11.2f}% {r['recall']*100:>11.2f}% {r['f1-score']*100:>11.2f}% {r['support']:>10.0f}")

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(14, 12))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Accuracy'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Normalized)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n💾 Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_metrics_bar(report, class_names, save_path=None):
    """Plot per-class metrics as bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    precisions = [report[c]['precision'] if c in report else 0 for c in class_names]
    recalls = [report[c]['recall'] if c in report else 0 for c in class_names]
    f1_scores = [report[c]['f1-score'] if c in report else 0 for c in class_names]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names)))
    
    axes[0].bar(class_names, precisions, color=colors)
    axes[0].set_title('Precision by Class')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(class_names, recalls, color=colors)
    axes[1].set_title('Recall by Class')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis='x', rotation=45)
    
    axes[2].bar(class_names, f1_scores, color=colors)
    axes[2].set_title('F1-Score by Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_ylim(0, 1.1)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"💾 Metrics chart saved to: {save_path}")
    
    plt.show()

def save_results(metrics, report, output_dir='./evaluation_results'):
    """Save evaluation results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'timestamp': timestamp,
        'metrics': metrics,
        'classification_report': report
    }
    
    filepath = os.path.join(output_dir, f'evaluation_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Evaluate SignSpeak ISL Model')
    parser.add_argument('--model', type=str, help='Path to Keras model (.h5)')
    parser.add_argument('--dataset', type=str, help='Path to Indian Sign Language dataset')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per class (for quick testing)')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting (for headless environments)')
    parser.add_argument('--save', action='store_true', help='Save results to JSON')
    args = parser.parse_args()
    
    print("="*60)
    print("🤙 SignSpeak ISL Model Evaluation")
    print("="*60)
    
    # Find model
    model_path = args.model
    if not model_path:
        for path in DEFAULT_MODEL_PATHS:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path or not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        print("   Tried paths:", DEFAULT_MODEL_PATHS)
        print("   Use --model to specify path")
        sys.exit(1)
    
    print(f"\n📦 Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"   Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output classes: {model.output_shape[-1]}")
    
    # Find dataset
    dataset_path = args.dataset
    if not dataset_path:
        for path in DEFAULT_DATASET_PATHS:
            if os.path.isdir(path):
                dataset_path = path
                break
    
    if not dataset_path or not os.path.isdir(dataset_path):
        print("❌ Error: Dataset folder not found!")
        print("   Tried paths:", DEFAULT_DATASET_PATHS)
        print("   Use --dataset to specify path")
        sys.exit(1)
    
    # Load dataset
    X, y, class_counts = load_dataset(dataset_path, args.max_samples)
    
    if len(X) == 0:
        print("❌ Error: No valid samples found in dataset!")
        sys.exit(1)
    
    # Get class names that have data
    classes_with_data = [ISL_ALPHABET[i] for i in sorted(set(y))]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Evaluate
    metrics, cm, report, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, classes_with_data)
    
    # Print results
    print_metrics(metrics)
    print_classification_report(report, classes_with_data)
    print_topk_explanation()
    print_loss_explanation(model)
    print_latency_explanation(metrics)
    
    # Save results
    if args.save:
        save_results(metrics, report)
    
    # Plot
    if not args.no_plot:
        print("\n📈 Generating visualizations...")
        
        # Create output directory
        os.makedirs('./evaluation_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        plot_confusion_matrix(
            cm, classes_with_data,
            save_path=f'./evaluation_results/confusion_matrix_{timestamp}.png'
        )
        
        plot_metrics_bar(
            report, classes_with_data,
            save_path=f'./evaluation_results/metrics_chart_{timestamp}.png'
        )
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    
    return metrics

if __name__ == '__main__':
    main()