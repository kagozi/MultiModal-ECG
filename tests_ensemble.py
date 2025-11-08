import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, 
    fbeta_score, classification_report
)
from tqdm import tqdm

# Import your models
from models import (CWT2DCNN, DualStreamCNN, ViTFusionECG, 
                    SwinTransformerECG, SwinTransformerEarlyFusion, 
                    ViTLateFusion, EfficientNetLateFusion, 
                    SwinTransformerLateFusion, HybridSwinTransformerECG ,HybridSwinTransformerEarlyFusion, 
                    HybridSwinTransformerLateFusion,
                    EfficientNetEarlyFusion, EfficientNetLateFusion,
                    EfficientNetFusionECG, ResNet50EarlyFusion, 
                    ResNet50LateFusion,
                    ResNet50ECG
)
from benchmark import XResNet1d101, load_ptbxl_dataset, aggregate_diagnostic_labels, preprocess_signals, prepare_labels

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/results/'
BASELINE_RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/results/'
DATA_PATH = '../datasets/ECG/'
ENSEMBLE_PATH = os.path.join(RESULTS_PATH, 'ensemble_results/')
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(ENSEMBLE_PATH, exist_ok=True)

print("="*80)
print("CONFUSION MATRICES & ENSEMBLE ANALYSIS (with ResNet1D Baseline)")
print("="*80)
print(f"Device: {DEVICE}")

# ============================================================================
# DATASET CLASSES
# ============================================================================

class CWTDataset(Dataset):
    """Memory-efficient dataset for CWT data"""
    
    def __init__(self, scalo_path, phaso_path, labels, mode='scalogram'):
        self.scalograms = np.load(scalo_path, mmap_mode='r')
        self.phasograms = np.load(phaso_path, mmap_mode='r')
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
        phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
        label = self.labels[idx]
        
        if self.mode == 'scalogram':
            return scalo, label
        elif self.mode == 'phasogram':
            return phaso, label
        elif self.mode == 'both':
            return (scalo, phaso), label
        elif self.mode == 'fusion':
            fused = torch.cat([scalo, phaso], dim=0)
            return fused, label

class ECGDataset(Dataset):
    """Dataset for raw 1D ECG signals (for ResNet1D baseline)"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (N, time, channels) -> (N, channels, time)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

def load_model_from_config(config, num_classes):
    """Load model architecture based on config"""
    mode = config['mode']
    raw_model_type = config['model']
    model_type = raw_model_type.split('-')[0].split('_')[0]
    adapter_strategy = config.get('adapter', 'learned')
        
    if config['model'] == 'DualStream':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif config['model'] == 'CWT2DCNN':
        num_ch = 24 if mode == 'fusion' else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif config['model'] == 'ViTFusionECG':
        model = ViTFusionECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerECG':
        model = SwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerEarlyFusion':
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'SwinTransformerLateFusion':
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ViTLateFusion':
        model = ViTLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'HybridSwinTransformerECG':
        model = HybridSwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'HybridSwinTransformerEarlyFusion':
        model = HybridSwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'HybridSwinTransformerLateFusion':
        model = HybridSwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'EfficientNetFusionECG':
        model = EfficientNetFusionECG(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'EfficientNetEarlyFusion':
        model = EfficientNetEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'EfficientNetLateFusion':
        model = EfficientNetLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ResNet50EarlyFusion':
        model = ResNet50EarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'ResNet50LateFusion':
        model = ResNet50LateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ResNet50ECG':
        model = ResNet50ECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_type == 'XResNet1d101':
        model = XResNet1d101(
            input_channels=12, 
            num_classes=num_classes,
            stem_ks=config.get('stem_ks', 7),
            block_ks=config.get('block_ks', 3),
            drop_rate=config.get('drop_rate', 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(DEVICE)


def load_model_checkpoint_safely(model, checkpoint_path, device):
    """
    Safely load model checkpoint with adapter key remapping
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        new_state_dict = {}
        model_keys = set(model.state_dict().keys())

        for k, v in state_dict.items():
            new_k = k

            # Fix: adapter.weight → adapter.adapter.weight
            if k == "adapter.weight" and "adapter.adapter.weight" in model_keys:
                new_k = "adapter.adapter.weight"
                print(f"      [Remap] {k} → {new_k}")
            elif k == "adapter.bias" and "adapter.adapter.bias" in model_keys:
                new_k = "adapter.adapter.bias"
                print(f"      [Remap] {k} → {new_k}")
            # Fix: reverse
            elif k == "adapter.adapter.weight" and "adapter.weight" in model_keys:
                new_k = "adapter.weight"
                print(f"      [Remap] {k} → {new_k}")
            elif k == "adapter.adapter.bias" and "adapter.bias" in model_keys:
                new_k = "adapter.bias"
                print(f"      [Remap] {k} → {new_k}")

            new_state_dict[new_k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        if missing:
            print(f"  [Warning] Missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")
        if unexpected:
            print(f"  [Warning] Unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

        # Fail only if non-adapter critical keys are missing
        if any(not m.startswith("adapter") for m in missing):
            print(f"  [Error] Critical missing keys → skipping")
            return False

        print(f"  [Success] Loaded with adapter remapping")
        return True

    except Exception as e:
        print(f"  [Error] Failed to load checkpoint: {e}")
        return False


# ============================================================================
# EVALUATION: ALL MODELS
# ============================================================================

def evaluate_all_models(metadata, X_test_raw, y_test):
    """Evaluate all trained models and generate confusion matrices"""
    
    print("\n[1/3] Loading all trained models...")
    
    all_model_results = {}
    y_true = y_test
    
    # ========================================================================
    # PART 1: Evaluate ResNet1D Baseline
    # ========================================================================
    
    baseline_checkpoint = os.path.join(BASELINE_RESULTS_PATH, 'best_xresnet1d101.pth')
    
    if os.path.exists(baseline_checkpoint):
        print(f"\n{'='*60}")
        print(f"Evaluating: ResNet1D-Baseline")
        print(f"{'='*60}")
        
        test_dataset_raw = ECGDataset(X_test_raw, y_test)
        test_loader_raw = DataLoader(
            test_dataset_raw, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        baseline_config = {
            'model': 'XResNet1d101',
            'mode': 'raw',
            'stem_ks': 7,
            'block_ks': 3,
            'drop_rate': 0.2
        }
        
        model = XResNet1d101(
            input_channels=12,
            num_classes=metadata['num_classes'],
            stem_ks=7,
            block_ks=3,
            drop_rate=0.2
        ).to(DEVICE)
        
        if load_model_checkpoint_safely(model, baseline_checkpoint, DEVICE):
            model.eval()
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for x, y in tqdm(test_loader_raw, desc="Predicting", leave=False):
                    x = x.to(DEVICE)
                    outputs = model(x)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.append(probs)
                    all_labels.append(y.numpy())
            
            y_scores = np.vstack(all_preds)
            y_true = np.vstack(all_labels)
            optimal_thresholds = np.ones(metadata['num_classes']) * 0.5
            y_pred = apply_thresholds(y_scores, optimal_thresholds)
            metrics = compute_all_metrics(y_true, y_pred, y_scores)
            
            print(f"\nMetrics:")
            print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
            print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
            
            plot_combined_confusion(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, "confusion_combined_ResNet1D-Baseline.png"),
                title="Confusion Matrix"
            )
            plot_confusion_matrix_multiclass(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, "confusion_multiclass_ResNet1D-Baseline.png"),
                title="Multi-Class Confusion Matrix"
            )
            
            all_model_results['ResNet1D-Baseline'] = {
                'metrics': metrics,
                'y_scores': y_scores,
                'y_pred': y_pred,
                'thresholds': optimal_thresholds,
                'config': baseline_config
            }
            print(f"Success: ResNet1D-Baseline evaluated")
        else:
            print(f"Failed: ResNet1D-Baseline")
    else:
        print(f"\nWarning: ResNet1D checkpoint not found: {baseline_checkpoint}")
    
    # ========================================================================
    # PART 2: Evaluate CWT-based Models
    # ========================================================================
    
    result_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('results_') and f.endswith('.json')]
    
    if not result_files:
        print("Warning: No CWT models found!")
    else:
        print(f"\nFound {len(result_files)} CWT models")
    
    for result_file in result_files:
        model_name = result_file.replace('results_', '').replace('.json', '')
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        with open(os.path.join(RESULTS_PATH, result_file), 'r') as f:
            results = json.load(f)
        
        config = results['config']
        optimal_thresholds = np.array(results['optimal_thresholds'])
        checkpoint_path = os.path.join(RESULTS_PATH, f"best_{model_name}.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"  Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model = load_model_from_config(config, metadata['num_classes'])
            
            if not load_model_checkpoint_safely(model, checkpoint_path, DEVICE):
                print(f"  Failed: Skipping {model_name}")
                continue
            
            model.eval()
            dataset_mode = config['mode']
            test_dataset = CWTDataset(
                os.path.join(WAVELETS_PATH, 'test_scalograms.npy'),
                os.path.join(WAVELETS_PATH, 'test_phasograms.npy'),
                y_test,
                mode=dataset_mode
            )
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            
            y_scores, y_true = get_predictions(model, test_loader, config)
            y_pred = apply_thresholds(y_scores, optimal_thresholds)
            metrics = compute_all_metrics(y_true, y_pred, y_scores)
            
            print(f"\nMetrics:")
            print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
            print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
            
            plot_confusion_matrix_multiclass(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{model_name}.png"),
                title=f"Multi-Class Confusion Matrix"
            )
            
            all_model_results[model_name] = {
                'metrics': metrics,
                'y_scores': y_scores,
                'y_pred': y_pred,
                'thresholds': optimal_thresholds,
                'config': config
            }
            print(f"Success: {model_name} evaluated")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return all_model_results, y_true


# ============================================================================
# PREDICTION & METRICS
# ============================================================================

@torch.no_grad()
def get_predictions(model, dataloader, config):
    model.eval()
    all_preds, all_labels = [], []
    mode = config['mode']
    is_dual = (config['model'] == 'DualStream') or (mode == 'both')
    
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        if isinstance(batch[0], (tuple, list)):
            (x1, x2), y = batch
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            if is_dual:
                outputs = model(x1, x2)
            else:
                if mode == 'scalogram':
                    outputs = model(x1)
                elif mode == 'phasogram':
                    outputs = model(x2)
                elif mode == 'fusion':
                    x_fused = torch.cat([x1, x2], dim=1)
                    outputs = model(x_fused)
                else:
                    outputs = model(x1)
        else:
            x, y = batch
            x = x.to(DEVICE)
            outputs = model(x)
        
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)

def apply_thresholds(y_scores, thresholds):
    y_pred = (y_scores > thresholds).astype(int)
    for i, pred in enumerate(y_pred):
        if pred.sum() == 0:
            y_pred[i, np.argmax(y_scores[i])] = 1
    return y_pred

def compute_all_metrics(y_true, y_pred, y_scores):
    metrics = {
        'macro_auc': roc_auc_score(y_true, y_scores, average='macro'),
        'micro_auc': roc_auc_score(y_true, y_scores, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f_beta_macro': fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0),
        'per_class_auc': []
    }
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        except:
            auc = 0.0
        metrics['per_class_auc'].append(auc)
    return metrics


# ============================================================================
# PLOTTING
# ============================================================================

def plot_confusion_matrix_multiclass(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

def plot_combined_confusion(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    fig = plt.figure(figsize=(16, 6))
    n_classes = len(class_names)
    for i, class_name in enumerate(class_names):
        ax = plt.subplot(2, n_classes, i+1)
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                   cbar=False)
        ax.set_title(f'{class_name}', fontsize=10)
        if i == 0:
            ax.set_ylabel('True', fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
    
    ax_multi = plt.subplot(2, 1, 2)
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    cm_multi = confusion_matrix(y_true_single, y_pred_single, labels=range(n_classes))
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', ax=ax_multi,
                xticklabels=class_names, yticklabels=class_names)
    ax_multi.set_title('Multi-Class View (Argmax)', fontsize=12, fontweight='bold')
    ax_multi.set_xlabel('Predicted', fontsize=10)
    ax_multi.set_ylabel('True', fontsize=10)
    plt.setp(ax_multi.get_xticklabels(), rotation=45, ha='right')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# ENSEMBLE & COMPARISON
# ============================================================================

def create_ensemble(model_results, y_true, metadata, method='average', top_k=None):
    print(f"\n[2/3] Creating ensemble (method={method}, top_k={top_k})...")
    if top_k:
        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['metrics']['macro_auc'], reverse=True)[:top_k]
        selected_names = [n for n, _ in sorted_models]
        print(f"Top {top_k} models: {[n for n in selected_names]}")
    else:
        selected_names = list(model_results.keys())
        print(f"Using all {len(selected_names)} models")
    
    all_scores = [model_results[n]['y_scores'] for n in selected_names]
    all_thresholds = [model_results[n]['thresholds'] for n in selected_names]
    
    if method == 'average':
        ensemble_scores = np.mean(all_scores, axis=0)
        ensemble_thresholds = np.mean(all_thresholds, axis=0)
    elif method == 'weighted':
        weights = np.array([model_results[n]['metrics']['macro_auc'] for n in selected_names])
        weights /= weights.sum()
        ensemble_scores = np.average(all_scores, axis=0, weights=weights)
        ensemble_thresholds = np.average(all_thresholds, axis=0, weights=weights)
    elif method == 'max':
        ensemble_scores = np.max(all_scores, axis=0)
        ensemble_thresholds = np.mean(all_thresholds, axis=0)
    
    ensemble_pred = apply_thresholds(ensemble_scores, ensemble_thresholds)
    metrics = compute_all_metrics(y_true, ensemble_pred, ensemble_scores)
    
    print(f"Ensemble AUC: {metrics['macro_auc']:.4f} | F1: {metrics['f1_macro']:.4f}")
    
    ensemble_name = f"ensemble_{method}" + (f"_top{top_k}" if top_k else "_all")
    plot_combined_confusion(y_true, ensemble_pred, metadata['classes'],
                            save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{ensemble_name}.png"))
    plot_confusion_matrix_multiclass(y_true, ensemble_pred, metadata['classes'],
                                     save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{ensemble_name}.png"))
    return metrics, ensemble_name

def plot_model_comparison(model_results, ensemble_results, metadata):
    print("\n[3/3] Creating comparison plots...")
    names = list(model_results.keys()) + [n for n, _ in ensemble_results]
    aucs = [model_results[n]['metrics']['macro_auc'] for n in model_results] + [m['macro_auc'] for _, m in ensemble_results]
    f1s = [model_results[n]['metrics']['f1_macro'] for n in model_results] + [m['f1_macro'] for _, m in ensemble_results]
    
    sorted_idx = np.argsort(aucs)[::-1]
    names = [names[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]
    f1s = [f1s[i] for i in sorted_idx]
    colors = ['#1f77b4'] * len(model_results) + ['#ff7f0e'] * len(ensemble_results)
    colors = [colors[i] for i in sorted_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(range(len(names)), aucs, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Macro AUC')
    ax1.set_title('Model AUC')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(aucs):
        ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
    ax2.barh(range(len(names)), f1s, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Macro F1')
    ax2.set_title('Model F1')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(f1s):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENSEMBLE_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: model_comparison.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n[Step 1] Loading data...")
    X, Y = load_ptbxl_dataset(DATA_PATH, PROCESSED_PATH, sampling_rate=100)
    Y = aggregate_diagnostic_labels(Y, DATA_PATH + 'scp_statements.csv')
    X, Y, y, mlb = prepare_labels(X, Y, min_samples=0)
    
    X_test = X[Y.strat_fold == 10]
    y_test = y[Y.strat_fold == 10]
    
    X_train = X[Y.strat_fold <= 8]
    X_val = X[Y.strat_fold == 9]
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test)
    
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    model_results, y_true = evaluate_all_models(metadata, X_test, y_test)
    
    if not model_results:
        print("\nError: No models evaluated!")
        return
    
    print(f"\nSuccess: Evaluated {len(model_results)} models")
    
    ensemble_results = []
    metrics, name = create_ensemble(model_results, y_true, metadata, method='average')
    ensemble_results.append((name, metrics))
    metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted')
    ensemble_results.append((name, metrics))
    
    if len(model_results) >= 3:
        metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=3)
        ensemble_results.append((name, metrics))
        metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=3)
        ensemble_results.append((name, metrics))
    
    if len(model_results) >= 5:
        metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=5)
        ensemble_results.append((name, metrics))
        metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=5)
        ensemble_results.append((name, metrics))
    
    plot_model_comparison(model_results, ensemble_results, metadata)
    
    summary = {
        'individual_models': {n: {k: v for k, v in r['metrics'].items() if k in ['macro_auc', 'f1_macro', 'f_beta_macro']}
                              for n, r in model_results.items()},
        'ensembles': {n: {k: v for k, v in m.items() if k in ['macro_auc', 'f1_macro', 'f_beta_macro']}
                      for n, m in ensemble_results}
    }
    with open(os.path.join(ENSEMBLE_PATH, 'complete_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Model':<40} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 80)
    for name, res in sorted(model_results.items(), key=lambda x: x[1]['metrics']['macro_auc'], reverse=True):
        m = res['metrics']
        print(f"{name:<40} | {m['macro_auc']:.4f}   | {m['f1_macro']:.4f}   | {m['f_beta_macro']:.4f}")
    print("-" * 80)
    for name, m in ensemble_results:
        print(f"{name:<40} | {m['macro_auc']:.4f}   | {m['f1_macro']:.4f}   | {m['f_beta_macro']:.4f}")
    
    best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['macro_auc'])
    best_ens = max(ensemble_results, key=lambda x: x[1]['macro_auc'])
    print(f"\nBest Model: {best_model[0]} (AUC: {best_model[1]['metrics']['macro_auc']:.4f})")
    print(f"Best Ensemble: {best_ens[0]} (AUC: {best_ens[1]['macro_auc']:.4f})")
    print("\nSuccess: Analysis complete! Results in:", ENSEMBLE_PATH)

if __name__ == '__main__':
    main()