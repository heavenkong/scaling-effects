# ==================================================================
# This code was wrriten by Gukhua Lee (KISTI, ghlee@kisti.re.kr) 
# ==================================================================

import os, json, time, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.parallel import DataParallel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score
)

# =============================
# Reproducibility / backend
# =============================
SEED = 69
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # speedup for fixed shapes

# DataLoader tuning
NUM_WORKERS = min(8, os.cpu_count() or 1)
PIN_MEMORY  = True
PERSISTENT  = True

# =============================
# Paths / IO
# =============================
OUT_DIR = "power_comp_eval"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = "/.../data/20200101_20241231.csv"

# =============================
# Data loading & filtering
# =============================
# Columns: ['DATE','JOBID','GID','UID','JOBNAME','QNAME','WAIT','RUN','CPUS','CPU_T',
#           'MEMKB','VMEM','STATUS','E_CPUS','E_RUN','EXIT_CODE','SRUTime','APPL',
#           'NODESU','MODE','WallTime','isBB']
df = pd.read_csv(
    DATA_PATH,
    header=None,
    delimiter=",",
    skipinitialspace=False,
    names=[
        'DATE','JOBID','GID','UID','JOBNAME','QNAME','WAIT','RUN','CPUS','CPU_T',
        'MEMKB','VMEM','STATUS','E_CPUS','E_RUN','EXIT_CODE','SRUTime','APPL',
        'NODESU','MODE','WallTime','isBB'
    ],
)

# drop rows with APPL == 0 (not part of the target taxonomy)
df_1 = df[df["APPL"] != 0].reset_index(drop=True)
NUM_CLASSES = 34  # we assume labels span 0..33 (except 0 filtered above)

# =============================
# Utilities
# =============================
class ClassifierDataset(Dataset):
    """Simple (X, y) dataset. Expects torch tensors."""
    def __init__(self, X_data: torch.Tensor, y_data: torch.Tensor):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]
    def __len__(self):
        return self.X_data.size(0)

class MulticlassClassification(nn.Module):
    """
    Deep MLP classifier for tabular inputs:
      D -> 8192 -> 4096 -> 2048 -> 1024 -> C
    BatchNorm + ReLU for each block, Dropout in hidden layers.
    """
    def __init__(self, num_feature: int, num_class: int):
        super().__init__()
        self.layer_1 = nn.Linear(num_feature, 8192)
        self.layer_2 = nn.Linear(8192, 4096)
        self.layer_3 = nn.Linear(4096, 2048)
        self.layer_4 = nn.Linear(2048, 1024)
        self.layer_out = nn.Linear(1024, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(8192)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.relu(self.bn1(self.layer_1(x)))
        x = self.dropout(self.relu(self.bn2(self.layer_2(x))))
        x = self.dropout(self.relu(self.bn3(self.layer_3(x))))
        x = self.dropout(self.relu(self.bn4(self.layer_4(x))))
        x = self.layer_out(x)
        return x

def _make_loader(dataset, batch_size, sampler=None, shuffle=False):
    """Dataloader with tuned params for speed."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT and NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        drop_last=False,
    )

def _class_weights_from_labels(y_int: np.ndarray, num_classes: int) -> torch.Tensor:
    """Vectorized class weight = 1/freq (length = num_classes)."""
    counts = np.bincount(y_int, minlength=num_classes)
    counts[counts == 0] = 1  # avoid division by zero
    w = 1.0 / counts
    return torch.tensor(w, dtype=torch.float32)

def plot_cm(cm_mat, title, fname, labels, normalize=False, tick_step=2):
    """Save confusion matrix (counts or normalized)."""
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_mat, annot=False, fmt=".2f" if normalize else "d")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(0, len(labels), tick_step))
    ax.set_yticks(np.arange(0, len(labels), tick_step))
    ax.set_xticklabels(np.arange(0, len(labels), tick_step))
    ax.set_yticklabels(np.arange(0, len(labels), tick_step))
    plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()

def extract_key_metrics(y_true, y_pred, report_text=None):
    """Compact set of metrics parsed from classification_report."""
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc = rep_dict.get('accuracy', np.nan)
    macro_f1 = rep_dict.get('macro avg', {}).get('f1-score', np.nan)
    weighted_f1 = rep_dict.get('weighted avg', {}).get('f1-score', np.nan)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "report_text": report_text}

# -------- Calibration / OOD helpers --------
def collect_logits_and_labels(model, loader, device):
    """Run model on loader and collect logits + labels (CPU tensors -> numpy)."""
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            logits_list.append(logits.cpu())
            y_list.append(yb.cpu())
    logits = torch.cat(logits_list, dim=0)
    y_true = torch.cat(y_list, dim=0).numpy()
    return logits, y_true

def softmax_np(z):
    """Stable softmax (numpy)."""
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def expected_calibration_error(y_true, probs, n_bins=15):
    """ECE with equal-width confidence bins in [0,1]. Lower is better."""
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece, N = 0.0, len(y_true)
    for i in range(n_bins):
        m = (confidences >= bins[i]) & (confidences < bins[i+1])
        if m.any():
            acc = np.mean(preds[m] == y_true[m])
            conf = np.mean(confidences[m])
            ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)

def brier_score_multiclass(y_true, probs):
    """Multiclass Brier = MSE(one-hot, probs). Lower is better."""
    C = probs.shape[1]
    oh = np.eye(C)[y_true]
    return float(np.mean(np.sum((probs - oh)**2, axis=1)))

def plot_reliability(probs, y_true, fname, n_bins=15):
    """Reliability diagram (confidence vs accuracy)."""
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    bins = np.linspace(0,1,n_bins+1)
    accs, confs = [], []
    for i in range(n_bins):
        m = (conf>=bins[i]) & (conf<bins[i+1])
        if m.any():
            accs.append((pred[m]==y_true[m]).mean())
            confs.append(conf[m].mean())
        else:
            accs.append(np.nan); confs.append(np.nan)
    accs, confs = np.array(accs), np.array(confs)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'--')
    plt.scatter(confs, accs)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0,1); plt.ylim(0,1); plt.tight_layout()
    plt.savefig(fname, dpi=200); plt.close()

def energy_scores_from_logits(logits_np):
    """Energy: E(x) = -logsumexp(logits). Lower ~ in-distribution."""
    m = logits_np.max(axis=1, keepdims=True)
    e = - (m + np.log(np.exp(logits_np - m).sum(axis=1, keepdims=True)))
    return e.ravel()

def fpr_at_95_tpr(scores_id, scores_ood):
    """FPR@95%TPR helper for OOD curves."""
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])
    fpr, tpr, th = roc_curve(y, s)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])

def tune_temperature_on_val(model, val_loader, device, max_iter=300, lr=0.1):
    """
    Learn a single temperature T by minimizing NLL on val logits.
    Returns float T.
    """
    model.eval()
    logits_t, y_val_np = collect_logits_and_labels(model, val_loader, device)
    logits_t = logits_t.to(device=device, dtype=torch.float32)
    y_val_t  = torch.from_numpy(y_val_np).to(device)

    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def _closure():
        opt.zero_grad(set_to_none=True)
        loss = nll(logits_t / T.clamp_min(1e-3), y_val_t)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(T.detach().item())

# =============================
# Training / Evaluation driver
# =============================
def run_experiment(
    feature_idx_list,
    tag="A",
    epochs=100,
    lr=1e-5,
    train_batch=4096,
    val_batch=8192,
    test_batch=8192,
):
    """
    Train/validate/test for a given feature set.
    Includes:
      - scaling (fit on train)
      - class-weighted CE + WeightedRandomSampler
      - AMP mixed precision
      - calibration (temperature scaling) with ECE/Brier
      - confusion matrices, learning curves
    Outputs saved under OUT_DIR.
    """
    print(f"\n=== Experiment {tag} | Features: {feature_idx_list} ===")

    # Slice features and target as numpy (float32 / int64)
    X = df_1.iloc[:, feature_idx_list].to_numpy(dtype=np.float32, copy=False)
    y = df_1.iloc[:, 17].to_numpy(dtype=np.int64, copy=False)

    # Stratified split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.46, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.40, stratify=y_trainval, random_state=SEED+43
    )

    # Scale (fit only on train → avoid leakage)
    sk_scaler = PowerTransformer()
    X_train = sk_scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val   = sk_scaler.transform(X_val).astype(np.float32, copy=False)
    X_test  = sk_scaler.transform(X_test).astype(np.float32, copy=False)

    # Torch tensors
    X_train_t = torch.from_numpy(X_train)
    X_val_t   = torch.from_numpy(X_val)
    X_test_t  = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train)
    y_val_t   = torch.from_numpy(y_val)
    y_test_t  = torch.from_numpy(y_test)

    # Datasets
    train_dataset = ClassifierDataset(X_train_t, y_train_t)
    val_dataset   = ClassifierDataset(X_val_t,   y_val_t)
    test_dataset  = ClassifierDataset(X_test_t,  y_test_t)

    # Class weights + sampler
    class_weights = _class_weights_from_labels(y_train, NUM_CLASSES)  # (C,)
    sample_weights = class_weights[y_train]                            # (N_train,)
    weighted_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Loaders
    train_loader = _make_loader(train_dataset, train_batch, sampler=weighted_sampler)
    val_loader   = _make_loader(val_dataset,   val_batch,   shuffle=False)
    test_loader  = _make_loader(test_dataset,  test_batch,  shuffle=False)

    # Model / Optim / AMP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature=X_train.shape[1], num_class=NUM_CLASSES)
    model = DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("LEARNING_RATE:", lr)
    print(model)

    train_acc, train_losses = [], []
    eval_accu, eval_losses  = [], []

    def pass_epoch(loader, train_mode=True):
        if train_mode: model.train()
        else:          model.eval()

        running_loss, correct, total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                    logits = model(Xb)
                    loss   = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                        logits = model(Xb)
                        loss   = criterion(logits, yb)

            running_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        return running_loss/len(loader), 100.0 * correct / total

    # Train
    best_va = -1.0
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = pass_epoch(train_loader, train_mode=True)
        va_loss, va_acc = pass_epoch(val_loader,   train_mode=False)

        train_losses.append(tr_loss); train_acc.append(tr_acc)
        eval_losses.append(va_loss);  eval_accu.append(va_acc)

        if va_acc > best_va:
            best_va = va_acc
            torch.save({"model": model.state_dict()}, os.path.join(OUT_DIR, f"best_{tag}.pt"))

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"[{tag}] Ep {ep:03d} | Tr {tr_loss:.3f}/{tr_acc:.2f}% | Va {va_loss:.3f}/{va_acc:.2f}%")

    # Test
    y_pred_list, y_true_list = [], []
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(Xb)
            y_pred_list.append(logits.argmax(1).cpu().numpy())
            y_true_list.append(yb.cpu().numpy())
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)

    # Report
    rep = classification_report(y_true, y_pred, digits=2, zero_division=0)
    print(f"\n[{tag}] Classification Report\n{rep}")
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as f:
        f.write(rep)

    # Confusion matrices (counts & row-normalized)
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_row = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plot_cm(cm,     f"Confusion Matrix (Counts) [{tag}]",   os.path.join(OUT_DIR, f"cm_counts_{tag}.png"), labels, normalize=False, tick_step=2)
    plot_cm(cm_row, f"Confusion Matrix (Row-Norm) [{tag}]", os.path.join(OUT_DIR, f"cm_row_{tag}.png"),    labels, normalize=True,  tick_step=2)

    # Curves
    plt.figure(); plt.plot(train_acc,'-.'); plt.plot(eval_accu,'-.')
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(["Train","Valid"])
    plt.title(f"Train vs Valid Accuracy [{tag}]"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"accuracy_{tag}.png"), dpi=300); plt.close()

    plt.figure(); plt.plot(train_losses,'-.'); plt.plot(eval_losses,'-.')
    plt.xlabel("epoch"); plt.ylabel("losses"); plt.legend(["Train","Valid"])
    plt.title(f"Train vs Valid Losses [{tag}]"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"losses_{tag}.png"), dpi=300); plt.close()

    # Calibration: learn T on validation, evaluate ECE/Brier on test
    T = tune_temperature_on_val(model, val_loader, device)
    print(f"[Calibration] Learned Temperature T = {T:.4f}")

    logits_test_t, y_test_flat = collect_logits_and_labels(model, test_loader, device)
    logits_test_np = logits_test_t.numpy()
    probs_test_raw = softmax_np(logits_test_np)
    probs_test_cal = softmax_np(logits_test_np / max(T, 1e-3))

    ece_raw = expected_calibration_error(y_test_flat, probs_test_raw)
    ece_cal = expected_calibration_error(y_test_flat, probs_test_cal)
    brier_raw = brier_score_multiclass(y_test_flat, probs_test_raw)
    brier_cal = brier_score_multiclass(y_test_flat, probs_test_cal)
    print(f"[Calibration] ECE raw={ece_raw:.4f}  cal={ece_cal:.4f}")
    print(f"[Calibration] Brier raw={brier_raw:.4f}  cal={brier_cal:.4f}")

    plot_reliability(probs_test_raw, y_test_flat, os.path.join(OUT_DIR, f"reliability_raw_{tag}.png"))
    plot_reliability(probs_test_cal, y_test_flat, os.path.join(OUT_DIR, f"reliability_cal_{tag}.png"))

    # Save compact metrics for downstream comparison
    key = extract_key_metrics(y_true, y_pred, report_text=rep)
    with open(os.path.join(OUT_DIR, f"metrics_{tag}.json"), "w") as f:
        json.dump({"tag": tag, **key}, f, indent=2)

    return {"tag": tag, "y_true": y_true, "y_pred": y_pred,
            "report": rep, "cm": cm, "cm_row": cm_row, "metrics": key}

# =============================
# Experiments: A vs B
# A: includes identity-like cols (e.g., JOBID/GID/UID)
# B: excludes identity-like cols (more generalizable)
# =============================
# Column index reference (0-based):
# ['DATE','JOBID','GID','UID','JOBNAME','QNAME','WAIT','RUN','CPUS','CPU_T',
#  'MEMKB','VMEM','STATUS','E_CPUS','E_RUN','EXIT_CODE','SRUTime','APPL',
#  'NODESU','MODE','WallTime','isBB']

feat_A = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21]
feat_B = [6,7,8,9,10,11,12,13,14,15,16,18,19,20,21]  # identity-like removed

res_A = run_experiment(feat_A, tag="A", epochs=100, lr=1e-5, train_batch=4096, val_batch=8192, test_batch=8192)
res_B = run_experiment(feat_B, tag="B", epochs=100, lr=1e-5, train_batch=4096, val_batch=8192, test_batch=8192)

# =============================
# Compare & visualize deltas
# =============================
df_cmp = pd.DataFrame([
    {"tag": "A", "accuracy": res_A["metrics"]["accuracy"], "macro_f1": res_A["metrics"]["macro_f1"], "weighted_f1": res_A["metrics"]["weighted_f1"]},
    {"tag": "B", "accuracy": res_B["metrics"]["accuracy"], "macro_f1": res_B["metrics"]["macro_f1"], "weighted_f1": res_B["metrics"]["weighted_f1"]},
])
df_cmp.to_csv(os.path.join(OUT_DIR, "metrics_comparison.csv"), index=False)
print("\n=== Metrics Comparison (A vs B) ===")
print(df_cmp)

# Row-normalized confusion matrix difference (A - B)
cm_diff = res_A["cm_row"] - res_B["cm_row"]
plt.figure(figsize=(10,8))
ax = sns.heatmap(cm_diff, center=0.0, annot=False)
ax.set_title("CM Difference (A - B), Row-Normalized")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cm_diff_A_minus_B.png"), dpi=300); plt.close()

print("\nSaved outputs under ./power_comp_eval:")
print(" - classification_report_A.txt / classification_report_B.txt")
print(" - metrics_A.json / metrics_B.json / metrics_comparison.csv")
print(" - cm_counts_A.png / cm_row_A.png / accuracy_A.png / losses_A.png")
print(" - cm_counts_B.png / cm_row_B.png / accuracy_B.png / losses_B.png")
print(" - cm_diff_A_minus_B.png")
print(" - reliability_raw_*.png / reliability_cal_*.png")
