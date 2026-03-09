"""
Deep Residual MLP - a simpler but powerful alternative to DCNv2.

Architecture:
- PLE encoding for numeric features with NaN embeddings
- Embeddings for categorical features
- Deep residual blocks with pre-norm, GELU activation
- Focal Loss for handling class imbalance
- EMA for model averaging
- Cosine annealing LR schedule
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda:0")

# Enable cuDNN benchmark for faster training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def dataloader_kwargs():
    """Optimized DataLoader for in-memory TensorDataset.
    
    For TensorDataset that already fits in memory, multiprocessing adds overhead.
    Use num_workers=0 for best performance with in-memory data.
    """
    return {
        "num_workers": 0,      # No multiprocessing overhead for in-memory data
        "pin_memory": False,   # Not needed when data is already on GPU
    }


# Hyperparameters for Deep Residual MLP
MODEL_HPARAMS = dict(
    emb_dim=8,
    ple_bins=48,
    ple_emb_dim=8,
    hidden_dim=512,
    n_blocks=8,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    epochs=15,
    patience=10,
    grad_clip=1.0,
    log_every=50,
    val_every=200,
    warmup_epochs=2,
    ema_decay=0.999,
    focal_gamma=2.0,
    focal_alpha=0.25,
    use_amp=True,  # Automatic Mixed Precision for faster training
)

# Teacher model hyperparameters (larger model)
TEACHER_HPARAMS = dict(
    emb_dim=8,
    ple_bins=48,
    ple_emb_dim=8,
    hidden_dim=768,  # Larger than student
    n_blocks=12,     # More blocks
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    epochs=20,       # More epochs for teacher
    patience=10,
    grad_clip=1.0,
    log_every=50,
    val_every=200,
    warmup_epochs=2,
    ema_decay=0.999,
    focal_gamma=2.0,
    focal_alpha=0.25,
    use_amp=True,  # Automatic Mixed Precision for faster training
)

# Distillation hyperparameters
DISTILLATION_HPARAMS = dict(
    temperature=3.0,  # Soften distributions
    alpha=0.7,        # Weight for hard labels (vs soft teacher predictions)
)


# ======================== LOSSES ========================

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1, else 1-p
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


def distillation_loss(student_logits, teacher_logits, targets, temperature=3.0, alpha=0.7, focal_gamma=2.0, focal_alpha=0.25):
    """Knowledge distillation loss combining hard targets and soft teacher predictions.

    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model
        targets: Ground truth labels
        temperature: Temperature for softening distributions
        alpha: Weight for hard labels (1-alpha for soft labels)
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter

    Returns:
        Combined loss per sample per target
    """
    # Hard target loss (focal loss with ground truth)
    hard_loss = focal_loss(student_logits, targets, gamma=focal_gamma, alpha=focal_alpha)

    # Soft target loss (KL divergence with teacher)
    # Soften distributions using temperature
    teacher_probs = torch.sigmoid(teacher_logits / temperature)

    # Use BCEWithLogitsLoss for student (fused sigmoid + BCE, more stable)
    soft_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_probs,
        reduction='none'
    )

    # Scale by temperature^2 (standard in knowledge distillation)
    soft_loss = soft_loss * (temperature ** 2)

    # Combine losses
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss

    return total_loss


# ======================== ENCODERS ========================

class PiecewiseLinearEncoding(nn.Module):
    """PLE: thermometer-style piecewise-linear encoding for numeric features."""

    def __init__(self, n_features, n_bins, emb_dim, edges):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.emb_dim = emb_dim
        self.register_buffer("left", edges[:, :-1])
        self.register_buffer("widths", edges[:, 1:] - edges[:, :-1])
        self.weight = nn.Parameter(torch.empty(n_features, n_bins, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        left = self.left.unsqueeze(0)
        widths = self.widths.unsqueeze(0)
        x_exp = x.unsqueeze(2)
        encoded = ((x_exp - left) / (widths + 1e-8)).clamp(0.0, 1.0)
        out = torch.einsum("bfk,fkd->bfd", encoded, self.weight)
        return out


class NumericEncoderPLE(nn.Module):
    """PLE for present values + learnable is_nan embedding per feature when NaN."""

    def __init__(self, n_features, ple_bins, ple_emb_dim, ple_edges):
        super().__init__()
        self.n_features = n_features
        self.ple_emb_dim = ple_emb_dim
        self.ple = PiecewiseLinearEncoding(n_features, ple_bins, ple_emb_dim, ple_edges)
        self.nan_embed = nn.Parameter(torch.empty(n_features, ple_emb_dim))
        nn.init.normal_(self.nan_embed, 0, 0.02)

    def forward(self, x_num, x_is_nan):
        B = x_num.shape[0]
        ple_out = self.ple(x_num)
        nan_mask = x_is_nan.unsqueeze(2).float()
        nan_emb = self.nan_embed.unsqueeze(0).expand(B, -1, -1)
        out = ple_out * (1 - nan_mask) + nan_emb * nan_mask
        return out.reshape(B, -1)


# ======================== RESIDUAL MLP ========================

class ResidualBlock(nn.Module):
    """Pre-norm residual block with SwiGLU activation."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # SwiGLU: gate and value projections
        self.fc_gate = nn.Linear(dim, dim * 2)
        self.fc_value = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual block with SwiGLU
        residual = x
        x = self.norm1(x)
        # SwiGLU: Swish(gate) * value
        gate = self.fc_gate(x)
        value = self.fc_value(x)
        # Swish activation: x * sigmoid(x)
        x = (gate * torch.sigmoid(gate)) * value
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class DeepResidualMLP(nn.Module):
    """Deep Residual MLP for multi-label classification."""

    def __init__(self, input_dim, hidden_dim=512, n_blocks=8, n_targets=41, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_targets)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.head(x)


# ======================== TRAINING UTILITIES ========================

def train_model(model, train_loader, val_loader, target_cols, hparams, exp_dir, model_name):
    """Train a model with EMA and return best predictions."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)
    # Compile model for kernel fusion (PyTorch 2.0+)
    model = torch.compile(model)

    # Initialize EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(hparams["ema_decay"]))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"],
        fused=True  # Single kernel update for all parameters
    )

    # Learning rate schedule: warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < hparams["warmup_epochs"]:
            return 0.1 + 0.9 * (epoch / hparams["warmup_epochs"])
        else:
            progress = (epoch - hparams["warmup_epochs"]) / (hparams["epochs"] - hparams["warmup_epochs"])
            return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # AMP for faster training
    use_amp = hparams.get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"AMP enabled: {use_amp}")

    writer = SummaryWriter(log_dir=str(exp_dir / "tb_logs" / model_name))

    best_auc = 0
    best_state = None
    no_improve = 0
    global_step = 0

    @torch.no_grad()
    def run_validation():
        ema_model.module.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)
            y = batch[3]

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ema_model.module(x_num, x_nan, x_cat)
            # Keep tensors on GPU, transfer once at the end
            val_preds.append(torch.sigmoid(logits))
            val_labels.append(y)

        # Single transfer to CPU
        val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
        if isinstance(val_labels[0], torch.Tensor):
            val_labels = torch.cat(val_labels, dim=0).cpu().numpy()
        else:
            val_labels = np.concatenate(val_labels)

        aucs = []
        for j in range(val_labels.shape[1]):
            try:
                aucs.append(roc_auc_score(val_labels[:, j], val_preds[:, j]))
            except ValueError:
                aucs.append(0.5)
        return np.mean(aucs), aucs, val_preds

    training_log = []
    for epoch in range(hparams["epochs"]):
        t_ep = time.time()
        model.train()
        total_loss = 0
        n_samples = 0

        for batch in train_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)
            y = batch[3].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x_num, x_nan, x_cat)

                # Focal loss per target
                losses_per_target = focal_loss(
                    logits, y, gamma=hparams["focal_gamma"], alpha=hparams["focal_alpha"]
                ).mean(dim=0)

                # Simple average of losses per target
                total_loss_batch = losses_per_target.mean()

            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            ema_model.update_parameters(model)

            bs = len(y)
            total_loss += total_loss_batch.item() * bs
            n_samples += bs
            global_step += 1

            if global_step % hparams["log_every"] == 0:
                writer.add_scalar(f"loss/train_step_{model_name}", total_loss_batch.item(), global_step)

            if global_step % hparams["val_every"] == 0:
                macro_auc, aucs, _ = run_validation()
                writer.add_scalar(f"auc/val_macro_{model_name}", macro_auc, global_step)
                if macro_auc > best_auc:
                    best_auc = macro_auc
                    best_state = {k: v.clone() for k, v in ema_model.module.state_dict().items()}
                model.train()

        scheduler.step()
        macro_auc, aucs, _ = run_validation()
        avg_loss = total_loss / max(n_samples, 1)
        epoch_time = time.time() - t_ep
        current_lr = optimizer.param_groups[0]['lr']

        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_auc': macro_auc,
            'lr': current_lr,
            'time': epoch_time
        })

        print(
            f"  Epoch {epoch+1:2d}/{hparams['epochs']}  "
            f"loss={avg_loss:.4f}  val_macro_auc={macro_auc:.6f}  "
            f"lr={current_lr:.2e}  {epoch_time:.0f}s"
        )

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_state = {k: v.clone() for k, v in ema_model.module.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hparams["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and get final predictions
    ema_model.module.load_state_dict(best_state)
    ema_model.module.eval()

    # Move to CPU before saving
    best_state_cpu = {k: v.cpu() for k, v in best_state.items()}
    torch.save(best_state_cpu, str(exp_dir / "models" / f"{model_name}.pt"))

    macro_auc, aucs, val_preds = run_validation()

    print(f"\n{model_name} Best Val Macro ROC-AUC: {macro_auc:.6f}")

    writer.close()

    return val_preds, macro_auc, aucs, training_log, ema_model


def train_student_with_distillation(student_model, teacher_model, train_loader, val_loader, target_cols, hparams, distill_hparams, exp_dir, model_name):
    """Train student model with knowledge distillation from teacher."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} with Knowledge Distillation")
    print(f"{'='*60}")

    student_model = student_model.to(DEVICE)
    student_model = torch.compile(student_model)
    teacher_model = teacher_model.to(DEVICE)
    teacher_model = torch.compile(teacher_model)
    teacher_model.eval()  # Teacher is frozen

    # Initialize EMA for student
    ema_student = AveragedModel(student_model, multi_avg_fn=get_ema_multi_avg_fn(hparams["ema_decay"]))

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"],
        fused=True  # Single kernel update for all parameters
    )

    # Learning rate schedule: warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < hparams["warmup_epochs"]:
            return 0.1 + 0.9 * (epoch / hparams["warmup_epochs"])
        else:
            progress = (epoch - hparams["warmup_epochs"]) / (hparams["epochs"] - hparams["warmup_epochs"])
            return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # AMP for faster training
    use_amp = hparams.get("use_amp", True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_params = sum(p.numel() for p in student_model.parameters())
    print(f"Student model params: {n_params:,}")
    print(f"Temperature: {distill_hparams['temperature']}, Alpha: {distill_hparams['alpha']}")
    print(f"AMP enabled: {use_amp}")

    writer = SummaryWriter(log_dir=str(exp_dir / "tb_logs" / model_name))

    best_auc = 0
    best_state = None
    no_improve = 0
    global_step = 0

    @torch.no_grad()
    def run_validation():
        ema_student.module.eval()
        val_preds, val_labels = [], []
        for batch in val_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)
            y = batch[3]

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ema_student.module(x_num, x_nan, x_cat)
            # Keep tensors on GPU, transfer once at the end
            val_preds.append(torch.sigmoid(logits))
            val_labels.append(y)

        # Single transfer to CPU
        val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
        if isinstance(val_labels[0], torch.Tensor):
            val_labels = torch.cat(val_labels, dim=0).cpu().numpy()
        else:
            val_labels = np.concatenate(val_labels)

        aucs = []
        for j in range(val_labels.shape[1]):
            try:
                aucs.append(roc_auc_score(val_labels[:, j], val_preds[:, j]))
            except ValueError:
                aucs.append(0.5)
        return np.mean(aucs), aucs, val_preds

    training_log = []
    for epoch in range(hparams["epochs"]):
        t_ep = time.time()
        student_model.train()
        total_loss = 0
        n_samples = 0

        for batch in train_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)
            y = batch[3].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # Get student logits
                student_logits = student_model(x_num, x_nan, x_cat)

                # Get teacher logits (no gradients)
                with torch.no_grad():
                    teacher_logits = teacher_model(x_num, x_nan, x_cat)

                # Distillation loss per target
                losses_per_target = distillation_loss(
                    student_logits, teacher_logits, y,
                    temperature=distill_hparams["temperature"],
                    alpha=distill_hparams["alpha"],
                    focal_gamma=hparams["focal_gamma"],
                    focal_alpha=hparams["focal_alpha"]
                ).mean(dim=0)

                # Simple average of losses per target
                total_loss_batch = losses_per_target.mean()

            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), hparams["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            ema_student.update_parameters(student_model)

            bs = len(y)
            total_loss += total_loss_batch.item() * bs
            n_samples += bs
            global_step += 1

            if global_step % hparams["log_every"] == 0:
                writer.add_scalar(f"loss/train_step_{model_name}", total_loss_batch.item(), global_step)

            if global_step % hparams["val_every"] == 0:
                macro_auc, aucs, _ = run_validation()
                writer.add_scalar(f"auc/val_macro_{model_name}", macro_auc, global_step)
                if macro_auc > best_auc:
                    best_auc = macro_auc
                    best_state = {k: v.clone() for k, v in ema_student.module.state_dict().items()}
                student_model.train()

        scheduler.step()
        macro_auc, aucs, _ = run_validation()
        avg_loss = total_loss / max(n_samples, 1)
        epoch_time = time.time() - t_ep
        current_lr = optimizer.param_groups[0]['lr']

        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_auc': macro_auc,
            'lr': current_lr,
            'time': epoch_time
        })

        print(
            f"  Epoch {epoch+1:2d}/{hparams['epochs']}  "
            f"loss={avg_loss:.4f}  val_macro_auc={macro_auc:.6f}  "
            f"lr={current_lr:.2e}  {epoch_time:.0f}s"
        )

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_state = {k: v.clone() for k, v in ema_student.module.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hparams["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and get final predictions
    ema_student.module.load_state_dict(best_state)
    ema_student.module.eval()

    # Move to CPU before saving
    best_state_cpu = {k: v.cpu() for k, v in best_state.items()}
    torch.save(best_state_cpu, str(exp_dir / "models" / f"{model_name}.pt"))

    macro_auc, aucs, val_preds = run_validation()

    print(f"\n{model_name} Best Val Macro ROC-AUC: {macro_auc:.6f}")

    writer.close()

    return val_preds, macro_auc, aucs, training_log, ema_student


# ======================== MAIN ========================

def load_split(prefix: Path):
    """Load a split parquet file and separate features from targets."""
    df = pl.read_parquet(f"{prefix}.parquet")
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    features = df.select(feature_cols)
    targets = df.select(["customer_id"] + target_cols) if target_cols else None
    return features, targets


def run(train_prefix: Path, val_prefix: Path, test_prefix: Path, submit_prefix: Path, exp_dir: Path):
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    (exp_dir / "models").mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump({
            "seed": SEED,
            "teacher_hparams": TEACHER_HPARAMS,
            "student_hparams": MODEL_HPARAMS,
            "distillation_hparams": DISTILLATION_HPARAMS,
        }, f, indent=2)

    # ===== LOADING =====
    print("Loading features …")
    t_load = time.time()

    # Load data using prefixes
    train_main, train_target = load_split(train_prefix)
    val_main, val_target = load_split(val_prefix)
    test_main, test_target = load_split(test_prefix)
    contest_main, _ = load_split(submit_prefix)

    print(f"  Files loaded in {time.time() - t_load:.1f}s")

    target_cols = [c for c in train_target.columns if c.startswith("target")]
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    print(f"  Targets: {len(target_cols)}")

    cat_cols = sorted([c for c in train_main.columns if c.startswith("cat_feature")])
    num_cols = sorted([c for c in train_main.columns if c.startswith("num_feature")])
    pruning_path = train_prefix.parent / "feature_pruning.json"
    if not pruning_path.exists():
        raise FileNotFoundError(
            f"Missing required pruning file: {pruning_path}\n"
            "This solution requires feature_pruning.json in the data directory."
        )
    with open(pruning_path) as f:
        pruning = json.load(f)
    drop_features = set(pruning.get("drop_features", []))
    cat_cols = [c for c in cat_cols if c not in drop_features]
    num_cols = [c for c in num_cols if c not in drop_features]
    print(
        f"  Applied feature pruning: dropped {len(drop_features)} columns "
        f"(remaining cat={len(cat_cols)}, num={len(num_cols)})"
    )
    print(f"  Cat: {len(cat_cols)}, Num: {len(num_cols)}")

    # ===== CATEGORIES: mappings =====
    t_enc = time.time()
    print("Building cat mappings …")

    all_data = pl.concat([
        train_main.select(cat_cols),
        val_main.select(cat_cols),
        test_main.select(cat_cols),
        contest_main.select(cat_cols),
    ])

    cat_mappings = {}
    cat_cardinalities = []
    for c in cat_cols:
        uniq = all_data[c].drop_nulls().unique().sort().to_list()
        mapping = {v: i + 1 for i, v in enumerate(uniq)}
        cat_mappings[c] = mapping
        cat_cardinalities.append(len(mapping) + 1)
    del all_data

    print(f"  Mappings built in {time.time() - t_enc:.1f}s")

    # ===== ENCODE CATEGORIES =====
    def encode_cat(df):
        encoded = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for j, c in enumerate(cat_cols):
            vals = df[c].to_list()
            mapping = cat_mappings[c]
            encoded[:, j] = [mapping.get(v, 0) for v in vals]
        return encoded

    # ===== NUMERIC: fill NaN with 0, compute is_nan mask =====
    t_num = time.time()
    print("Processing numeric features …")

    def get_num(df):
        return df.select(num_cols).fill_null(0).to_numpy().astype(np.float32)

    def get_is_nan(df):
        is_nan = df.select(num_cols).to_numpy()
        return np.isnan(is_nan).astype(np.int64)

    train_num = get_num(train_main)
    val_num = get_num(val_main)
    test_num = get_num(test_main)
    contest_num = get_num(contest_main)

    train_is_nan = get_is_nan(train_main)
    val_is_nan = get_is_nan(val_main)
    test_is_nan = get_is_nan(test_main)
    contest_is_nan = get_is_nan(contest_main)

    train_idx = np.arange(len(train_num))
    val_idx = np.arange(len(val_num))
    print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}")

    # ===== PLE EDGES =====
    n_bins = MODEL_HPARAMS["ple_bins"]
    quantiles = np.linspace(0, 1, n_bins + 1)
    ple_edges = np.zeros((len(num_cols), n_bins + 1), dtype=np.float32)
    for j in range(len(num_cols)):
        col_vals = train_num[train_idx, j]
        col_vals = col_vals[~np.isnan(col_vals)]
        if len(col_vals) > 0:
            ple_edges[j] = np.quantile(col_vals, quantiles)
        else:
            ple_edges[j] = np.linspace(0, 1, n_bins + 1)
    ple_edges_t = torch.tensor(ple_edges, dtype=torch.float32)
    print(f"  PLE edges computed: {ple_edges_t.shape}")

    print(f"  Numeric done in {time.time() - t_num:.1f}s")

    # ===== ENCODE CATEGORIES =====
    t_cat = time.time()
    print("Encoding categories …")
    train_cat = encode_cat(train_main)
    val_cat = encode_cat(val_main)
    test_cat = encode_cat(test_main)
    contest_cat = encode_cat(contest_main)
    print(f"  Cat encoding done in {time.time() - t_cat:.1f}s")

    val_customer_ids = val_main.select("customer_id")
    test_customer_ids = test_main.select("customer_id")
    contest_customer_ids = contest_main.select("customer_id")

    y_train = train_target.select(target_cols).to_numpy().astype(np.float32)
    y_val = val_target.select(target_cols).to_numpy().astype(np.float32)

    del train_main, val_main, test_main, contest_main, train_target, val_target, test_target
    gc.collect()

    print(f"  Total preprocessing: {time.time() - t0:.1f}s")

    # ===== DATASETS =====
    t_ds = time.time()
    print("Creating datasets …")

    # Create datasets directly on GPU to eliminate PCIe transfer bottleneck
    train_ds = TensorDataset(
        torch.tensor(train_num, device=DEVICE),
        torch.tensor(train_is_nan, device=DEVICE),
        torch.tensor(train_cat, device=DEVICE),
        torch.tensor(y_train, device=DEVICE),
    )
    val_ds = TensorDataset(
        torch.tensor(val_num, device=DEVICE),
        torch.tensor(val_is_nan, device=DEVICE),
        torch.tensor(val_cat, device=DEVICE),
        torch.tensor(y_val, device=DEVICE),
    )

    del train_num, train_is_nan, train_cat, y_train
    del val_num, val_is_nan, val_cat, y_val
    gc.collect()

    train_loader = DataLoader(
        train_ds,
        batch_size=MODEL_HPARAMS["batch_size"],
        shuffle=True,
        **dataloader_kwargs(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=MODEL_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )
    print(f"  Datasets created in {time.time() - t_ds:.1f}s")

    # ===== BUILD MODEL =====
    class DeepResidualMLPWithEncoding(nn.Module):
        """Wrapper that includes feature encoding."""

        def __init__(self, n_num, cat_cardinalities, n_targets, hparams, ple_edges):
            super().__init__()
            emb_dim = hparams["emb_dim"]
            ple_emb_dim = hparams["ple_emb_dim"]

            # Feature encoders
            self.num_encoder = NumericEncoderPLE(
                n_num, hparams["ple_bins"], ple_emb_dim, ple_edges
            )
            self.embeddings = nn.ModuleList([
                nn.Embedding(card + 1, emb_dim) for card in cat_cardinalities
            ])

            # Calculate input dimension
            input_dim = n_num * ple_emb_dim + len(cat_cardinalities) * emb_dim

            # Main network
            self.model = DeepResidualMLP(
                input_dim=input_dim,
                hidden_dim=hparams["hidden_dim"],
                n_blocks=hparams["n_blocks"],
                n_targets=n_targets,
                dropout=hparams["dropout"]
            )

        def forward(self, x_num, x_is_nan, x_cat):
            # Encode features
            x_num_enc = self.num_encoder(x_num, x_is_nan)
            embs = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))]
            x = torch.cat([x_num_enc] + embs, dim=1)
            return self.model(x)

    # ===== BUILD TEACHER MODEL =====
    print("\n" + "="*60)
    print("PHASE 1: Training Large Teacher Model")
    print("="*60)

    teacher_model = DeepResidualMLPWithEncoding(
        len(num_cols), cat_cardinalities, len(target_cols),
        TEACHER_HPARAMS, ple_edges_t
    )

    # ===== TRAIN TEACHER MODEL =====
    teacher_val_preds, teacher_val_auc, teacher_val_aucs, teacher_training_log, teacher_ema_model = train_model(
        teacher_model, train_loader, val_loader, target_cols,
        TEACHER_HPARAMS, exp_dir, "teacher_model"
    )

    print(f"\nTeacher Model Val AUC: {teacher_val_auc:.6f}")

    # ===== BUILD STUDENT MODEL =====
    print("\n" + "="*60)
    print("PHASE 2: Training Student Model with Distillation")
    print("="*60)

    student_model = DeepResidualMLPWithEncoding(
        len(num_cols), cat_cardinalities, len(target_cols),
        MODEL_HPARAMS, ple_edges_t
    )

    # ===== TRAIN STUDENT WITH DISTILLATION =====
    val_preds, val_auc, val_aucs, student_training_log, ema_model = train_student_with_distillation(
        student_model, teacher_ema_model.module, train_loader, val_loader, target_cols,
        MODEL_HPARAMS, DISTILLATION_HPARAMS, exp_dir, "student_distilled"
    )

    print(f"\nStudent Model Val AUC: {val_auc:.6f}")
    print(f"Improvement over teacher: {val_auc - teacher_val_auc:+.6f}")

    # Compute per-target metrics
    y_val_for_metrics = val_loader.dataset.tensors[3].cpu().numpy()
    per_target_auc = {}
    for j, col in enumerate(target_cols):
        try:
            auc = roc_auc_score(y_val_for_metrics[:, j], val_preds[:, j])
        except ValueError:
            auc = 0.5
        per_target_auc[col] = round(auc, 6)

    macro_auc = np.mean(list(per_target_auc.values()))

    print(f"\n{'='*60}")
    print(f"Final Val Macro ROC-AUC: {macro_auc:.6f}")
    print(f"{'='*60}")

    # Save val predictions
    val_preds_df = pl.DataFrame(val_preds, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    val_submit = val_customer_ids.hstack(val_preds_df)
    val_submit.write_parquet(exp_dir / "val_predictions.parquet")
    print(f"Saved val_predictions.parquet ({len(val_submit):,} rows)")

    # ===== PREDICTION ON LOCAL_TEST =====
    print("\nPredicting on local_test …")
    test_ds = TensorDataset(
        torch.tensor(test_num, device=DEVICE),
        torch.tensor(test_is_nan, device=DEVICE),
        torch.tensor(test_cat, device=DEVICE),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=MODEL_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )

    test_preds_list = []
    ema_model.module.eval()
    use_amp = MODEL_HPARAMS.get("use_amp", True)

    with torch.no_grad():
        for batch in test_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ema_model.module(x_num, x_nan, x_cat)
            test_preds_list.append(torch.sigmoid(logits))

    test_preds = torch.cat(test_preds_list, dim=0).cpu().numpy().astype(np.float64)

    test_preds_df = pl.DataFrame(test_preds, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    test_submit = test_customer_ids.hstack(test_preds_df)
    test_submit.write_parquet(exp_dir / "test_predictions.parquet")
    print(f"Saved test_predictions.parquet ({len(test_submit):,} rows)")

    # ===== PREDICTION ON CONTEST TEST =====
    print("\nPredicting on contest test …")
    contest_ds = TensorDataset(
        torch.tensor(contest_num, device=DEVICE),
        torch.tensor(contest_is_nan, device=DEVICE),
        torch.tensor(contest_cat, device=DEVICE),
    )
    contest_loader = DataLoader(
        contest_ds,
        batch_size=MODEL_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )

    contest_preds_list = []

    with torch.no_grad():
        for batch in contest_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ema_model.module(x_num, x_nan, x_cat)
            contest_preds_list.append(torch.sigmoid(logits))

    contest_preds = torch.cat(contest_preds_list, dim=0).cpu().numpy().astype(np.float64)

    contest_preds_df = pl.DataFrame(contest_preds, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    submit = contest_customer_ids.hstack(contest_preds_df)
    submit.write_parquet(exp_dir / "submission.parquet")
    print(f"Saved submission.parquet ({len(submit):,} rows)")

    elapsed = time.time() - t0
    metrics = dict(
        val_macro_roc_auc=round(macro_auc, 6),
        teacher_val_macro_roc_auc=round(teacher_val_auc, 6),
        improvement_over_teacher=round(val_auc - teacher_val_auc, 6),
        per_target_auc=per_target_auc,
        n_num_features=len(num_cols),
        n_cat_features=len(cat_cols),
        n_targets=len(target_cols),
        elapsed_min=round(elapsed / 60, 2),
    )
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed training logs for agent analysis
    training_logs = {
        "teacher_model": {
            "name": "Teacher Model (Large)",
            "logs": teacher_training_log,
            "final_auc": teacher_val_auc,
            "hparams": TEACHER_HPARAMS,
        },
        "student_distilled": {
            "name": "Student Model with Distillation",
            "logs": student_training_log,
            "final_auc": val_auc,
            "hparams": MODEL_HPARAMS,
            "distillation_hparams": DISTILLATION_HPARAMS,
        },
        "improvement": {
            "student_over_teacher": val_auc - teacher_val_auc,
        },
        "target_cols": target_cols,
        "n_features": len(num_cols) + len(cat_cols),
        "n_targets": len(target_cols),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "elapsed_min": round(elapsed / 60, 2),
    }
    with open(exp_dir / "training_logs.json", "w") as f:
        json.dump(training_logs, f, indent=2)

    print(f"Done in {elapsed / 60:.1f} min")
    print(f"Metrics saved to {exp_dir / 'metrics.json'}")
    print(f"Training logs saved to {exp_dir / 'training_logs.json'}")
    print(f"TensorBoard logs saved to {exp_dir / 'tb_logs'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Residual MLP model")
    parser.add_argument("--train", required=True, help="Path prefix for training data (without .parquet)")
    parser.add_argument("--val", required=True, help="Path prefix for validation data (without .parquet)")
    parser.add_argument("--test", required=True, help="Path prefix for test data (without .parquet)")
    parser.add_argument("--submit", required=True, help="Path prefix for submission data (without .parquet)")
    parser.add_argument("output_dir", help="Output directory for models and predictions")
    args = parser.parse_args()

    run(
        train_prefix=Path(args.train),
        val_prefix=Path(args.val),
        test_prefix=Path(args.test),
        submit_prefix=Path(args.submit),
        exp_dir=Path(args.output_dir),
    )