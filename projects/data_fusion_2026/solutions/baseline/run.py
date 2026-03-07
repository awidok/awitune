"""
Ensemble of two diverse models with per-target optimized weights:
1. Model A: DCNv2 with ASL loss (good for imbalanced targets)
2. Model B: DCNv2 with group-shared towers (good for correlated targets in groups)

Both models are trained from scratch, then per-target weights are optimized on validation set.
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
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda:0")


def dataloader_kwargs():
    workers = max(1, min(8, (os.cpu_count() or 1) // 2))
    return {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }

# Shared hyperparameters
SHARED_HPARAMS = dict(
    emb_dim=8,
    ple_bins=48,
    ple_emb_dim=8,
    bottleneck_dim=512,
    dropout=0.05,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    epochs=10,
    patience=7,
    grad_clip=1.0,
    log_every=50,
    val_every=200,
    warmup_epochs=2,
    ema_decay=0.999,
)

# Model A: ASL with positive class weighting (good for imbalanced targets)
MODEL_A_HPARAMS = dict(
    **SHARED_HPARAMS,
    cross_layers=4,
    deep_dims=[256, 128],
    tower_dim=96,
    gamma_neg=4,
    gamma_pos=0,
    alpha_type="adaptive",  # Adaptive alpha per target
)

# Model B: Group-shared towers (good for correlated targets)
MODEL_B_HPARAMS = dict(
    **SHARED_HPARAMS,
    cross_layers=3,
    deep_dims=[512, 256, 128],
    tower_dim=128,
    use_group_towers=True,  # Share towers within product groups
)


# ======================== LOSSES ========================

def asymmetric_loss(logits, targets, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, alpha=None):
    """ASL: Asymmetric loss for multi-label classification."""
    p = torch.sigmoid(logits)
    p = p.clamp(min=eps, max=1-eps)

    if alpha is None:
        alpha = 0.25

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.unsqueeze(0)

    loss_pos = -alpha * targets * torch.pow(1 - p, gamma_pos) * torch.log(p)
    p_neg = (p + clip).clamp(max=1)
    loss_neg = -(1 - alpha) * (1 - targets) * torch.pow(p_neg, gamma_neg) * torch.log(1 - p + eps)
    return loss_pos + loss_neg


def bce_with_logits_loss(logits, targets, pos_weight=None):
    """Standard BCE loss with optional positive class weighting."""
    if pos_weight is not None:
        pos_weight = pos_weight.unsqueeze(0)
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')


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


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        return x0 * self.W(x) + self.b + x


class SwiGLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, norm="batch"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_main, x_gate = self.proj(x).chunk(2, dim=-1)
        x = x_main * F.silu(x_gate)
        x = self.norm(x)
        x = self.drop(x)
        return x


# ======================== MODEL A: Individual Towers with ASL ========================

class DCNv2ModelA(nn.Module):
    """DCNv2 with individual task towers and ASL loss."""

    def __init__(self, n_num, cat_cardinalities, n_targets, hparams, ple_edges, target_groups=None):
        super().__init__()
        emb_dim = hparams["emb_dim"]
        ple_emb_dim = hparams["ple_emb_dim"]
        bottleneck_dim = hparams["bottleneck_dim"]
        tower_dim = hparams["tower_dim"]
        self.n_targets = n_targets

        self.num_encoder = NumericEncoderPLE(
            n_num, hparams["ple_bins"], ple_emb_dim, ple_edges
        )
        raw_dim = n_num * ple_emb_dim + len(cat_cardinalities) * emb_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim) for card in cat_cardinalities
        ])

        self.bottleneck = SwiGLUBlock(raw_dim, bottleneck_dim, dropout=0.0, norm="layer")

        self.cross_layers = nn.ModuleList([
            CrossLayer(bottleneck_dim) for _ in range(hparams["cross_layers"])
        ])
        self.cross_norm = nn.LayerNorm(bottleneck_dim)

        deep_layers = []
        prev = bottleneck_dim
        for dim in hparams["deep_dims"]:
            deep_layers.append(
                SwiGLUBlock(prev, dim, dropout=hparams["dropout"], norm="batch")
            )
            prev = dim
        self.deep = nn.Sequential(*deep_layers)

        shared_dim = bottleneck_dim + prev

        # Individual towers for each target
        self.towers = nn.ModuleList([
            nn.Sequential(
                SwiGLUBlock(shared_dim, tower_dim, dropout=hparams["dropout"], norm="batch"),
                nn.Linear(tower_dim, 1),
            ) for _ in range(n_targets)
        ])

        # Learnable loss weights (uncertainty weighting)
        self.log_vars = nn.Parameter(torch.zeros(n_targets))

    def forward(self, x_num, x_is_nan, x_cat):
        x_num_enc = self.num_encoder(x_num, x_is_nan)
        embs = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))]
        x = self.bottleneck(torch.cat([x_num_enc] + embs, dim=1))

        x_cross = x
        for cl in self.cross_layers:
            x_cross = cl(x, x_cross)
        x_cross = self.cross_norm(x_cross)

        x_deep = self.deep(x)
        shared_rep = torch.cat([x_cross, x_deep], dim=1)

        outs = [tower(shared_rep) for tower in self.towers]
        return torch.cat(outs, dim=1)


# ======================== MODEL B: Group-Shared Towers ========================

class DCNv2ModelB(nn.Module):
    """DCNv2 with group-shared towers (one tower per product group)."""

    def __init__(self, n_num, cat_cardinalities, n_targets, hparams, ple_edges, target_groups):
        super().__init__()
        emb_dim = hparams["emb_dim"]
        ple_emb_dim = hparams["ple_emb_dim"]
        bottleneck_dim = hparams["bottleneck_dim"]
        tower_dim = hparams["tower_dim"]
        self.n_targets = n_targets
        self.target_groups = target_groups  # Dict: group_id -> [target_indices]

        self.num_encoder = NumericEncoderPLE(
            n_num, hparams["ple_bins"], ple_emb_dim, ple_edges
        )
        raw_dim = n_num * ple_emb_dim + len(cat_cardinalities) * emb_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim) for card in cat_cardinalities
        ])

        self.bottleneck = SwiGLUBlock(raw_dim, bottleneck_dim, dropout=0.0, norm="layer")

        self.cross_layers = nn.ModuleList([
            CrossLayer(bottleneck_dim) for _ in range(hparams["cross_layers"])
        ])
        self.cross_norm = nn.LayerNorm(bottleneck_dim)

        deep_layers = []
        prev = bottleneck_dim
        for dim in hparams["deep_dims"]:
            deep_layers.append(
                SwiGLUBlock(prev, dim, dropout=hparams["dropout"], norm="batch")
            )
            prev = dim
        self.deep = nn.Sequential(*deep_layers)

        shared_dim = bottleneck_dim + prev

        # Group-specific towers: one tower per group
        n_groups = len(target_groups)
        self.group_towers = nn.ModuleList([
            nn.Sequential(
                SwiGLUBlock(shared_dim, tower_dim, dropout=hparams["dropout"], norm="batch"),
                nn.Linear(tower_dim, len(target_groups[g])),
            ) for g in range(n_groups)
        ])

        # Map target index to (group_id, position_in_group)
        self.target_to_group = {}
        for g, targets in target_groups.items():
            for pos, t in enumerate(targets):
                self.target_to_group[t] = (g, pos)

        self.log_vars = nn.Parameter(torch.zeros(n_targets))

    def forward(self, x_num, x_is_nan, x_cat):
        x_num_enc = self.num_encoder(x_num, x_is_nan)
        embs = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))]
        x = self.bottleneck(torch.cat([x_num_enc] + embs, dim=1))

        x_cross = x
        for cl in self.cross_layers:
            x_cross = cl(x, x_cross)
        x_cross = self.cross_norm(x_cross)

        x_deep = self.deep(x)
        shared_rep = torch.cat([x_cross, x_deep], dim=1)

        # Compute outputs per group
        outputs = torch.zeros(x_num.shape[0], self.n_targets, device=x_num.device)
        for g, tower in enumerate(self.group_towers):
            group_out = tower(shared_rep)  # (B, n_targets_in_group)
            for pos, t in enumerate(self.target_groups[g]):
                outputs[:, t] = group_out[:, pos]

        return outputs


# ======================== TRAINING UTILITIES ========================

def get_target_groups(target_cols):
    """Extract product groups from target names (target_X_Y -> group X)."""
    groups = {}
    for i, col in enumerate(target_cols):
        # Parse target_1_1 -> group 1
        parts = col.split('_')
        group_id = int(parts[1])
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)

    # Convert to consecutive indices
    group_list = {}
    for new_id, (old_id, targets) in enumerate(sorted(groups.items())):
        group_list[new_id] = targets

    return group_list


def compute_alpha_weights(y_train, target_cols):
    """Compute per-target alpha weights based on positive rates."""
    pos_rates = y_train.mean(axis=0)
    # Higher alpha for rare positive classes
    alphas = np.array([min(0.75, 0.25 + 0.5 * (1 - pr)) for pr in pos_rates])
    return torch.tensor(alphas, dtype=torch.float32)


def train_model(model, train_loader, val_loader, target_cols, hparams, exp_dir, model_name,
                loss_type="asl", alpha=None, pos_weight=None):
    """Train a model with EMA and return best predictions."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = model.to(DEVICE)

    # Initialize EMA
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(hparams["ema_decay"]))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )

    # Learning rate schedule
    def warmup_lambda(epoch):
        if epoch < hparams["warmup_epochs"]:
            return 0.1 + 0.9 * (epoch / hparams["warmup_epochs"])
        else:
            progress = (epoch - hparams["warmup_epochs"]) / (hparams["epochs"] - hparams["warmup_epochs"])
            return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

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

            logits = ema_model.module(x_num, x_nan, x_cat)
            val_preds.append(torch.sigmoid(logits).cpu().numpy())
            val_labels.append(y.numpy())

        val_preds = np.concatenate(val_preds)
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

            logits = model(x_num, x_nan, x_cat)

            if loss_type == "asl":
                losses_per_target = asymmetric_loss(
                    logits, y, gamma_neg=hparams["gamma_neg"],
                    gamma_pos=hparams["gamma_pos"], alpha=alpha
                ).mean(dim=0)
            else:
                losses_per_target = bce_with_logits_loss(
                    logits, y, pos_weight=pos_weight
                ).mean(dim=0)

            log_vars = model.log_vars
            total_loss_batch = torch.sum(
                torch.exp(-log_vars) * losses_per_target + log_vars
            )

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
            optimizer.step()

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
                    best_state = {k: v.cpu().clone() for k, v in ema_model.module.state_dict().items()}
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
            best_state = {k: v.cpu().clone() for k, v in ema_model.module.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= hparams["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and get final predictions
    ema_model.module.load_state_dict(best_state)
    ema_model.module.eval()

    torch.save(best_state, str(exp_dir / "models" / f"{model_name}.pt"))

    macro_auc, aucs, val_preds = run_validation()

    print(f"\n{model_name} Best Val Macro ROC-AUC: {macro_auc:.6f}")

    writer.close()

    return val_preds, macro_auc, aucs, training_log, ema_model


def optimize_ensemble_weights(val_preds_a, val_preds_b, val_labels):
    """Optimize per-target weights for ensemble using validation set."""
    n_targets = val_labels.shape[1]
    best_weights = []

    print("\nOptimizing per-target ensemble weights...")

    for t in range(n_targets):
        y_true = val_labels[:, t]
        pred_a = val_preds_a[:, t]
        pred_b = val_preds_b[:, t]

        def ensemble_loss(w):
            # w[0] = weight for model A, w[1] = weight for model B
            # Ensure weights sum to 1
            w_sum = w[0] + w[1]
            if w_sum < 1e-6:
                return 1.0

            w_a = w[0] / w_sum
            w_b = w[1] / w_sum

            ensemble_pred = w_a * pred_a + w_b * pred_b

            # Negative AUC (we want to maximize)
            try:
                auc = roc_auc_score(y_true, ensemble_pred)
                return 1.0 - auc
            except ValueError:
                return 1.0

        # Try multiple initializations
        best_auc = 0
        best_w = [0.5, 0.5]

        for init in [[0.5, 0.5], [0.7, 0.3], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9]]:
            result = minimize(ensemble_loss, init, method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 0.01})
            w = result.x
            w_sum = w[0] + w[1]
            if w_sum > 1e-6:
                w_a = w[0] / w_sum
                w_b = w[1] / w_sum
                ensemble_pred = w_a * pred_a + w_b * pred_b
                try:
                    auc = roc_auc_score(y_true, ensemble_pred)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = [w_a, w_b]
                except ValueError:
                    pass

        best_weights.append(best_w)

        if (t + 1) % 10 == 0:
            print(f"  Target {t+1}/{n_targets}: weight_A={best_w[0]:.3f}, weight_B={best_w[1]:.3f}")

    return np.array(best_weights)


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
            "model_a_hparams": MODEL_A_HPARAMS,
            "model_b_hparams": MODEL_B_HPARAMS,
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
    n_bins = SHARED_HPARAMS["ple_bins"]
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
    # Get target groups for Model B
    target_groups = get_target_groups(target_cols)
    print(f"  Target groups: {len(target_groups)} groups")

    # Compute alpha weights for Model A
    alpha_weights = compute_alpha_weights(y_train, target_cols)
    print(f"  Alpha weights computed: range [{alpha_weights.min():.4f}, {alpha_weights.max():.4f}]")

    del train_main, val_main, test_main, contest_main, train_target, val_target, test_target
    gc.collect()

    print(f"  Total preprocessing: {time.time() - t0:.1f}s")

    # ===== DATASETS =====
    t_ds = time.time()
    print("Creating datasets …")

    train_ds = TensorDataset(
        torch.tensor(train_num),
        torch.tensor(train_is_nan),
        torch.tensor(train_cat),
        torch.tensor(y_train),
    )
    val_ds = TensorDataset(
        torch.tensor(val_num),
        torch.tensor(val_is_nan),
        torch.tensor(val_cat),
        torch.tensor(y_val),
    )

    # Save validation labels for ensemble optimization before deleting
    y_val_for_ensemble = y_val.copy()

    del train_num, train_is_nan, train_cat, y_train
    del val_num, val_is_nan, val_cat, y_val
    gc.collect()

    train_loader = DataLoader(
        train_ds,
        batch_size=SHARED_HPARAMS["batch_size"],
        shuffle=True,
        **dataloader_kwargs(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=SHARED_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )
    print(f"  Datasets created in {time.time() - t_ds:.1f}s")

    # ===== TRAIN MODEL A =====
    model_a = DCNv2ModelA(
        len(num_cols), cat_cardinalities, len(target_cols),
        MODEL_A_HPARAMS, ple_edges_t, target_groups
    )

    val_preds_a, auc_a, aucs_a, log_a, ema_a = train_model(
        model_a, train_loader, val_loader, target_cols,
        MODEL_A_HPARAMS, exp_dir, "model_a_asl",
        loss_type="asl", alpha=alpha_weights.to(DEVICE, non_blocking=True)
    )

    # ===== TRAIN MODEL B =====
    model_b = DCNv2ModelB(
        len(num_cols), cat_cardinalities, len(target_cols),
        MODEL_B_HPARAMS, ple_edges_t, target_groups
    )

    val_preds_b, auc_b, aucs_b, log_b, ema_b = train_model(
        model_b, train_loader, val_loader, target_cols,
        MODEL_B_HPARAMS, exp_dir, "model_b_groups",
        loss_type="bce", pos_weight=None
    )

    # ===== OPTIMIZE ENSEMBLE WEIGHTS =====
    best_weights = optimize_ensemble_weights(val_preds_a, val_preds_b, y_val_for_ensemble)

    # Apply optimized weights to get ensemble predictions
    val_preds_ensemble = np.zeros_like(val_preds_a)
    for t in range(len(target_cols)):
        val_preds_ensemble[:, t] = (
            best_weights[t, 0] * val_preds_a[:, t] +
            best_weights[t, 1] * val_preds_b[:, t]
        )

    # Compute final metrics
    per_target_auc = {}
    for j, col in enumerate(target_cols):
        try:
            auc = roc_auc_score(y_val_for_ensemble[:, j], val_preds_ensemble[:, j])
        except ValueError:
            auc = 0.5
        per_target_auc[col] = round(auc, 6)

    macro_auc = np.mean(list(per_target_auc.values()))

    print(f"\n{'='*60}")
    print(f"ENSEMBLE Val Macro ROC-AUC: {macro_auc:.6f}")
    print(f"Model A (ASL): {auc_a:.6f}")
    print(f"Model B (Groups): {auc_b:.6f}")
    print(f"{'='*60}")

    # Save val predictions
    val_preds_df = pl.DataFrame(val_preds_ensemble, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    val_submit = val_customer_ids.hstack(val_preds_df)
    val_submit.write_parquet(exp_dir / "val_predictions.parquet")
    print(f"Saved val_predictions.parquet ({len(val_submit):,} rows)")

    # ===== PREDICTION ON LOCAL_TEST =====
    print("\nPredicting on local_test …")
    test_ds = TensorDataset(
        torch.tensor(test_num),
        torch.tensor(test_is_nan),
        torch.tensor(test_cat),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=SHARED_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )

    # Get predictions from both models
    test_preds_a_list = []
    test_preds_b_list = []

    ema_a.module.eval()
    ema_b.module.eval()

    with torch.no_grad():
        for batch in test_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)

            logits_a = ema_a.module(x_num, x_nan, x_cat)
            test_preds_a_list.append(torch.sigmoid(logits_a).cpu().numpy())

            logits_b = ema_b.module(x_num, x_nan, x_cat)
            test_preds_b_list.append(torch.sigmoid(logits_b).cpu().numpy())

    test_preds_a = np.concatenate(test_preds_a_list).astype(np.float64)
    test_preds_b = np.concatenate(test_preds_b_list).astype(np.float64)

    # Ensemble with optimized weights
    test_preds_ensemble = np.zeros_like(test_preds_a)
    for t in range(len(target_cols)):
        test_preds_ensemble[:, t] = (
            best_weights[t, 0] * test_preds_a[:, t] +
            best_weights[t, 1] * test_preds_b[:, t]
        )

    test_preds_df = pl.DataFrame(test_preds_ensemble, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    test_submit = test_customer_ids.hstack(test_preds_df)
    test_submit.write_parquet(exp_dir / "test_predictions.parquet")
    print(f"Saved test_predictions.parquet ({len(test_submit):,} rows)")

    # ===== PREDICTION ON CONTEST TEST =====
    print("\nPredicting on contest test …")
    contest_ds = TensorDataset(
        torch.tensor(contest_num),
        torch.tensor(contest_is_nan),
        torch.tensor(contest_cat),
    )
    contest_loader = DataLoader(
        contest_ds,
        batch_size=SHARED_HPARAMS["batch_size"] * 2,
        **dataloader_kwargs(),
    )

    contest_preds_a_list = []
    contest_preds_b_list = []

    with torch.no_grad():
        for batch in contest_loader:
            x_num = batch[0].to(DEVICE, non_blocking=True)
            x_nan = batch[1].to(DEVICE, non_blocking=True)
            x_cat = batch[2].to(DEVICE, non_blocking=True)

            logits_a = ema_a.module(x_num, x_nan, x_cat)
            contest_preds_a_list.append(torch.sigmoid(logits_a).cpu().numpy())

            logits_b = ema_b.module(x_num, x_nan, x_cat)
            contest_preds_b_list.append(torch.sigmoid(logits_b).cpu().numpy())

    contest_preds_a = np.concatenate(contest_preds_a_list).astype(np.float64)
    contest_preds_b = np.concatenate(contest_preds_b_list).astype(np.float64)

    # Ensemble with optimized weights
    contest_preds_ensemble = np.zeros_like(contest_preds_a)
    for t in range(len(target_cols)):
        contest_preds_ensemble[:, t] = (
            best_weights[t, 0] * contest_preds_a[:, t] +
            best_weights[t, 1] * contest_preds_b[:, t]
        )

    contest_preds_df = pl.DataFrame(contest_preds_ensemble, schema=predict_cols).cast(
        {c: pl.Float64 for c in predict_cols}
    )
    submit = contest_customer_ids.hstack(contest_preds_df)
    submit.write_parquet(exp_dir / "submission.parquet")
    print(f"Saved submission.parquet ({len(submit):,} rows)")

    elapsed = time.time() - t0
    metrics = dict(
        val_macro_roc_auc=round(macro_auc, 6),
        model_a_val_auc=round(auc_a, 6),
        model_b_val_auc=round(auc_b, 6),
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
        "model_a": {
            "name": "ASL with Individual Towers",
            "logs": log_a,
            "final_auc": auc_a,
            "hparams": MODEL_A_HPARAMS,
        },
        "model_b": {
            "name": "Group-Shared Towers",
            "logs": log_b,
            "final_auc": auc_b,
            "hparams": MODEL_B_HPARAMS,
        },
        "ensemble": {
            "final_auc": macro_auc,
            "per_target_auc": per_target_auc,
            "best_weights": best_weights.tolist(),
            "weight_stats": {
                "mean_a": float(best_weights[:, 0].mean()),
                "mean_b": float(best_weights[:, 1].mean()),
                "std_a": float(best_weights[:, 0].std()),
                "std_b": float(best_weights[:, 1].std()),
            },
        },
        "target_cols": target_cols,
        "target_groups": {k: list(v) for k, v in target_groups.items()},
        "alpha_weights": alpha_weights.tolist(),
        "n_features": len(num_cols),
        "n_targets": len(target_cols),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "elapsed_min": round(elapsed / 60, 2),
    }
    with open(exp_dir / "training_logs.json", "w") as f:
        json.dump(training_logs, f, indent=2)

    print(f"Done in {elapsed / 60:.1f} min")
    print(f"Metrics saved to {exp_dir / 'metrics.json'}")
    print(f"Training logs saved to {exp_dir / 'training_logs.json'}")
    print(f"TensorBoard logs saved to {exp_dir / 'tb_logs'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCNv2 ensemble model")
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