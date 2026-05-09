import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# =========================================================
# Config
# =========================================================
DATASET = "tissuemnist"
NUM_CLIENTS = 5
ALPHA = 0.6
NUM_ROUNDS = 1 

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
PIN = (DEVICE == "cuda")

EMBED_DIM = 256

BATCH_SIZE = 128
N_WORKERS = 0

# Local embedding training (per client) - now CE-based
# EPOCHS_EMB = 8
EPOCHS_EMB = 50
# LR_EMB = 3e-4
LR_EMB = 1e-4
BACKBONE_TRAINABLE = True  # if False, only train last layer + classifier

# Partitioning mode
PARTITION_MODE = "iid_uniform" # {"iid_uniform", "iid_stratified", "dirichlet"}
PRINT_KL = True

# Gaussian modelling mode
FIT_MODE = "moments_full"  # "vi_diag" or "moments_full"

# Pyro VI
NUM_SVI_STEPS = 800
SVI_LR = 1e-3

# Full covariance
# RIDGE = 1e-3
RIDGE=1e-3
# SHRINK = 0.20
SHRINK=0.1
# USE_TIED_COV = False
USE_TIED_COV=True

# Synthetic generation + prior
# PER_CLASS_SYNTH = 8000
PER_CLASS_SYNTH =20000
# PRIOR_MIX_BETA = 0.1
PRIOR_MIX_BETA=0.5
PRIOR_CAP = 0.25
PRIOR_FLOOR = 0.05

# Head training (one-shot + continual)
# HEAD_LR = 1e-3
HEAD_LR=5e-4
HEAD_WD = 1e-4
# HEAD_EPOCHS = 30
HEAD_EPOCHS=100
HEAD_BIAS = True
LABEL_SMOOTH = 0.0

# Classifier type for the head
USE_COSINE_HEAD = False
# COS_S = 14.0
COS_S = 10.0

# Continual evaluation
USE_CONTINUAL_FED = True
NUM_TASKS = 4

# Continual recipe (replay + KD)
# REPLAY_PER_OLD_CLASS = 2000
REPLAY_PER_OLD_CLASS=8000
REPLAY_RATIO = 1.0
# DISTILL_LAMBDA = 0.5
DISTILL_LAMBDA = 0.2
DISTILL_T = 2.0


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_div(p: np.ndarray, q: np.ndarray, eps=1e-12) -> float:
    p = p.astype(np.float64) + eps
    q = q.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float((p * np.log(p / q)).sum())


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def mask_logits_to_classes(logits: torch.Tensor, class_ids: List[int]) -> torch.Tensor:
    """Mask all non-class_ids logits to -inf so argmax is only over those classes."""
    if logits.numel() == 0:
        return logits
    m = torch.full_like(logits, -1e9)
    m[:, class_ids] = logits[:, class_ids]
    return m


def distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


# =========================================================
# Data: MedMNIST
# =========================================================
from medmnist import INFO, TissueMNIST, PathMNIST, OrganAMNIST, OrganSMNIST,OCTMNIST,DermaMNIST,BloodMNIST

def build_medmnist(name: str):
    name = name.lower()
    info = INFO[name]
    DataClass = {
        "tissuemnist": TissueMNIST,
        "pathmnist": PathMNIST,
        "organamnist": OrganAMNIST,
        "organsmnist": OrganSMNIST,
        "octmnist": OCTMNIST,
        "dermamnist": DermaMNIST,
        "bloodmnist": BloodMNIST,
    }[name]

    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    print(f"MedMNIST={name}, in_ch={n_channels}, num_classes={n_classes}")

    to3 = T.Grayscale(num_output_channels=3) if n_channels == 1 else T.Lambda(lambda x: x)

    tr_transform = T.Compose([
        to3,
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    te_transform = tr_transform

    train_ds = DataClass(split="train", transform=tr_transform, download=True, root="./data")
    val_ds = DataClass(split="val", transform=te_transform, download=True, root="./data")
    test_ds = DataClass(split="test", transform=te_transform, download=True, root="./data")

    def _extract_labels(ds):
        ys = []
        for i in range(len(ds)):
            y = ds[i][1]
            if np.ndim(y) > 0:
                y = y[0]
            ys.append(int(y))
        return np.array(ys, dtype=int)

    y_train = _extract_labels(train_ds)
    return train_ds, val_ds, test_ds, y_train, n_channels, n_classes


# =========================================================
# Partitioners
# =========================================================
class IIDUniformPartitioner:
    def __init__(self, num_samples: int, num_partitions: int, seed: int = 0):
        self.num_samples = int(num_samples)
        self.num_partitions = int(num_partitions)
        self.seed = seed

    def load_partition(self, cid: int) -> List[int]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(self.num_samples)
        rng.shuffle(indices)
        parts = np.array_split(indices, self.num_partitions)
        return parts[cid].tolist()


class IIDStratifiedPartitioner:
    def __init__(self, labels: np.ndarray, num_partitions: int, seed: int = 0):
        self.labels = labels.astype(int)
        self.num_partitions = num_partitions
        self.seed = seed
        self.classes = np.unique(self.labels)

    def load_partition(self, cid: int) -> List[int]:
        rng = np.random.default_rng(self.seed)
        indices_by_c = {c: np.where(self.labels == c)[0] for c in self.classes}
        for c in self.classes:
            rng.shuffle(indices_by_c[c])

        buckets = [[] for _ in range(self.num_partitions)]
        for c in self.classes:
            idx = indices_by_c[c]
            parts = np.array_split(idx, self.num_partitions)
            for k in range(self.num_partitions):
                buckets[k].extend(parts[k].tolist())
        return buckets[cid]


class DirichletPartitioner:
    """Non-IID partition based on Dirichlet(alpha) over labels."""
    def __init__(self, labels: np.ndarray, num_partitions: int, alpha: float, min_size=10, seed: int = 0):
        self.labels = labels.astype(int)
        self.num_partitions = num_partitions
        self.alpha = float(alpha)
        self.min_size = int(min_size)
        self.seed = seed
        self.classes = np.unique(self.labels)

    def load_partition(self, cid: int) -> List[int]:
        rng = np.random.default_rng(self.seed)
        indices_by_c = {c: np.where(self.labels == c)[0] for c in self.classes}
        for c in self.classes:
            rng.shuffle(indices_by_c[c])

        props = rng.dirichlet([self.alpha] * len(self.classes), self.num_partitions)
        buckets = [[] for _ in range(self.num_partitions)]

        for ci, c in enumerate(self.classes):
            idx = indices_by_c[c]
            counts = (props[:, ci] * len(idx)).astype(int)
            start = 0
            for k in range(self.num_partitions):
                end = min(start + counts[k], len(idx))
                if start < end:
                    buckets[k].extend(idx[start:end].tolist())
                start = end

        # enforce minimal size (best-effort)
        for k in range(self.num_partitions):
            if len(buckets[k]) < self.min_size:
                pool = np.concatenate([np.array(b) for j, b in enumerate(buckets) if j != k],
                                      axis=0) if any(len(b) > 0 for b in buckets) else np.array([], dtype=int)
                if pool.size > 0:
                    needed = self.min_size - len(buckets[k])
                    chosen = rng.choice(pool, size=min(needed, len(pool)), replace=False)
                    buckets[k].extend(chosen.tolist())
        return buckets[cid]


# =========================================================
# Backbone + CE classifier for local embedding training
# =========================================================
class MobileNetAttentionEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 128, backbone_trainable: bool = True, in_channels: int = 3):
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
  

        if in_channels != 3:
            conv = base.features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=conv.bias is not None,
            )
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
                else:
                    new_conv.weight[:, :3] = conv.weight
                    if in_channels > 3:
                        for c in range(3, in_channels):
                            new_conv.weight[:, c] = conv.weight[:, c % 3]
            base.features[0][0] = new_conv

        self.features = base.features

        if not backbone_trainable:
            for p in self.features.parameters():
                p.requires_grad = False

        # keep BN frozen for stability
        def _freeze_bn(mod):
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
        self.features.apply(_freeze_bn)

        # light attention + projection
        self.attn = nn.MultiheadAttention(embed_dim=1280, num_heads=8, batch_first=True)
        self.fc = nn.Linear(1280, embed_dim, bias=False)
        self.neck = nn.LayerNorm(embed_dim, elementwise_affine=True)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")

    def forward(self, x):
        f = self.features(x)  # (B, 1280, H, W)
        B, C, H, W = f.shape
        f_flat = f.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        attn_out, _ = self.attn(f_flat, f_flat, f_flat)
        v = attn_out.mean(dim=1)  # (B, C)
        z = self.fc(v)
        z = self.neck(z)
        return z


# class MobileNetAttentionEmbedder(nn.Module):
#     def __init__(self, embed_dim: int = 128, backbone_trainable: bool = True, in_channels: int = 3):
#         super().__init__()
#         base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)  # ← only this line changes

#         if in_channels != 3:
#             conv = base.features[0][0]
#             new_conv = nn.Conv2d(
#                 in_channels, conv.out_channels, kernel_size=conv.kernel_size,
#                 stride=conv.stride, padding=conv.padding, bias=conv.bias is not None,
#             )
#             with torch.no_grad():
#                 if in_channels == 1:
#                     new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
#                 else:
#                     new_conv.weight[:, :3] = conv.weight
#                     if in_channels > 3:
#                         for c in range(3, in_channels):
#                             new_conv.weight[:, c] = conv.weight[:, c % 3]
#             base.features[0][0] = new_conv

#         self.features = base.features

#         if not backbone_trainable:
#             for p in self.features.parameters():
#                 p.requires_grad = False

#         def _freeze_bn(mod):
#             if isinstance(mod, nn.BatchNorm2d):
#                 mod.eval()
#                 for param in mod.parameters():
#                     param.requires_grad = False
#         self.features.apply(_freeze_bn)

#         # ← ONLY THESE THREE LINES CHANGE FROM 1280 → 960
#         self.attn = nn.MultiheadAttention(embed_dim=960, num_heads=8, batch_first=True)
#         self.fc = nn.Linear(960, embed_dim, bias=False)
#         self.neck = nn.LayerNorm(embed_dim, elementwise_affine=True)
#         nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")

#     def forward(self, x):
#         f = self.features(x)  # (B, 960, H, W)
#         B, C, H, W = f.shape
#         f_flat = f.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
#         attn_out, _ = self.attn(f_flat, f_flat, f_flat)
#         v = attn_out.mean(dim=1)  # (B, C)
#         z = self.fc(v)
#         z = self.neck(z)
#         return z
        


class CEClassifier(nn.Module):
    """Used only for local embedding training."""
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=True)

    def forward(self, z):
        return self.fc(z)


# =========================================================
# Heads (server)
# =========================================================
class CosineHead(nn.Module):
    def __init__(self, dim, num_classes, s=COS_S, bias=HEAD_BIAS):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.s = s
        self.bias = nn.Parameter(torch.zeros(num_classes)) if bias else None

    def forward(self, z):
        z = F.normalize(z, dim=1)
        w = F.normalize(self.weight, dim=1)
        logits = self.s * (z @ w.t())
        if self.bias is not None:
            logits = logits + self.bias
        return logits


class LinearHead(nn.Module):
    def __init__(self, dim, num_classes, bias=True):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=bias)

    def forward(self, z):
        return self.fc(z)


def make_head(num_classes: int) -> nn.Module:
    if USE_COSINE_HEAD:
        return CosineHead(EMBED_DIM, num_classes, s=COS_S, bias=HEAD_BIAS)
    return LinearHead(EMBED_DIM, num_classes, bias=True)


# =========================================================
# Data loaders per client
# =========================================================
def make_client_loaders(train_ds, val_ds, train_idx, val_idx):
    tr_subset = Subset(train_ds, train_idx)
    va_subset = Subset(val_ds, val_idx)

    labels = np.array([
        int(train_ds[i][1] if np.ndim(train_ds[i][1]) == 0 else train_ds[i][1][0])
        for i in train_idx
    ])
    num_classes = int(labels.max()) + 1
    class_counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (class_counts[labels] + 1e-6)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_loader = DataLoader(tr_subset, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=N_WORKERS, pin_memory=PIN, drop_last=False)
    va_loader = DataLoader(va_subset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=N_WORKERS, pin_memory=PIN, drop_last=False)
    return tr_loader, va_loader


# =========================================================
# Local training (CE)
# =========================================================
def train_embedder_ce(backbone: nn.Module, clf: nn.Module, train_loader, val_loader):
    backbone.to(DEVICE)
    clf.to(DEVICE)

    params = list(backbone.parameters()) + list(clf.parameters())
    opt = torch.optim.AdamW(params, lr=LR_EMB, weight_decay=1e-4)

    best_val = float("inf")

    for ep in range(1, EPOCHS_EMB + 1):
        backbone.train()
        clf.train()
        loss_accum = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Train CE ep={ep}", ncols=0, leave=False)
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).view(-1).long()

            z = backbone(x)
            logits = clf(z)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            loss_accum += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss_accum / max(1, n):.4f}")
        print(f"[Local train] ep={ep} loss={loss_accum / max(1, n):.4f}")

        backbone.eval()
        clf.eval()
        val_loss = 0.0
        m = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True).view(-1).long()
                z = backbone(x)
                logits = clf(z)
                val = F.cross_entropy(logits, y).item()
                bs = x.size(0)
                val_loss += val * bs
                m += bs
        val_loss /= max(1, m)
        print(f"[Local val] ep={ep} val_loss={val_loss:.4f}")
        best_val = min(best_val, val_loss)

    print(f"Best local val_loss={best_val:.4f}")


# =========================================================
# Collect embeddings from a backbone
# =========================================================
def collect_embeddings(backbone, loader, normalize: bool = True):
    backbone.eval()
    Z, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.view(-1).long()
            z = backbone(x)
            if normalize:
                z = F.normalize(z, p=2, dim=1)
            Z.append(z.cpu())
            Y.append(y.cpu())
    return torch.cat(Z, 0), torch.cat(Y, 0)


# =========================================================
# VI-based per-class diagonal Gaussians on embeddings
# =========================================================
def fit_feature_gaussians_vi_diag(
    Z: torch.Tensor,
    Y: torch.Tensor,
    num_classes: int,
    steps: int = NUM_SVI_STEPS,
    lr: float = SVI_LR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert Z.ndim == 2 and Y.ndim == 1
    D = Z.size(1)
    pyro.clear_param_store()

    Z_tensor = Z.to(DEVICE)
    Y_tensor = Y.to(DEVICE)

    def model():
        with pyro.plate("classes", num_classes):
            mu = pyro.sample("mu", dist.Normal(torch.zeros(D, device=DEVICE), 1.0).to_event(1))
            log_sigma = pyro.sample("log_sigma", dist.Normal(torch.zeros(D, device=DEVICE), 1.0).to_event(1))

        with pyro.plate("data", Z_tensor.size(0)):
            c = Y_tensor
            sigma = torch.exp(log_sigma[c])
            pyro.sample("z_obs", dist.Normal(mu[c], sigma).to_event(1), obs=Z_tensor)

    def guide():
        mu_loc = pyro.param("mu_loc", torch.zeros(num_classes, D, device=DEVICE))
        mu_scale = pyro.param("mu_scale", torch.ones(num_classes, D, device=DEVICE),
                              constraint=dist.constraints.positive)
        log_sigma_loc = pyro.param("log_sigma_loc", torch.zeros(num_classes, D, device=DEVICE))
        log_sigma_scale = pyro.param("log_sigma_scale", torch.ones(num_classes, D, device=DEVICE),
                                     constraint=dist.constraints.positive)
        with pyro.plate("classes", num_classes):
            pyro.sample("mu", dist.Normal(mu_loc, mu_scale).to_event(1))
            pyro.sample("log_sigma", dist.Normal(log_sigma_loc, log_sigma_scale).to_event(1))

    svi = SVI(model, guide, pyro.optim.Adam({"lr": lr}), loss=Trace_ELBO())
    for step in tqdm(range(steps), desc="SVI fit diag Gaussians", ncols=0, leave=False):
        loss = svi.step()
        if (step + 1) % 200 == 0 or step == 0:
            print(f"[SVI] step={step+1}/{steps}, loss={loss:.4f}")

    mu_loc = pyro.param("mu_loc").detach().cpu()
    log_sigma_loc = pyro.param("log_sigma_loc").detach().cpu()

    mu_out = mu_loc
    L_out = torch.diag_embed(torch.exp(log_sigma_loc))  # diag as Cholesky
    return mu_out, L_out


# =========================================================
# Moment-based full covariance Gaussian per class
# =========================================================
def fit_feature_gaussians_fullrank(
    Z: torch.Tensor,
    Y: torch.Tensor,
    num_classes: int,
    ridge: float = RIDGE,
    shrink: float = SHRINK,
    tied_cov: bool = USE_TIED_COV,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert Z.ndim == 2 and Y.ndim == 1
    D = Z.size(1)
    Zc = Z.cpu()
    Yc = Y.cpu()
    mus = torch.zeros(num_classes, D)
    covs = torch.zeros(num_classes, D, D)

    eye = torch.eye(D)

    if tied_cov:
        mean_all = Zc.mean(dim=0, keepdim=True)
        S = ((Zc - mean_all).T @ (Zc - mean_all)) / max(1, Zc.size(0) - 1)
        S = (1 - shrink) * S + shrink * S.diag().mean() * eye
        S = S + ridge * eye
        for c in range(num_classes):
            idx = (Yc == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            zc = Zc[idx]
            mus[c] = zc.mean(dim=0)
            covs[c] = S
    else:
        for c in range(num_classes):
            idx = (Yc == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            zc = Zc[idx]
            mus[c] = zc.mean(dim=0)
            if zc.size(0) > 1:
                S = ((zc - mus[c]).T @ (zc - mus[c])) / max(1, zc.size(0) - 1)
            else:
                S = eye.clone()
            S = (1 - shrink) * S + shrink * S.diag().mean() * eye
            S = S + ridge * eye
            covs[c] = S

    Ls = torch.zeros_like(covs)
    for c in range(num_classes):
        if covs[c].abs().sum() == 0:
            continue
        # add a tiny jitter for safety
        Ls[c] = torch.linalg.cholesky(covs[c] + 1e-6 * eye)

    return mus, Ls


# =========================================================
# Synthetic feature generation (server-side)
# =========================================================
def make_synth_loader_weighted(
    agg: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]],
    class_prior: torch.Tensor,
    total_synth: int,
    batch_size: int = BATCH_SIZE,
):
    Xs, Ys = [], []
    C = class_prior.numel()
    pr = class_prior.cpu().numpy()

    for c in range(C):
        items = agg.get(c, [])
        if len(items) == 0:
            continue
        n_c = max(1, int(round(total_synth * pr[c])))
        cnts = np.array([w for (_, _, w) in items], dtype=np.float64)
        cnts = cnts / (cnts.sum() + 1e-12)
        alloc = np.random.multinomial(n_c, cnts)

        for (mu_c, L_c, _), k in zip(items, alloc):
            if k <= 0:
                continue
            eps = torch.randn(k, mu_c.numel())
            x = eps @ L_c.T + mu_c
            x = F.normalize(x, p=2, dim=1)
            y = torch.full((k,), c, dtype=torch.long)
            Xs.append(x)
            Ys.append(y)

    if not Xs:
        raise RuntimeError("No synthetic samples generated.")
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)

    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=N_WORKERS, pin_memory=False, drop_last=False)


def make_synth_loader_weighted_subset(
    agg: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]],
    class_ids: List[int],
    class_prior: torch.Tensor,
    total_synth: int,
    batch_size: int = BATCH_SIZE,
):
    Xs, Ys = [], []
    pr = class_prior.detach().cpu().numpy()
    pr = pr / (pr.sum() + 1e-12)

    for j, c_global in enumerate(class_ids):
        items = agg.get(int(c_global), [])
        if len(items) == 0:
            continue

        n_c = max(1, int(round(total_synth * pr[j])))

        cnts = np.array([w for (_, _, w) in items], dtype=np.float64)
        cnts = cnts / (cnts.sum() + 1e-12)
        alloc = np.random.multinomial(n_c, cnts)

        for (mu_c, L_c, _), k in zip(items, alloc):
            if k <= 0:
                continue
            eps = torch.randn(k, mu_c.numel())
            x = eps @ L_c.T + mu_c
            x = F.normalize(x, p=2, dim=1)
            y = torch.full((k,), int(c_global), dtype=torch.long)
            Xs.append(x)
            Ys.append(y)

    if not Xs:
        raise RuntimeError("No synthetic samples generated for subset (agg missing those classes).")

    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=N_WORKERS, pin_memory=False, drop_last=False)


# =========================================================
# Continual helpers + evaluation
# =========================================================
def build_class_incremental_tasks(num_classes: int, num_tasks: int) -> List[List[int]]:
    num_tasks = max(1, min(num_tasks, num_classes))
    classes = np.arange(num_classes)
    splits = np.array_split(classes, num_tasks)
    return [split.astype(int).tolist() for split in splits]


def make_task_subset_indices(ds, cls_ids: List[int]) -> List[int]:
    cls_set = set(int(c) for c in cls_ids)
    indices = []
    for i in range(len(ds)):
        _, y = ds[i]
        y_int = int(y if np.ndim(y) == 0 else y[0])
        if y_int in cls_set:
            indices.append(i)
    return indices


def eval_class_il_on_subset(backbone, head, ds, eval_class_ids: List[int]) -> float:
    """Evaluate accuracy on ds restricted to eval_class_ids, predicting among eval_class_ids."""
    idx = make_task_subset_indices(ds, eval_class_ids)
    loader = DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=N_WORKERS, pin_memory=PIN)
    head.eval()
    backbone.eval()
    tot, cor = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).view(-1).long()
            z = F.normalize(backbone(x), p=2, dim=1)
            logits = head(z)
            logits = mask_logits_to_classes(logits, eval_class_ids)
            pred = logits.argmax(1)
            tot += x.size(0)
            cor += (pred == y).sum().item()
    return 100.0 * cor / max(1, tot)


# =========================================================
# Continual shared-head (replay CE + KD on replay)
# =========================================================
def run_continual_shared_head(
    global_backbone: nn.Module,
    agg: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]],
    prior: torch.Tensor,
    val_ds,
    test_ds,
    num_classes: int,
    client_gauss: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Optional[nn.Module]:

    if not USE_CONTINUAL_FED:
        print("[Continual] disabled.")
        return None

    device = next(global_backbone.parameters()).device
    global_backbone.eval()
    for p in global_backbone.parameters():
        p.requires_grad = False

    task_splits = build_class_incremental_tasks(num_classes, NUM_TASKS)

    print("\n=== Continual federated evaluation (single-head class-incremental) ===")
    for t, cls_ids in enumerate(task_splits):
        print(f"[Task {t}] classes: {cls_ids}")

    # -------------------------------------------------
    # Accuracy matrix A_{t,k} for forgetting
    # -------------------------------------------------
    task_acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS), dtype=np.float32)

    head = make_head(num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)

    # init head from average mu
    with torch.no_grad():
        avg_mu = torch.stack([m for m, _ in client_gauss]).mean(0)
        init_w = F.normalize(avg_mu, p=2, dim=1)
        if hasattr(head, "fc"):
            head.fc.weight.copy_(init_w.to(device))
            if head.fc.bias is not None:
                head.fc.bias.zero_()

    seen_classes: List[int] = []

    # =========================================================
    # Continual training loop
    # =========================================================
    for task_id, cls_ids in enumerate(task_splits):
        cls_ids = list(map(int, cls_ids))
        print(f"\n--- Task {task_id} --- classes={cls_ids}")

        teacher = make_head(num_classes).to(device)
        teacher.load_state_dict(head.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Current task synthetic data
        prior_task = prior[cls_ids]
        prior_task = prior_task / prior_task.sum().clamp_min(1e-8)
        synth_loader = make_synth_loader_weighted_subset(
            agg, cls_ids, prior_task,
            PER_CLASS_SYNTH * len(cls_ids),
            BATCH_SIZE
        )

        # Replay loader
        replay_iter = None
        if len(seen_classes) > 0:
            prior_old = prior[seen_classes]
            prior_old = prior_old / prior_old.sum().clamp_min(1e-8)
            replay_loader = make_synth_loader_weighted_subset(
                agg, seen_classes, prior_old,
                REPLAY_PER_OLD_CLASS * len(seen_classes),
                BATCH_SIZE
            )
            replay_iter = cycle_loader(replay_loader)

        # ---- Train head ----
        for ep in range(HEAD_EPOCHS):
            head.train()
            for z_cur, y_cur in synth_loader:
                z_cur = z_cur.to(device)
                y_cur = y_cur.to(device)

                logits_cur = head(z_cur)
                loss = F.cross_entropy(logits_cur, y_cur)

                if replay_iter is not None:
                    z_rep, y_rep = next(replay_iter)
                    z_rep = z_rep.to(device)
                    y_rep = y_rep.to(device)

                    logits_rep = head(z_rep)
                    ce_rep = F.cross_entropy(logits_rep, y_rep)

                    with torch.no_grad():
                        logits_t = teacher(z_rep)
                    kd = distill_kl(logits_rep, logits_t, DISTILL_T)

                    loss = loss + ce_rep + DISTILL_LAMBDA * kd

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Update seen classes
        for c in cls_ids:
            if c not in seen_classes:
                seen_classes.append(c)

        # =====================================================
        # Evaluate on all learned tasks (for forgetting)
        # =====================================================
        for k in range(task_id + 1):
            acc = eval_class_il_on_subset(
                global_backbone,
                head,
                test_ds,
                task_splits[k]
            )
            task_acc_matrix[task_id, k] = acc
            print(f"[Task {task_id}] TEST acc on Task {k}: {acc:.2f}%")

    # =========================================================
    # FINAL TEST (overall accuracy)
    # =========================================================
    print("\n=== FINAL TEST ===")
    overall = eval_class_il_on_subset(
        global_backbone, head, test_ds, list(range(num_classes))
    )
    print(f"[FINAL TEST] overall acc={overall:.2f}% (class-IL, all classes)")

    # =========================================================
    # Forgetting computation
    # =========================================================
    forgetting = []
    
    T = NUM_TASKS
    
    for i in range(T - 1):
        # max accuracy on task i before the final task
        best_pre_final = task_acc_matrix[:T-1, i].max()
        # final accuracy on task i
        final = task_acc_matrix[T-1, i]
        forgetting.append(best_pre_final - final)
    
    avg_forgetting = float(np.mean(forgetting))
    
    print("\n=== Forgetting Analysis (paper-aligned) ===")
    for i, f in enumerate(forgetting):
        print(f"Task {i} forgetting: {f:.2f}%")
    print(f"Average forgetting: {avg_forgetting:.2f}%")

    
    # forgetting = []

    # for k in range(NUM_TASKS - 1):
    #     # Best accuracy achieved on task k at any time after it appeared
    #     best = task_acc_matrix[k:, k].max()
    #     # Final accuracy on task k
    #     final = task_acc_matrix[-1, k]
    #     forgetting.append(best - final)
    
    # avg_forgetting = float(np.mean(forgetting))
    
    # print("\n=== Forgetting Analysis (standard CL) ===")
    # for k, f in enumerate(forgetting):
    #     print(f"Task {k} forgetting: {f:.2f}%")
    # print(f"Average forgetting: {avg_forgetting:.2f}%")
    
    # forgetting = []
    # for k in range(NUM_TASKS - 1):
    #     best = task_acc_matrix[k, k]
    #     final = task_acc_matrix[-1, k]
    #     forgetting.append(best - final)

    # avg_forgetting = float(np.mean(forgetting))

    # print("\n=== Forgetting Analysis ===")
    # for k, f in enumerate(forgetting):
    #     print(f"Task {k} forgetting: {f:.2f}%")
    # print(f"Average forgetting: {avg_forgetting:.2f}%")

    return head



# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_ds, val_ds, test_ds, y_train_all, in_ch, num_classes = build_medmnist(DATASET)
    TOTAL_SYNTH = int(PER_CLASS_SYNTH * num_classes)

    # ---- partition train across clients ----
    if PARTITION_MODE == "dirichlet":
        tr_part = DirichletPartitioner(y_train_all, NUM_CLIENTS, ALPHA, min_size=50, seed=SEED)
        get_tr_idx = lambda cid: tr_part.load_partition(cid)
    elif PARTITION_MODE == "iid_uniform":
        tr_part = IIDUniformPartitioner(num_samples=len(y_train_all), num_partitions=NUM_CLIENTS, seed=SEED)
        get_tr_idx = lambda cid: tr_part.load_partition(cid)
    elif PARTITION_MODE == "iid_stratified":
        tr_part = IIDStratifiedPartitioner(labels=y_train_all, num_partitions=NUM_CLIENTS, seed=SEED)
        get_tr_idx = lambda cid: tr_part.load_partition(cid)
    else:
        raise ValueError(f"Unknown PARTITION_MODE: {PARTITION_MODE}")

    # split VAL across clients (like your original)
    val_idx_all = np.arange(len(val_ds))
    val_splits = np.array_split(val_idx_all, NUM_CLIENTS)

    client_tr_loaders, client_va_loaders = [], []
    client_label_counts: List[np.ndarray] = []

    for cid in range(NUM_CLIENTS):
        tr_idx = get_tr_idx(cid)
        va_idx = val_splits[cid].tolist()
        loader_tr, loader_va = make_client_loaders(train_ds, val_ds, tr_idx, va_idx)
        client_tr_loaders.append(loader_tr)
        client_va_loaders.append(loader_va)

        ys = [int(train_ds[i][1] if np.ndim(train_ds[i][1]) == 0 else train_ds[i][1][0]) for i in tr_idx]
        counts = np.bincount(np.array(ys), minlength=num_classes)
        print(f"[Client {cid}] Train class distribution:", {i: int(c) for i, c in enumerate(counts) if c > 0})
        client_label_counts.append(counts)

    if PRINT_KL:
        global_counts = sum(client_label_counts)
        global_freq = global_counts / max(1, global_counts.sum())
        for cid, cnt in enumerate(client_label_counts):
            freq = cnt / max(1, cnt.sum())
            dkl = kl_div(freq, global_freq)
            print(f"[Client {cid}] KL(freq || global) = {dkl:.6f}")

    print(f"\nDataset: {DATASET} | embed_dim={EMBED_DIM} | num_classes={num_classes} | num_clients={NUM_CLIENTS} | "
          f"alpha={ALPHA} | rounds={NUM_ROUNDS} | partition={PARTITION_MODE}")
    print(f"Local LR={LR_EMB:g} | FIT_MODE={FIT_MODE} | head={'cosine' if USE_COSINE_HEAD else 'linear'}")

    # ---- Local training (CE) ----
    client_backbones = [MobileNetAttentionEmbedder(EMBED_DIM, BACKBONE_TRAINABLE, in_channels=3).to(DEVICE)
                        for _ in range(NUM_CLIENTS)]
    client_clfs = [CEClassifier(EMBED_DIM, num_classes).to(DEVICE) for _ in range(NUM_CLIENTS)]

    for cid in range(NUM_CLIENTS):
        print(f"\n--- Client {cid} (CE) ---")
        train_embedder_ce(client_backbones[cid], client_clfs[cid],
                          client_tr_loaders[cid], client_va_loaders[cid])

    global_backbone = MobileNetAttentionEmbedder(EMBED_DIM, BACKBONE_TRAINABLE, in_channels=3).to(DEVICE)
    with torch.no_grad():
        avg = {}
        keys = client_backbones[0].state_dict().keys()
        for k in keys:
            ts = [m.state_dict()[k] for m in client_backbones]
            avg[k] = torch.stack(ts).mean(0) if ts[0].dtype.is_floating_point else ts[0]
        global_backbone.load_state_dict(avg)

    # ---- Fit per-client class Gaussians on embeddings ----
    client_gauss: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for cid in range(NUM_CLIENTS):
        Zc, Yc = collect_embeddings(global_backbone, client_tr_loaders[cid], normalize=True)
        if FIT_MODE == "vi_diag":
            mu, L = fit_feature_gaussians_vi_diag(Zc, Yc, num_classes)
        elif FIT_MODE == "moments_full":
            mu, L = fit_feature_gaussians_fullrank(Zc, Yc, num_classes)
        else:
            raise ValueError("FIT_MODE must be 'vi_diag' or 'moments_full'")
        client_gauss.append((mu, L))

    # ---- Aggregate (server) ----
    agg: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, int]]] = {c: [] for c in range(num_classes)}
    for cid, (mu, L) in enumerate(client_gauss):
        cnts = client_label_counts[cid]
        for c in range(num_classes):
            # keep only if class exists on this client
            if int(cnts[c]) > 0:
                agg[c].append((mu[c].cpu(), L[c].cpu(), int(cnts[c])))

    # ---- Prior ----
    total_counts = np.zeros(num_classes, dtype=np.int64)
    for cnt in client_label_counts:
        total_counts += cnt
    emp = torch.from_numpy(total_counts.astype(np.float32))
    emp = emp / max(1, emp.sum())
    uniform = torch.full_like(emp, 1.0 / num_classes)
    prior = PRIOR_MIX_BETA * emp + (1 - PRIOR_MIX_BETA) * uniform
    prior = torch.clamp(prior, PRIOR_FLOOR, PRIOR_CAP)
    prior = prior / prior.sum()
    print("Mixed class prior:", {i: f"{prior[i].item():.4f}" for i in range(num_classes)})

    # ---- One-shot synthetic loader ----
    synth_loader = make_synth_loader_weighted(agg, prior, TOTAL_SYNTH, BATCH_SIZE)

    # ---- Train one-shot head on synthetic features ----
    head = make_head(num_classes).to(DEVICE)
    opt_head = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)

    # init from avg mu
    with torch.no_grad():
        avg_mu = torch.stack([m for m, _ in client_gauss]).mean(0)
        init_w = F.normalize(avg_mu, p=2, dim=1)
        if USE_COSINE_HEAD and hasattr(head, "weight") and head.weight.shape == init_w.shape:
            head.weight.copy_(init_w.to(DEVICE))
            if getattr(head, "bias", None) is not None:
                head.bias.zero_()
        if (not USE_COSINE_HEAD) and hasattr(head, "fc") and head.fc.weight.shape == init_w.shape:
            head.fc.weight.copy_(init_w.to(DEVICE))
            if head.fc.bias is not None:
                head.fc.bias.zero_()

    for ep in range(1, HEAD_EPOCHS + 1):
        head.train()
        total = 0
        run_loss = 0.0
        correct = 0
        pbar = tqdm(synth_loader, desc=f"Head(SYN) ep={ep}/{HEAD_EPOCHS}", ncols=0, leave=False)
        for z_cpu, y_cpu in pbar:
            z = z_cpu.to(DEVICE, non_blocking=True)
            y = y_cpu.to(DEVICE, non_blocking=True).view(-1).long()
            logits = head(z)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)

            opt_head.zero_grad(set_to_none=True)
            loss.backward()
            opt_head.step()

            bs = z.size(0)
            total += bs
            run_loss += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            pbar.set_postfix(
                avg_loss=f"{run_loss / max(1, total):.4f}",
                acc=f"{100.0 * correct / max(1, total):.2f}%",
            )
        print(f"[Head(SYN) ep={ep:02d}] loss={run_loss / max(1, total):.4f} acc={100.0 * correct / max(1, total):.2f}%")

    # ---- Evaluate one-shot head on real test (all classes) ----
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=N_WORKERS, pin_memory=PIN)
    head.eval()
    global_backbone.eval()
    tot = 0
    cor = 0
    run = 0.0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Head(EVAL real)", ncols=0, leave=False)
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).view(-1).long()
            z = F.normalize(global_backbone(x), p=2, dim=1)
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            bs = x.size(0)
            tot += bs
            run += loss.item() * bs
            cor += (logits.argmax(1) == y).sum().item()
            pbar.set_postfix(acc=f"{100*cor/max(1,tot):.2f}%")
    print(f"[One-shot Head EVAL] test_loss={run/max(1,tot):.4f} test_acc={100*cor/max(1,tot):.2f}%")

    # ---- Continual learning evaluation (VAL each task, FINAL TEST once) ----
    _ = run_continual_shared_head(
        global_backbone=global_backbone,
        agg=agg,
        prior=prior,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=num_classes,
        client_gauss=client_gauss,
    )

    print("Federated one-shot + continual evaluation done.")


if __name__ == "__main__":
    main()
