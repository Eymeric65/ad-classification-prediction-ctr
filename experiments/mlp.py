"""Torch MLP that predicts the 8 PCA curve-scores, conforming to harness's
model_factory contract (fit / predict).

Design notes
------------
* Preprocessing reuses the repo's idea: one-hot the (low-cardinality) categoricals,
  median-impute + standardize the numerics. We build it with sklearn ColumnTransformer
  so the feature space is identical to what LightGBM sees.
* Target handling: PCA components are ORTHONORMAL, so ||score_pred - score_true||^2
  equals the squared error of the reconstructed (log) curve. Hence the principled
  training loss is *unweighted raw-score* MSE. For optimisation stability we standardise
  each target component to unit variance and re-weight its loss by var_c, which is
  algebraically identical to raw-score MSE but keeps the network outputs O(1).
* Per-sample weights (tail boost * recency) are passed straight through from the harness.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_preproc(cat, num):
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
    ])


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLP:
    """harness model_factory contract: fit(X_df, Y, sample_weight) / predict(X_df)."""
    def __init__(self, cat, num, hidden=(256, 128), dropout=0.2, lr=2e-3, wd=1e-5,
                 epochs=60, batch=1024, seed=0, val_frac=0.1, patience=8, verbose=False):
        self.cat, self.num = cat, num
        self.hidden, self.dropout, self.lr, self.wd = hidden, dropout, lr, wd
        self.epochs, self.batch, self.seed = epochs, batch, seed
        self.val_frac, self.patience, self.verbose = val_frac, patience, verbose

    def fit(self, Xdf, Y, sample_weight=None):
        _seed(self.seed)
        self.pre = make_preproc(self.cat, self.num)
        Xt = self.pre.fit_transform(Xdf).astype(np.float32)
        Y = np.asarray(Y, np.float32)
        n, d_in = Xt.shape
        d_out = Y.shape[1]

        self.y_mean = Y.mean(0)
        self.y_std = Y.std(0) + 1e-8
        Ys = (Y - self.y_mean) / self.y_std
        # loss weight per component = var_c  => standardized MSE == raw-score MSE
        self.comp_w = torch.tensor(self.y_std ** 2 / np.mean(self.y_std ** 2),
                                   dtype=torch.float32, device=DEVICE)

        sw = np.ones(n, np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        sw = sw * (n / sw.sum())

        # internal validation split for early stopping
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        n_val = int(n * self.val_frac)
        vi, ti = perm[:n_val], perm[n_val:]

        Xt_t = torch.tensor(Xt, device=DEVICE)
        Ys_t = torch.tensor(Ys, device=DEVICE)
        sw_t = torch.tensor(sw, device=DEVICE)
        ti_t = torch.tensor(ti, device=DEVICE)
        vi_t = torch.tensor(vi, device=DEVICE)

        self.model = MLP(d_in, d_out, self.hidden, self.dropout).to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            self.model.train()
            order = ti_t[torch.randperm(len(ti_t), device=DEVICE)]
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                opt.zero_grad()
                pred = self.model(Xt_t[b])
                err = (pred - Ys_t[b]) ** 2 * self.comp_w
                loss = (err.sum(1) * sw_t[b]).mean()
                loss.backward()
                opt.step()
            sched.step()
            # validation (unweighted raw-score MSE — the honest proxy)
            self.model.eval()
            with torch.no_grad():
                vp = self.model(Xt_t[vi_t])
                verr = (((vp - Ys_t[vi_t]) ** 2) * self.comp_w).sum(1).mean().item()
            if verr < best_val - 1e-6:
                best_val, best_state, bad = verr, {k: v.detach().clone() for k, v in self.model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose:
                print(f"    ep{ep:02d} val={verr:.5f} best={best_val:.5f}")
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, Xdf):
        Xt = self.pre.transform(Xdf).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(Xt, device=DEVICE)).cpu().numpy()
        return out * self.y_std + self.y_mean


def mlp_factory(**kw):
    return lambda cat, num: TorchMLP(cat, num, **kw)
