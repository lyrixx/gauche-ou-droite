# Stabilité : courbes ROC sur le bloc d'entraînement et sur le bloc de test, pour chacun des
# 15 blocs (validation croisée 5 blocs × 3 répétitions, mêmes découpages que bench.py).
# Hyperparamètres fixés aux valeurs retenues par bench.py.
#
# Section « Train contre test » de media/bench. Écrit tools/data/stability.json.

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

root = Path(__file__).resolve().parent.parent
data = json.loads((root / ".cache/tools/embeddings.json").read_text())
X = np.array(data["X"], dtype=np.float32)
y = np.array(data["y"])
n = len(y)

MODELS = [
    ("svm_rbf", "SVM noyau RBF (C = 10)", lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))),
    ("logreg", "Régression logistique (C = 0,003)", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.003, max_iter=5000))),
    ("prod", "Régression logistique actuelle (λ = 0,01)", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1 / (0.01 * n * 0.8), max_iter=5000))),
]
GRID = np.linspace(0, 1, 101)
splits = [f for r in range(3) for f in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y)]


def curve(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    out = np.interp(GRID, fpr, tpr)
    out[0] = 0.0
    return out


results = []
for mid, name, make in MODELS:
    folds = []
    for train, test in splits:
        model = make().fit(X[train], y[train])
        s_tr, s_te = model.decision_function(X[train]), model.decision_function(X[test])
        folds.append({
            "aucTrain": roc_auc_score(y[train], s_tr), "aucTest": roc_auc_score(y[test], s_te),
            "accTrain": accuracy_score(y[train], s_tr > 0), "accTest": accuracy_score(y[test], s_te > 0),
            "rocTrain": curve(y[train], s_tr), "rocTest": curve(y[test], s_te),
        })
    rt, re = np.array([f["rocTrain"] for f in folds]), np.array([f["rocTest"] for f in folds])
    r4 = lambda v: round(float(v), 4)
    res = {
        "id": mid, "name": name,
        "folds": [{k: r4(f[k]) for k in ("aucTrain", "aucTest", "accTrain", "accTest")} for f in folds],
        "rocTestFolds": [[r4(v) for v in c] for c in re],
        "train": {"mean": [r4(v) for v in rt.mean(0)], "std": [r4(v) for v in rt.std(0)]},
        "test": {"mean": [r4(v) for v in re.mean(0)], "std": [r4(v) for v in re.std(0)]},
    }
    for k in ("aucTrain", "aucTest", "accTrain", "accTest"):
        vals = [f[k] for f in folds]
        res[k] = r4(np.mean(vals))
        res[k + "Std"] = r4(np.std(vals))
    results.append(res)
    print(f"{name:45s} AUC train {res['aucTrain']:.3f} ± {res['aucTrainStd']:.3f}  test {res['aucTest']:.3f} ± {res['aucTestStd']:.3f}  "
          f"| acc train {res['accTrain']:.3f}  test {res['accTest']:.3f} ± {res['accTestStd']:.3f}")

(root / "tools/data/stability.json").write_text(json.dumps({"grid": [round(float(g), 2) for g in GRID], "models": results}, ensure_ascii=False))
print("tools/data/stability.json écrit.")
