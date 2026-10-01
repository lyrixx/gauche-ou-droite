# Compare plusieurs classifieurs sur les embeddings EmbeddingGemma des exemples.
#
# Évaluation : validation croisée 5 blocs répétée 3 fois (mêmes découpages pour tous les modèles).
# Les hyperparamètres sont choisis à l'intérieur de chaque bloc d'entraînement (validation croisée
# imbriquée, 3 blocs), pour que le choix ne voie jamais le bloc de test.
#
# Données de la page « Banc d'essai des classifieurs » (media/bench).
# Lit .cache/tools/embeddings.json (tools/export-embeddings.mjs), écrit tools/data/bench.json.

import json
import time
import warnings
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent
data = json.loads((root / ".cache/tools/embeddings.json").read_text())
X = np.array(data["X"], dtype=np.float32)
y = np.array(data["y"])
n = len(y)
REPEATS, FOLDS = 3, 5
INNER = StratifiedKFold(3, shuffle=True, random_state=123)


def grid(estimator, params):
    return GridSearchCV(estimator, params, cv=INNER, scoring="accuracy", n_jobs=-1) if params else estimator


MODELS = [
    # (id, nom, famille, estimateur, grille)
    ("prod", "Régression logistique (actuelle, λ = 0,01)", "linéaire",
     make_pipeline(StandardScaler(), LogisticRegression(C=1 / (0.01 * n * 0.8), max_iter=5000)), None),
    ("logreg", "Régression logistique (C réglé)", "linéaire",
     make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
     {"logisticregression__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]}),
    ("pca_logreg", "ACP + régression logistique", "linéaire",
     make_pipeline(StandardScaler(), PCA(random_state=0), LogisticRegression(max_iter=5000)),
     {"pca__n_components": [32, 64, 128, 256], "logisticregression__C": [0.001, 0.01, 0.1]}),
    ("svm_lin", "SVM linéaire", "linéaire",
     make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, random_state=0)),
     {"svc__C": [0.0003, 0.001, 0.003, 0.01]}),
    ("svm_rbf", "SVM noyau RBF", "noyau",
     make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0)),
     {"svc__C": [1, 3, 10, 30]}),
    ("pca_svm_rbf", "ACP + SVM noyau RBF", "noyau",
     make_pipeline(StandardScaler(), PCA(random_state=0), SVC(kernel="rbf", probability=True, random_state=0)),
     {"pca__n_components": [32, 64, 128], "svc__C": [1, 3, 10]}),
    ("knn", "k plus proches voisins (cosinus)", "voisins",
     KNeighborsClassifier(metric="cosine", weights="distance"),
     {"n_neighbors": [5, 10, 20, 40]}),
    ("rf", "Random Forest", "arbres",
     RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1),
     {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]}),
    ("xgb", "XGBoost", "arbres",
     XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3, n_jobs=4, random_state=0, verbosity=0),
     {"max_depth": [2, 4]}),
    ("lgbm", "LightGBM", "arbres",
     LGBMClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, n_jobs=4, random_state=0, verbose=-1),
     {"num_leaves": [7, 15]}),
    ("catboost", "CatBoost", "arbres",
     CatBoostClassifier(iterations=500, learning_rate=0.05, depth=4, thread_count=-1, random_seed=0, verbose=0, allow_writing_files=False),
     None),
    ("pca_xgb", "ACP + XGBoost", "arbres",
     make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0),
                   XGBClassifier(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5, n_jobs=4, random_state=0, verbosity=0)),
     {"xgbclassifier__max_depth": [2, 4]}),
]

splits = [list(StratifiedKFold(FOLDS, shuffle=True, random_state=r).split(X, y)) for r in range(REPEATS)]


def evaluate(estimator, params):
    oof = np.zeros((REPEATS, n))
    chosen = []
    for r, folds in enumerate(splits):
        for train, test in folds:
            model = grid(estimator, params)
            model.fit(X[train], y[train])
            oof[r, test] = model.predict_proba(X[test])[:, 1]
            if params:
                chosen.append(json.dumps(model.best_params_, sort_keys=True))
    return oof, chosen


results = []
for mid, name, family, estimator, params in MODELS:
    t0 = time.time()
    oof, chosen = evaluate(estimator, params)
    p = np.clip(oof, 1e-6, 1 - 1e-6)
    acc = [accuracy_score(y, o >= 0.5) for o in oof]
    auc = [roc_auc_score(y, o) for o in oof]
    ll = [log_loss(y, o) for o in p]
    fpr, tpr, _ = roc_curve(y, oof[0])
    keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
    cm = confusion_matrix(y, oof[0] >= 0.5).tolist()  # lignes : vrai gauche/droite, colonnes : prédit gauche/droite
    best = max(set(chosen), key=chosen.count) if chosen else None
    results.append({
        "id": mid, "name": name, "family": family,
        "acc": round(float(np.mean(acc)), 4), "accStd": round(float(np.std(acc)), 4),
        "auc": round(float(np.mean(auc)), 4), "aucStd": round(float(np.std(auc)), 4),
        "logloss": round(float(np.mean(ll)), 4),
        "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep],
        "cm": cm, "best": json.loads(best) if best else None,
        "bestShare": round(chosen.count(best) / len(chosen), 2) if chosen else None,
        "seconds": round(time.time() - t0, 1),
    })
    print(f"{name:45s} acc {np.mean(acc):.3f} ± {np.std(acc):.3f}  AUC {np.mean(auc):.3f}  log-loss {np.mean(ll):.3f}  ({time.time() - t0:.0f} s) {best}", flush=True)

# Réduction de dimension : précision en fonction du nombre de composantes de l'ACP (C réglé à chaque fois).
sweep = []
for k in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
    est = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0), LogisticRegression(max_iter=5000))
    oof, _ = evaluate(est, {"logisticregression__C": [0.001, 0.01, 0.1, 1]})
    acc = [accuracy_score(y, o >= 0.5) for o in oof]
    sweep.append({"k": k, "acc": round(float(np.mean(acc)), 4), "accStd": round(float(np.std(acc)), 4)})
    print(f"ACP {k:4d} composantes : acc {np.mean(acc):.3f} ± {np.std(acc):.3f}", flush=True)

(root / "tools/data/bench.json").write_text(json.dumps({
    "n": int(n), "nDroite": int(y.sum()), "repeats": REPEATS, "folds": FOLDS,
    "models": results, "pcaSweep": sweep,
}, ensure_ascii=False))
print("tools/data/bench.json écrit.")
