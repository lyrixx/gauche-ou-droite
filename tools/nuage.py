# Données de la page « Le nuage des exemples » (media/nuage) :
# - projection des embeddings en 2D, par t-SNE et par ACP ;
# - pour chaque exemple, la probabilité « droite » donnée par une régression logistique qui ne l'a
#   pas vu (validation croisée 5 blocs, même régularisation que scripts/train.mjs) ;
# - les thèmes : un k-means sur la carte t-SNE, nommés à la main d'après leur contenu.
#
# Lit .cache/tools/embeddings.json (tools/export-embeddings.mjs), écrit tools/data/points.json.

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent
data = json.loads((root / ".cache/tools/embeddings.json").read_text())
X = np.array(data["X"], dtype=np.float32)
y = np.array(data["y"])

# train.mjs minimise mean(log-loss) + λ/2·‖w‖² avec λ = 0,01 ; en sklearn, C = 1 / (λ·n).
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1 / (0.01 * len(y)), max_iter=5000))
p_cv = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
accuracy = float(((p_cv >= 0.5) == y).mean())
print(f"Précision en validation croisée : {accuracy:.1%}")

tsne = TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", random_state=42).fit_transform(X)
pca = PCA(n_components=2, random_state=0)
pca_xy = pca.fit_transform(X)
print(f"Variance expliquée par les 2 premiers axes de l'ACP : {pca.explained_variance_ratio_.sum():.1%}")

# Les noms correspondent aux groupes obtenus avec ces graines et ces données. Si les exemples
# changent, relancer, lire le contenu de chaque groupe (affiché ci-dessous) et renommer.
THEMES = {
    0: "École, santé, société", 1: "Lieux", 2: "Métiers et figures sociales", 3: "Écologie, énergie, animaux",
    4: "Loisirs, sports, transports", 5: "Personnalités", 6: "Médias, culture, organisations",
    7: "Pratiques et mobilisations", 8: "Nourriture et boissons", 9: "Partis et étiquettes politiques",
    10: "Personnalités", 11: "Valeurs et grands principes", 12: "Vêtements et accessoires",
    13: "Fiscalité, économie, immigration",
}
# Sur les coordonnées arrondies telles qu'enregistrées, pour que la numérotation soit reproductible.
clusters = KMeans(len(THEMES), n_init=10, random_state=0).fit_predict(np.round(tsne.astype(np.float64), 2))
errors = (p_cv >= 0.5) != y
themes = []
for name in dict.fromkeys(THEMES.values()):
    idx = np.where([THEMES[c] == name for c in clusters])[0]
    centre = tsne[idx].mean(0)
    themes.append({
        "name": name, "n": int(len(idx)), "droite": round(float(y[idx].mean()), 3), "err": round(float(errors[idx].mean()), 3),
        "cx": round(float(centre[0]), 2), "cy": round(float(centre[1]), 2),
    })
    sample = np.random.default_rng(0).choice(idx, min(10, len(idx)), replace=False)
    print(f"{name} ({len(idx)}) : " + " | ".join(data["texts"][i] for i in sample))

points = [
    {"t": t, "y": int(c), "p": round(float(p), 3), "th": THEMES[k],
     "tx": round(float(a[0]), 2), "ty": round(float(a[1]), 2), "px": round(float(b[0]), 4), "py": round(float(b[1]), 4)}
    for t, c, p, k, a, b in zip(data["texts"], y, p_cv, clusters, tsne, pca_xy)
]
(root / "tools/data/points.json").write_text(json.dumps({
    "accuracy": round(accuracy, 4),
    "pcaVariance": round(float(pca.explained_variance_ratio_.sum()), 4),
    "points": points,
    "themes": themes,
}, ensure_ascii=False))

# Chiffres cités dans le texte de la page : voisins d'exemples précis, et part des 10 plus
# proches voisins (dans les 768 dimensions) qui ont la même étiquette.
S = X @ X.T
for q in ["la Rolex", "les vêtements de seconde main", "la loi anti-squat"]:
    i = data["texts"].index(q)
    nn = np.argsort(-S[i])[1:9]
    print(f"{q} → " + " | ".join(f"{data['texts'][j]} ({'D' if y[j] else 'G'})" for j in nn))
same = [np.mean(y[np.argsort(-S[i])[1:11]] == y[i]) for i in range(len(y))]
print(f"Part des 10 plus proches voisins de même étiquette : {np.mean(same):.1%}")
print(f"{len(points)} points écrits dans tools/data/points.json.")
