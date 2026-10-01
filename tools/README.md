# Outils

Les scripts qui produisent les pages de [`media/`](../media) : le nuage des exemples et le banc d'essai des classifieurs. Ils ne servent ni à l'entraînement ni au site.

Il faut Node.js (comme pour le reste du projet) et [uv](https://docs.astral.sh/uv/), qui installe à la volée les dépendances Python (scikit-learn, XGBoost, LightGBM, CatBoost).

```bash
castor media:data    # recalcule tout : embeddings, nuage, banc d'essai (une dizaine de minutes)
castor media:build   # régénère les pages de media/ à partir de tools/data/ (instantané)
```

## Les étapes

| Script | Lit | Écrit |
|---|---|---|
| `export-embeddings.mjs` | `data/exemples.csv` | `.cache/tools/embeddings.json` et `.csv` |
| `nuage.py` | les embeddings | `tools/data/points.json` |
| `bench.py` | les embeddings | `tools/data/bench.json` |
| `stability.py` | les embeddings | `tools/data/stability.json` |
| `build-media.py` | `tools/templates/`, `tools/data/`, `media/slides/src/` | `media/*/index.html` |

- **Les embeddings** sont calculés avec le même modèle et le même préfixe que `scripts/train.mjs`. Le CSV (`texte, camp, y, e0…e767`) permet d'explorer les données avec d'autres outils. Il n'est pas versionné (8 Mo) : `castor media:data` le recrée.
- **Les résultats** (`tools/data/*.json`) sont versionnés : régénérer les pages ne demande pas de relancer les calculs.
- **Les textes d'analyse** des pages (conclusions, chiffres cités) sont écrits à la main dans `tools/templates/`. Si les exemples changent, il faut les relire après `castor media:data`. `nuage.py` affiche le contenu de chaque thème et les chiffres cités dans le texte. Les noms des thèmes, donnés à la main, sont dans `nuage.py`.
- **Les slides** sont dans `media/slides/src/` : `deck.json` donne leur ordre, et chaque slide est un fichier HTML à styles en ligne, sur une toile de 1920×1080. `build-media.py` les assemble dans une visionneuse statique.
