# De gauche ou de droite ?

**👉 https://lyrixx.github.io/gauche-ou-droite/**

Vous tapez n'importe quoi (« la raclette », « le vélo », « Chateaubriand »…) et une IA vous dit si c'est de gauche ou de droite. C'est un clone de [degaucheoudedroite.delemazure.fr](https://degaucheoudedroite.delemazure.fr/) de [Théo Delemazure](https://twitter.com/DelemazureTheo), avec une différence : **le modèle tourne entièrement dans votre navigateur**. Il n'y a pas de serveur ni d'API, et rien de ce que vous tapez ne quitte votre machine.

L'IA a été entraînée sur des clichés. À ne pas prendre au sérieux.

## En bref

**Comment ça marche.** Un petit modèle de langage ([EmbeddingGemma](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX), environ 200 Mo, téléchargé une fois puis gardé en cache) transforme le texte en une liste de nombres qui représente son sens. Un classifieur, entraîné sur un millier d'exemples étiquetés à la main dans [`data/exemples.csv`](data/exemples.csv), en déduit une probabilité « gauche » ou « droite ». Si le texte fait partie des exemples, on répond directement.

**Installer en local.** Il faut [Node.js](https://nodejs.org/) 24, [Castor](https://castor.jolicode.com/) et PHP (pour `castor serve`) :

```bash
castor install   # installe les dépendances
castor train     # entraîne le classifieur (télécharge le modèle la première fois)
castor serve     # ouvre le site sur http://localhost:8000
```

**Les autres tâches**

```bash
castor predict "le vélo" "facho"   # teste depuis le terminal
castor build                       # install + train, c'est ce que lance la CI
castor media:build                 # régénère les pages de media/ (voir tools/README.md)
```

**Améliorer les réponses.** Ajoutez ou corrigez des lignes dans [`data/exemples.csv`](data/exemples.csv), relancez `castor train`, rechargez la page.

**Déploiement.** Chaque push sur `main` lance [la CI](.github/workflows/deploy.yml). Elle réentraîne le classifieur avec `castor build` et publie le dossier `site/` sur GitHub Pages, avec les pages de `media/` sous [`/media/`](https://lyrixx.github.io/gauche-ou-droite/media/).

**Les coulisses.** [`media/`](https://lyrixx.github.io/gauche-ou-droite/media/) rassemble les slides du lightning talk, une carte des exemples (le nuage) et un banc d'essai de douze classifieurs. Ces pages sont générées par les scripts de [`tools/`](tools/README.md).

---

## Dans le détail

Cette partie raconte comment le projet est construit, ce qu'on a essayé, et pourquoi on a fait ces choix.

### Le cahier des charges

- **Un site statique sur GitHub Pages**, donc sans serveur pour faire tourner un modèle.
- **Le modèle tourne dans le navigateur du visiteur**, sans API payante et sans clé à cacher.
- **Rien à réinventer côté UI** : on tape un sujet, on obtient « de gauche » ou « de droite » avec un score.

Au départ, on avait pensé à [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), un modèle de TypeSafe AI sorti en septembre 2026. Il ne génère pas de texte : il renvoie une décision et un score de confiance, exactement ce qu'il nous faut. Mais c'est un modèle propriétaire, accessible seulement par API, qui ne peut pas tourner dans un navigateur. On a gardé l'idée d'un petit modèle qui décide au lieu de discuter, et on l'a construite avec des briques open source.

### Pourquoi pas un LLM dans le navigateur ?

Trois approches étaient possibles :

| Approche | Téléchargement | Matériel | Verdict |
|---|---|---|---|
| **Embeddings + classifieur** | 120 à 200 Mo | CPU | ✅ retenue |
| Classification zero-shot (NLI) | environ 300 Mo | CPU | moins fiable, rien à corriger soi-même |
| Petit LLM (WebLLM, Qwen, Llama 1B) | 1 Go ou plus | WebGPU obligatoire | trop lourd, et médiocre en français à cette taille |

La différence tient au volume de calcul. Un modèle d'embeddings fait **un seul passage** sur la phrase. Un LLM génère sa réponse mot par mot, avec un passage complet du modèle pour chaque mot.

### Comment marche la V1

```
"le vélo" ──► modèle d'embeddings ──► vecteur de 384 nombres ──► régression logistique ──► 72 % gauche
              (le "sens" du texte)                               (poids dans un JSON)
```

La V1 repose sur deux briques.

**1. Le modèle d'embeddings.** Il transforme n'importe quel texte en vecteur. Deux textes de sens proche donnent des vecteurs proches. Il ne sait rien de la politique : il sait seulement représenter le sens. En V1, on utilise [`multilingual-e5-small`](https://huggingface.co/Xenova/multilingual-e5-small) :
- 118 Mo en version quantifiée (q8) ;
- des vecteurs de 384 nombres ;
- environ 10 ms par phrase sur CPU ;
- il tourne grâce à [transformers.js](https://huggingface.co/docs/transformers.js), qui s'appuie sur onnxruntime en WebAssembly.

**2. La régression logistique.** C'est la partie qui décide. Elle apprend quelle direction dans l'espace des vecteurs correspond à « gauche » et laquelle à « droite ». Un sujet absent des exemples tombe quelque part dans cet espace, et la régression dit de quel côté. Le résultat de l'entraînement est une liste de 384 poids et un biais, enregistrés dans `site/classifier.json` (quelques Ko). L'entraînement est écrit à la main dans [`scripts/train.mjs`](scripts/train.mjs), sans dépendance :
- descente de gradient ;
- régularisation L2 (λ = 0,01) ;
- 2 000 itérations ;
- variables standardisées au préalable.

**Le même modèle partout.** L'entraînement tourne dans Node, la prédiction dans le navigateur. Les deux doivent produire exactement les mêmes vecteurs. Le modèle et sa configuration sont donc définis une seule fois, dans [`site/config.js`](site/config.js), et le calcul de la prédiction dans [`site/classifier.js`](site/classifier.js). Ces deux fichiers sont importés à la fois par les scripts Node et par le site.

**Mesurer honnêtement.** Tester le classifieur sur ses propres exemples ne dit rien : il les a appris. `castor train` fait donc une **validation croisée** :
1. on cache 1/5 des exemples ;
2. on entraîne sur les 4/5 restants ;
3. on mesure sur la partie cachée ;
4. on recommence 5 fois, pour que chaque exemple soit caché une fois.

Le script affiche la précision obtenue et la liste des exemples mal classés. Cette liste est une bonne source d'idées d'exemples à ajouter.

#### Les données

Tout repose sur [`data/exemples.csv`](data/exemples.csv), une ligne par exemple : `camp;texte`. Le jeu a évolué comme suit :

| Étape | Exemples | Précision (validation croisée) |
|---|---|---|
| Premier jet : clichés de société | 253 | 66-68 % |
| Ajout des étiquettes politiques, partis, personnalités, lieux, métiers, marques… | 952 | 68,7 % |
| Réglage de la régularisation | 952 | 69,7 % |
| Ajout des militants et électeurs de chaque camp | 972 | environ 69 % |

On a quadruplé les données pour gagner à peine 3 points. **Avec un modèle d'embeddings figé, ajouter des exemples corrige des cas précis, mais relève peu la précision globale.** Le goulot d'étranglement, c'est le modèle.

Les ajouts ont quand même corrigé des cas flagrants. Avant, « facho » sortait **de gauche à 84 %** et « Marine Le Pen » de gauche à 86 %. Le modèle voyait « un mot politique » sans en saisir le bord.

#### La table de correspondance

Si un texte fait partie des exemples, il doit toujours obtenir la bonne réponse. Le classifieur, lui, ne réussissait même pas à apprendre tous ses exemples : environ 92 % seulement. `classifier.json` contient donc aussi une **table de correspondance** entre texte normalisé et camp, consultée avant le modèle. La page affiche d'où vient la réponse : « tirée des exemples » ou « devinée par le modèle ».

La normalisation (`normalize()` dans [`site/classifier.js`](site/classifier.js)) a eu ses bugs :
- « jean marie le pen » ne trouvait pas « Jean-Marie Le Pen ». Les traits d'union et les accents sont maintenant ignorés.
- La suppression de l'article ne vérifiait pas que c'était un mot entier : « Lénine » devenait « nine ».

Cette normalisation ne sert **qu'à la table**. Le modèle, lui, reçoit le texte brut. Il a appris le français avec ses accents, et lui enlever les accents dégraderait ses vecteurs. C'est pour ça que « développeur » et « developpeur » peuvent recevoir des scores différents quand le mot n'est pas dans les exemples.

### De la V1 à la V2 : le banc d'essai

La V1 plafonnait autour de 69-70 %. Avant de sortir l'artillerie lourde, on a cherché ce qui compte vraiment. Le banc d'essai :
- **11 modèles d'embeddings**, dont 7 assez légers pour un navigateur et 4 trop lourds, gardés comme références ;
- **jusqu'à 8 formulations** du texte d'entrée ;
- **3 méthodes de classification** : une régression ridge (proche de la régression logistique, mais calculable directement, donc plus rapide pour balayer la régularisation), les 10 plus proches voisins, et le zero-shot ;
- **des combinaisons** de formulations et de modèles.

Pour que des écarts d'un point veuillent dire quelque chose, toutes les configurations sont mesurées sur **les mêmes découpages** : validation croisée 5 blocs, répétée 3 fois avec des graines fixes. Les 972 exemples ont été utilisés à chaque fois.

#### Les modèles

| Modèle | Taille (quantifié) | Vitesse CPU (Node) | Précision |
|---|---|---|---|
| paraphrase-multilingual-MiniLM | 118 Mo | 4 ms | 67 % |
| distiluse-base-multilingual | 135 Mo | 3 ms | 69,5 % |
| multilingual-e5-base | 279 Mo | 6 ms | 70 % |
| **multilingual-e5-small (V1)** | **118 Mo** | **4 ms** | **70,6 %** |
| paraphrase-multilingual-mpnet | 279 Mo | 6 ms | 71,5 % |
| bge-m3 | 570 Mo | 17 ms | 73 % |
| multilingual-e5-large | 562 Mo | 17 ms | 76 % |
| Snowflake Arctic Embed v2 | 311 Mo | 15 ms | 77 % |
| **EmbeddingGemma 300M (V2)** | **197 Mo** | **15-40 ms** | **81-82 %** |
| Qwen3-Embedding 0.6B | 614 Mo | 60-95 ms | 89 % |

#### Ce qu'on en retient

1. **Le choix du modèle compte plus que tout le reste.** EmbeddingGemma, sorti par Google en 2025, gagne 11 points sur la V1 pour 80 Mo de plus. Les modèles plus gros mais plus anciens (e5-large, bge-m3) font moins bien.
2. **Reformuler le texte ne sert presque à rien sur les modèles classiques.** On a essayé « Opinion politique sur : X », « X : est-ce de gauche ou de droite ? », « Un électeur qui aime X », « En politique française, X », etc. L'écart reste d'environ 1 point, dans le bruit de mesure.
3. **Sauf sur Qwen3, où la formulation change tout.** Avec « Les gens qui aiment X votent pour », on passe de 80 à 89 %. Qwen3 est construit à partir d'un LLM, et son vecteur est calculé sur le **dernier mot** de la phrase. Si la phrase se termine par « votent pour… », ce vecteur anticipe la suite, c'est-à-dire un parti, et encode directement l'orientation politique. Les formulations proches donnent 87-88 %, donc ce n'est pas un coup de chance.
4. **Ce qui ne sert à rien :**
   - combiner plusieurs formulations ou plusieurs modèles (Qwen3 + Gemma : 88,8 %, soit 800 Mo pour ne rien gagner) ;
   - la version q8 de Gemma par rapport à la q4 ;
   - les plus proches voisins à la place de la régression ;
   - le **zero-shot**, qui consiste à comparer le texte à « de gauche » et « de droite » sans apprentissage. Il donne 48 à 60 %, pas mieux que le hasard. Et « de droite » a aussi un sens spatial, par opposition à « de gauche ».

Ces chiffres sont légèrement optimistes, d'un demi-point environ, parce que la régularisation est choisie sur les mêmes découpages que ceux qui servent à mesurer. Dans le vrai pipeline, EmbeddingGemma donne **79 %**.

### La V2

La V2 garde exactement la même architecture, avec un nouveau modèle et quelques ajustements.

**Le modèle** : [EmbeddingGemma 300M](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX), quantifié en q4.
- Vecteurs de 768 nombres.
- Le texte est envoyé sous la forme `task: classification | query: En politique française, X`. Le préfixe est la consigne de tâche prévue par le modèle, et le contexte « politique française » fait gagner un peu.
- Le chargement passe par `AutoTokenizer` et `AutoModel` au lieu du `pipeline` de transformers.js, parce que Gemma renvoie directement un vecteur de phrase (`sentence_embedding`).

**Le piège de la quantification.** La version q4 standard fonctionne dans Node, mais pas dans le navigateur. Elle utilise une opération (`GatherBlockQuantized`) que le moteur WebAssembly d'onnxruntime ne sait pas exécuter. On utilise donc la variante `model_no_gather_q4`, prévue pour le web, **pour l'entraînement comme pour le site**, afin que les vecteurs soient identiques des deux côtés.

**Le Web Worker.** Avec Gemma, un calcul prend environ 1 seconde dans le navigateur, contre environ 10 ms avec e5-small. Deux raisons : le modèle est plus gros, et GitHub Pages ne permet pas d'envoyer les en-têtes HTTP (COOP/COEP) qui autorisent le calcul sur plusieurs cœurs. Pendant ce temps, la page gelait. L'option `env.backends.onnx.wasm.proxy = true` déplace le calcul dans un Web Worker, et la page reste à 60 images/s.

**L'interface pendant les chargements.**
- Le formulaire est verrouillé tant que le modèle n'est pas prêt. Sans ce verrou, un envoi rechargeait la page et relançait le téléchargement de zéro.
- La progression du téléchargement additionne tous les fichiers du modèle.
- Un message « Préparation du modèle… » couvre la seconde qui suit le téléchargement.
- Pendant chaque calcul, l'aiguille de l'hémicycle hésite entre la gauche et la droite.

### Et après ?

- **Affiner le modèle ([SetFit](https://github.com/huggingface/setfit)).** Aujourd'hui, seule la régression apprend, et le modèle d'embeddings reste figé. SetFit réentraîne aussi le modèle pour rapprocher les exemples d'un même camp et éloigner les camps opposés. Le gain habituel est de 10 à 20 points, avec la même taille et la même vitesse dans le navigateur. Il faudrait :
  - ajouter une étape en Python ;
  - publier le modèle sur Hugging Face Hub, puisque GitHub refuse les fichiers de plus de 100 Mo.
- **Beaucoup plus de données**, étiquetées par un gros LLM puis relues. Seules, elles aident peu, on l'a vu. Combinées à SetFit, c'est là qu'elles paient.
- **Un « mode précis » avec Qwen3** (89 %), en option, pour qui accepte de télécharger 600 Mo.
- **WebGPU**, quand le navigateur le propose, pour accélérer les calculs.

### Structure du projet

```
data/exemples.csv          les exemples étiquetés : c'est ici qu'on améliore les réponses
scripts/train.mjs          entraînement + validation croisée → site/classifier.json
scripts/predict.mjs        prédictions depuis le terminal
site/                      le site statique publié sur GitHub Pages
  index.html, app.js       la page et son interface
  config.js                le modèle utilisé (partagé Node / navigateur)
  classifier.js            normalisation, table, prédiction (partagé Node / navigateur)
  classifier.json          généré par l'entraînement, non versionné
media/                     les pages publiées sous /media/ : slides, nuage des exemples, banc d'essai
tools/                     les scripts qui génèrent media/ (voir tools/README.md)
castor.php                 les tâches install, train, serve, predict, build, media:*
.github/workflows/         déploiement GitHub Pages
```
