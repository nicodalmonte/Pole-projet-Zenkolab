# src/generalization — Vue d'ensemble

Ce module regroupe toutes les expériences liées à la **généralisation inter-domaines** pour la détection du glaucome. L'idée centrale est de quantifier puis réduire le domain gap entre datasets d'entraînement et de test, en utilisant principalement un backbone DINOv3-Small figé.

---

## Fil conducteur des expériences

```
1. Comprendre le domain gap
   dataset_clustering.py
        ↓ embeddings + distances centroïdes
   analyze_distance_relation.py  →  analyze_deep_features.py
        ↓ quelles features expliquent les clusters ?
   normalization_analysis.py  →  standardization_test.py  →  test_deep_normalization.py
        ↓ peut-on réduire le gap par préprocessing ?
   test_embedding_synthesis.py
        ↓ peut-on le simuler en espace de features ?

2. Entraîner des modèles généralisables
   train_cluster_generalization.py  (conditions A–D, paper principal)
        ↓ sauvegarder les checkpoints
   eval_generalization.py           (évaluer A–D sur RIMONE/Fundus/AIRROGS)
   dann_ablation.py                 (DANN, conditions domain-adversarial)
   train_cluster_c_dataset_split.py (généralisation vers Cluster C étendu)

3. Visualiser / projeter
   project_new_datasets.py
```

---

## Fichiers

### `dataset_clustering.py`

**Ce que ça fait :** Extrait les embeddings de tous les datasets via un backbone timm (par défaut DINOv3-Small), puis génère un UMAP 2D et une matrice de distances centroïdes par dataset.

**Pourquoi :** Point de départ de toute l'analyse. Le UMAP révèle que les datasets se regroupent en 3 clusters visuels distincts, dont la distance au cluster de RIMONE prédit la performance de généralisation.

**Commande type :**
```bash
uv run python -m src.generalization.dataset_clustering --backbone vit_small_patch16_dinov3.lvd1689m
```

**Outputs — `figures/clustering/<backbone_slug>/`**
| Fichier | Contenu |
|---|---|
| `features.npy` | Matrice d'embeddings bruts (N × D) |
| `labels.json` | Label dataset pour chaque embedding |
| `centroid_distances.json` | Matrice de distances L2 inter-centroïdes |
| `centroid_heatmap.png` | Heatmap de la matrice de distances |
| `umap_all_datasets.png` | UMAP coloré par dataset |
| `umap_glaucoma.png` | UMAP coloré glaucome/sain |
| `umap_embedding.npy` | Coordonnées UMAP 2D |
| `pair_plots/` | Scatter plots par paire de datasets |

---

### `analyze_distance_relation.py`

**Ce que ça fait :** Double analyse sur les sorties de `dataset_clustering.py` :
1. **Qu'est-ce qui explique les clusters DINO ?** — Extrait des features images simples (résolution, ratio bords noirs, ratio luminosité centre/bord, saturation couleur…), calcule la matrice de distances dans cet espace, et fait un test de Mantel pour voir si elle corrèle avec la matrice DINO.
2. **La distance DINO prédit-elle l'AUC ?** — Fit plusieurs régressions (OLS, Huber, log-distance, quadratique) entre distance DINO et AUC zero-shot, en détectant les outliers par résidus Huber.

**Inputs requis :**
- `figures/clustering/vit_small_patch16_dinov3_lvd1689m/centroid_distances.json`
- `figures/distance_generalization/results_jraigs.json` (produit par `train_cluster_generalization.py`)

**Outputs — `figures/distance_analysis/`**
| Fichier | Contenu |
|---|---|
| `cluster_explanation.json` | Corrélation Mantel par feature (ρ, p-value) |
| `feature_vs_dino.png` | Scatter feature-distance vs DINO-distance, par feature |
| `per_feature_bars.png` | Bar chart des ρ de Mantel par feature |
| `robust_distance_auc.png` | Scatter distance DINO → AUC avec toutes les régressions |

---

### `analyze_deep_features.py`

**Ce que ça fait :** Round 2 de l'analyse Mantel, cette fois avec des features "profondes" non-cosmétiques : ratio basses fréquences FFT, profil fréquentiel radial, histogramme LBP (texture locale). Compare avec les features cosmétiques du round 1.

**Pourquoi :** `analyze_distance_relation.py` a montré que les features cosmétiques (bords noirs, saturation) corrèlent avec le clustering DINO, mais `test_feature_normalization.py` a montré que les neutraliser ne réduit pas les clusters. Ce script cherche les vraies causes : est-ce que des features structurales plus profondes corrèlent encore plus ?

**Input requis :**
- `figures/clustering/.../centroid_distances.json`
- `figures/distance_analysis/cluster_explanation.json` (pour comparaison)

**Outputs — `figures/deep_features/`**
| Fichier | Contenu |
|---|---|
| `deep_features.json` | Corrélations Mantel des features profondes |
| `deep_feature_bars.png` | Bar chart comparatif features profondes vs cosmétiques |

---

### `normalization_analysis.py`

**Ce que ça fait :** Compare l'impact de 7 stratégies de préprocessing sur la géométrie UMAP : `raw`, `imagenet`, `grayscale`, `green_ch`, `clahe`, `disc_crop`, `ben_graham`. Pour chaque stratégie, recalcule les embeddings et le UMAP, puis compare les matrices de distances centroïdes via corrélation de Spearman.

**Outputs — `figures/clustering/normalization_analysis/`**
| Fichier | Contenu |
|---|---|
| `umap_<strategy>.png` | UMAP coloré par dataset pour chaque stratégie |
| `umap_rgb.png` | UMAP coloré par couleur RGB moyenne réelle |
| `<strategy>/features.npy` | Embeddings de la stratégie |
| `centroid_distances_<s>.json` | Matrice de distances pour la stratégie |
| `spearman_matrix.png/.json` | Corrélation rang entre stratégies |

---

### `standardization_test.py`

**Ce que ça fait :** Ablation cumulatif de chaque étape de standardisation d'image (FOV crop → normalisation illumination → CLAHE → masque circulaire → z-score). Pour chaque configuration, mesure deux choses : (1) la précision d'un classifieur de domaine linéaire (5-fold CV), et (2) la distance centroïde moyenne. L'objectif est que le domaine devienne *indiscernable* (précision proche du hasard).

**Outputs — `figures/standardization_test/`**
| Fichier | Contenu |
|---|---|
| `ablation_summary.json` | Balanced accuracy + centroid dist par étape |
| `ablation_curves.png` | Courbe d'accuracy et de distance selon les étapes |
| `<step>/umap.png` | UMAP pour chaque configuration |

---

### `test_deep_normalization.py`

**Ce que ça fait :** Round 2 de neutralisation des features, ciblant les correlats profonds identifiés par `analyze_deep_features.py` (FFT low-freq, LBP, texture). Teste 4 stratégies : `raw` (contrôle), `clahe`, `hist_equalize`, `disc_crop`. Pour chaque stratégie, recalcule la matrice de distances centroïdes et un résumé pairwise.

**Outputs — `figures/deep_normalization/`**
| Fichier | Contenu |
|---|---|
| `centroid_<strategy>.json` | Matrice de distances centroïdes par stratégie |
| `centroid_heatmap_<strategy>.png` | Heatmap correspondante |
| `pairwise_distance_summary.png` | Comparaison toutes stratégies côte à côte |
| `summary.json` | Distances moyennes inter/intra-cluster par stratégie |

---

### `test_embedding_synthesis.py`

**Ce que ça fait :** Teste 5 méthodes pour simuler les embeddings d'un domaine cible (RIMONE) à partir de sources (ACRIMA, Harvard), **entièrement en espace de features** sans recharger les images :
1. Baseline brut
2. Interpolation linéaire vers le centroïde RIMONE (sweep α)
3. AdaIN (transfert de moyenne + std)
4. MixStyle (mix convexe de stats source/cible)
5. ZCA (transfert de covariance dans l'espace PCA-128)

Mesure pour chaque méthode : distance centroïde source→RIMONE, séparabilité glaucome/sain, diversité intra.

**Input requis :** `figures/clustering/features.npy` + `labels.json` (produit par `dataset_clustering.py`)

**Output :** Console uniquement — tableau comparatif de métriques.

---

### `train_cluster_generalization.py`

**Ce que ça fait :** Expérience principale de généralisation. Entraîne DINOv3-Small (2 phases : backbone figé puis débloqué) sur 5 combinaisons de clusters, toujours avec un budget de 2000 images équilibrées 50/50 :

| Condition | Données d'entraînement | Question |
|---|---|---|
| `single_A` | JRAIGS | Cluster A seul (loin de RIMONE) |
| `single_B` | ORIGA + LAG | Cluster B seul |
| `single_C` | ACRIMA + Harvard | Cluster C (proche de RIMONE) |
| `multi_AB` | JRAIGS + ORIGA/LAG | Diversité 2 clusters, sans cluster proche |
| `multi_ABC` | JRAIGS + ORIGA/LAG + ACRIMA/Harvard | Tous les clusters |

Test sets : RIMONE (unifié), Fundus (unifié), AIRROGS.

**Outputs**
| Chemin | Contenu |
|---|---|
| `checkpoints/<condition>/` | Checkpoints phase 1 et 2 (best val AUC) |
| `figures/cluster_generalization/results.json` | AUC + métriques par condition × test set |
| `figures/cluster_generalization/*.png` | Courbes d'entraînement + barres de métriques |

---

### `eval_generalization.py`

**Ce que ça fait :** Charge les checkpoints sauvegardés des 4 conditions principales (A-D, définis en dur dans le fichier), trouve le seuil optimal par méthode de Youden sur un sous-ensemble val, puis évalue sur RIMONE, Fundus et AIRROGS. Produit un tableau console et une figure de comparaison.

**Inputs requis (chemins codés en dur) :**
- `checkpoints/generalization_v3/<condition>/*.ckpt`

**Outputs — `figures/generalization_v3/`**
| Fichier | Contenu |
|---|---|
| `comparison_<condition>.png` | Barres AUC/Sensitivity/Specificity/F1 par test set |
| Console | Tableau comparatif complet |

---

### `dann_ablation.py`

**Ce que ça fait :** Ablation du DANN (Domain-Adversarial Neural Network). Pour chaque paire (condition × λ_domain), entraîne un modèle avec backbone DINOv3-Small + tête glaucome + GRL + tête de domaine. λ=0.0 sert de baseline (pas d'adversarial). Deux conditions de données : `mixed` (5 domaines) et `c` (3 domaines, miroir de la condition C).

**Outputs**
| Chemin | Contenu |
|---|---|
| `checkpoints/dann_ablation/<condition>_lam<λ>/` | Checkpoints |
| `figures/dann_ablation/ablation_results.json` | AUC par (condition, λ, test set) |
| `figures/dann_ablation/*.png` | Courbes d'entraînement |

---

### `train_cluster_c_dataset_split.py`

**Ce que ça fait :** Variante plus rigoureuse de l'expérience Cluster C étendu. Entraîne sur l'ancien Cluster C (JRAIGS + AIRROGS), mais utilise des datasets cibles *entiers* pour la validation (BEH + FIVES) plutôt qu'un split du train — ce qui rend l'early-stopping et le calibrage de seuil représentatifs du domaine cible. Test sur les datasets restants du nouveau Cluster C (PAPILA, sjchoi86-HRF, OIA-ODIR, DRISHTI-GS1, CRFO-v4, via MultichannelGlaucomaBenchmark).

**Outputs — `figures/cluster_c_dataset_split/`**
| Fichier | Contenu |
|---|---|
| `results_cluster_c_dataset_split.json` | AUC + métriques par dataset test |
| `results_cluster_c_dataset_split.png` | Bar chart par dataset (val vs test distingués) |

---

### `project_new_datasets.py`

**Ce que ça fait :** Projette de nouveaux datasets sur un espace UMAP déjà construit, sans le recalculer. Charge les features de `dataset_clustering.py`, fitte le UMAP sur les anciens datasets seulement (mêmes hyperparamètres + seed), puis transforme les nouveaux datasets et les superpose sur le même plan.

**Input requis :** `figures/clustering/<backbone>/features.npy` + `labels.json`

**Outputs**
| Fichier | Contenu |
|---|---|
| `umap_new_projected.png` | UMAP avec anciens datasets (contours) + nouveaux (points pleins) |

---

## Archive/

Contient les versions antérieures, conservées pour référence :

| Fichier | Description |
|---|---|
| `train_generalization.py` | Version 1 — architecture initiale |
| `train_generalization_v2.py` | Version 2 — ajout des conditions A–D |
| `train_generalization_v3.py` | Version 3 — version qui a produit les checkpoints `generalization_v3` |
| `image_synthesis_cluster.py` | Synthèse d'images par cluster |
| `image_synthesis_neural.py` | Synthèse d'images par méthode neurale |
| `image_synthesis_test.py` | Tests de synthèse d'images |
