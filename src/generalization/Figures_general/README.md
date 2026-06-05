# Benchmark — Vue d'ensemble

Ce dossier contient les scripts de **visualisation et d'analyse finale** des résultats de généralisation. Ce ne sont pas des scripts d'entraînement — ils lisent les JSON produits par les expériences et génèrent les figures et tableaux destinés au papier.

---

## Fichiers

### `generate_generalization_figures.py`

**Ce que ça fait :** Génère deux figures de synthèse pour le papier :

1. **Distance DINO → AUC** — Scatter plot de chaque dataset : distance centroïde DINO (axe X) vs AUC zero-shot (axe Y), avec une droite de régression linéaire et une couleur par zone de distance. Montre que plus un dataset est loin dans l'espace DINO, plus l'AUC chute.

2. **Plus de clusters d'entraînement → meilleure généralisation** — Bar chart des 5 conditions d'entraînement (A seul, B seul, C seul, A+B, A+B+C) × 3 test sets (RIMONE, AIRROGS, Fundus). Montre que diversifier les clusters d'entraînement améliore systématiquement l'AUC sur les domaines non vus.

**Inputs requis :**
- `figures/distance_generalization/results_jraigs.json`
- `figures/cluster_gen/results.json`

**Outputs**
| Fichier | Contenu |
|---|---|
| `figures/distance_vs_auc.png` | Scatter distance DINO → AUC avec régression |
| `figures/cluster_generalization.png` | Bar chart conditions × test sets |

---

### `generate_cluster_figure.py`

**Ce que ça fait :** Variante plus détaillée de la figure 2 ci-dessus, avec un layout en deux parties : une légende explicite du mapping cluster/dataset en haut, et les barres d'AUC en bas. Met en évidence visuellement à quel cluster (A/B/C) appartient chaque train set et chaque test set, pour rendre le raisonnement sur la proximité de cluster immédiatement lisible.

**Input requis :**
- `figures/cluster_gen/results.json`

**Output**
| Fichier | Contenu |
|---|---|
| `figures/cluster_generalization_v2.png` | Figure avec légende clusters + barres AUC |

---

### `generate_generalization_table.py`

**Ce que ça fait :** Lit les résultats des 4 conditions principales (A-D) et produit deux choses :
1. **Console** — Tableau texte AUC + Accuracy par condition × test set, suivi d'observations clés rédigées (ex. : JRAIGS-only → RIMONE : Sensitivity=0.046 = catastrophique).
2. **Figure heatmap** — Double heatmap (AUC | Accuracy) colorée `RdYlGn`, une ligne par condition, une colonne par test set. Format compact pour intégration directe dans le papier.

**Input requis :**
- `figures/generalization_v3/eval_unified_test_sets.json`

**Outputs**
| Fichier | Contenu |
|---|---|
| `generalization_performance_table.png` | Double heatmap AUC + Accuracy (sauvegardé à la racine du projet) |
| Console | Tableau texte + observations + conclusion |

---

### `benchmark_dino.py`

**Ce que ça fait :** Mesure le temps d'inférence CPU de DINOv3-Small et DINOv3-Large sur une image 1000×1000 synthétique, via `timm`. Affiche le ratio de vitesse Large/Small. Utile pour justifier le choix de DINOv3-Small dans le papier (compromis vitesse/performance).

**Aucun input requis** — génère une image aléatoire en mémoire.

**Output :** Console uniquement — temps en ms + taille d'embedding pour chaque modèle.

---

### `compute_pearson_correlation.py`

Fichier vide — pas encore implémenté.

---

## Inputs/Outputs résumés

```
figures/distance_generalization/results_jraigs.json  ──►  generate_generalization_figures.py  ──►  figures/distance_vs_auc.png
figures/cluster_gen/results.json                     ──►  generate_generalization_figures.py  ──►  figures/cluster_generalization.png
figures/cluster_gen/results.json                     ──►  generate_cluster_figure.py          ──►  figures/cluster_generalization_v2.png
figures/generalization_v3/eval_unified_test_sets.json ──► generate_generalization_table.py   ──►  generalization_performance_table.png
```
