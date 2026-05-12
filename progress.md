# Project Progress & Documentation

## Obiettivo del progetto

Classificatore binario di glaucoma (glaucoma / non-glaucoma) su immagini retiniche (fundus photography), basato su Vision Transformers della famiglia DINOv3. Il workflow è in due fasi: training di un teacher grande, poi distillazione in uno student compatto.

---

## Teoria

### Cos'è DINOv3

DINO (Self-Distillation with NO labels) è un metodo di self-supervised learning per Vision Transformers (ViT). Il modello viene preaddestrato senza etichette su grandi dataset di immagini. I modelli usati qui sono della famiglia **DINOv2 di Meta AI**, preaddestrati su LVD-142M (142M immagini curate automaticamente), identificati in timm con il suffisso `.lvd1689m`.

Architettura base: l'immagine viene divisa in patch 16×16, ognuna trattata come un token dal transformer. Il **CLS token** finale aggrega le informazioni di tutta l'immagine e viene usato come feature vector per la classificazione.

Varianti disponibili (ordine crescente di parametri):
- `vit_small_patch16_dinov3.lvd1689m` — ~22M parametri
- `vit_small_plus_patch16_dinov3.lvd1689m`
- `vit_large_patch16_dinov3.lvd1689m` — ~307M parametri
- `vit_huge_plus_patch16_dinov3.lvd1689m` — >600M parametri

### Come viene usato il modello

Il backbone viene caricato con `num_classes=0` (nessun classificatore interno) e produce un vettore di feature. Sopra ci viene aggiunta una testa di classificazione custom:

```
Immagine (224×224)
    ↓
ViT backbone (pretrained, frozen inizialmente)
    ↓
CLS token → feature vector [B, embed_dim]
    ↓
Head: LayerNorm → Dropout → Linear(embed_dim → 2)
    ↓
Logits [B, 2]  →  CrossEntropy
```

Per i primi `unfreeze_backbone_epoch` epoch il backbone è congelato — si allena solo la testa. Poi si scongela tutto insieme.

### Cos'è la Knowledge Distillation

La Knowledge Distillation (KD, Hinton et al. 2015) trasferisce la conoscenza da un modello grande (teacher) a uno piccolo (student), ottenendo un modello compatto con performance quasi equivalenti.

**Perché non basta allenare lo student direttamente?** Le label hard (0/1) non contengono informazione relazionale. Il teacher, invece, produce distribuzioni di probabilità "morbide": se predice `[0.12, 0.88]` su un caso borderline, sta comunicando *quanto* è incerto, non solo la classe. Questa informazione è preziosa per lo student.

**Formula della loss:**

```
Loss_totale = α × KD_loss + (1 - α) × Hard_loss

KD_loss = KL_div(
    log_softmax(student_logits / T),
    softmax(teacher_logits / T)
) × T²

Hard_loss = CrossEntropy(student_logits, label, weight=class_weights)
```

- **T (temperature, default=4.0)**: ammorbidisce le distribuzioni. A T=1 il modello è "sicuro", a T alto le probabilità si distribuiscono più uniformemente, rendendo le differenze tra classi più informative. Il fattore T² compensa la riduzione di grandezza del gradiente.
- **α (kd_alpha, default=0.7)**: 70% del segnale viene dal teacher (soft labels), 30% dalle etichette vere.

Il teacher è tenuto frozen in eval mode durante tutta la distillazione. Solo lo student viene aggiornato.

---

## Architettura del codice

### Struttura

```
src/
├── models/
│   ├── dino_v3_1.py      # Teacher: DinoV3_1
│   └── student.py         # Student: StudentGlaucomaDistilled
├── datasets/
│   ├── JRAIGS.py          # Dataset principale per il teacher
│   ├── ACRIMA.py
│   ├── LAG.py
│   ├── ORIGA.py
│   ├── REFUGE2.py
│   ├── Fundus_Train_Val_Data.py
│   └── augmentations.py
├── train/
│   └── train.py           # Script training teacher
└── distilation/
    └── distilation.py     # Script distillazione
job.batch                  # SLURM: lancia train.py
distilation.batch          # SLURM: lancia distilation.py
```

### DinoV3_1 (teacher) — `src/models/dino_v3_1.py`

Classificatore binario glaucoma. Usa timm per caricare il backbone ViT pretrained.

- **Loss**: CrossEntropy con class weights (per sbilanciamento glaucoma/non-glaucoma)
- **Ottimizzatore**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3, monitor val_auc)
- **Metriche**: AUC, Accuracy, F1, Sensitivity (Recall), Specificity
- **Freeze strategy**: backbone congelato per i primi `unfreeze_backbone_epoch` epoch

### StudentGlaucomaDistilled (student) — `src/models/student.py`

Stesso schema del teacher ma con testa leggermente più profonda e loss KD aggiuntiva.

- **Testa**: LayerNorm → Dropout → Linear → GELU → Dropout → Linear (due layer invece di uno)
- **Loss**: α×KD_loss + (1-α)×CE (vedi teoria sopra)
- **Teacher**: tenuto frozen, usato solo per inferenza in `_compute_loss`
- **Ottimizzatore**: AdamW (lr=3e-4, più alto del teacher)
- **Scheduler**: ReduceLROnPlateau (stesso schema)

### Dataset — JRAIGS

Dataset principale per il training del teacher. Contiene immagini retiniche con label da CSV (`filtered_labels.csv`):
- `RG` → glaucoma (1)
- `NRG` → non-glaucoma (0)

~81000 immagini totali, ~3.2% glaucoma → forte sbilanciamento. Il training usa un subset bilanciato: **tutte le immagini glaucoma + campionamento casuale non-glaucoma fino a 8000 totali** (seed=42).

Class weights calcolati automaticamente con inverse-frequency balancing:
```
w_neg = total / (2 × n_non_glaucoma)  ≈ 0.52
w_pos = total / (2 × n_glaucoma)       ≈ 15.5
```

---

## Workflow completo

### Fase 1 — Training Teacher (`sbatch job.batch`)

| Split | Dataset | Note |
|---|---|---|
| **Train** | JRAIGS (subset bilanciato ~8000) | Tutti i glaucoma + campionamento NRG |
| **Val** | ACRIMA + ORIGA + LAG | Split `train` di ciascuno |
| **Test** | REFUGE2 | Held-out, valutato solo a fine training |

Output: checkpoint in `checkpoints/version_0/dinov3_1_v0-epochXX-val_aucYYYY.ckpt`

### Fase 2 — Distillazione (`sbatch distilation.batch`)

| Split | Dataset | Note |
|---|---|---|
| **Train** | ACRIMA + FundusTrainVal(train) + LAG(train) + ORIGA | Dataset diversi dal teacher |
| **Val** | FundusTrainVal(val) + LAG(val) | — |
| **Test** | REFUGE2 | Held-out, valutato solo a fine distillazione |

Il batch carica automaticamente il best checkpoint del teacher. Output in `checkpoints_student/`.

---

## Modifiche apportate

### 1. `job.batch` — Aggiunta esplicita del backbone large

Il codice di default usava `vit_huge_plus`. Aggiunto `--backbone` esplicito:

```bash
# Prima
uv run src/train/train.py --batch_size 32 --precision 32 --image_size 224

# Dopo
uv run src/train/train.py --batch_size 32 --precision 32 --image_size 224 --backbone vit_large_patch16_dinov3.lvd1689m
```

### 2. `distilation.batch` — Sostituzione backbone e checkpoint dinamico

**Backbone teacher e student:**
```bash
# Prima
--teacher_backbone vit_huge_plus_patch16_dinov3.lvd1689m
--student_backbone vit_small_plus_patch16_dinov3.lvd1689m

# Dopo
--teacher_backbone vit_large_patch16_dinov3.lvd1689m
--student_backbone vit_small_patch16_dinov3.lvd1689m
```

**Checkpoint teacher — da hardcoded (e inesistente) ad auto-detect:**
```bash
# Prima (hardcoded, file non esistente)
TEACHER_CKPT=$PROJECT/checkpoints/dinov3_1-v_num=00-epoch=39-val_auc=0.9924.ckpt
...
--teacher_ckpt checkpoints/dinov3_1-v_num=00-epoch=28-val_auc=0.9942.ckpt

# Dopo (seleziona automaticamente il best checkpoint dopo il training)
TEACHER_CKPT=$(ls checkpoints/version_0/dinov3_1_v0-*.ckpt 2>/dev/null | grep -v last | sort -t= -k3 -rn | head -1)
...
--teacher_ckpt "$TEACHER_CKPT"
```

La logica del glob: lista tutti i `.ckpt` di version_0, esclude `last.ckpt`, ordina per val_auc (terzo campo dopo split su `=`) in ordine numerico inverso, prende il primo.

### 3. Pulizia — Eliminati tutti i log e checkpoint dei training precedenti

Rimossi:
- `checkpoints/version_2/` (4 file .ckpt del teacher precedente)
- `lightning_logs/dinov3_1/` (version_0, version_1, version_2)
- `logs/train_3146.out`, `logs/train_2788.out`, `logs/train_2518.out`
