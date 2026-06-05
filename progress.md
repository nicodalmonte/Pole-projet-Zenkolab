# Project Progress & Documentation

## Obiettivo del progetto

Classificatore binario di glaucoma (glaucoma / non-glaucoma) su immagini retiniche (fundus photography).
Si addestrano 3 modelli standalone, poi si distilla la conoscenza dei 2 modelli grandi in uno piccolo.
Alla fine si confrontano tutte le configurazioni sul test set REFUGE2.

---

## Teoria

### Cos'è DINOv3

DINO è un metodo di self-supervised learning per Vision Transformers (ViT): il modello viene
preaddestrato senza etichette su grandi dataset. I modelli usati sono della famiglia **DINOv2 di Meta AI**,
preaddestrati su LVD-142M, identificati in timm con il suffisso `.lvd1689m`.

L'immagine viene divisa in patch 16×16, ognuna è un token del transformer. Il **CLS token** finale
aggrega tutto e viene usato come feature vector per la classificazione.

Varianti usate in questo progetto:
- `vit_small_patch16_dinov3.lvd1689m` — ~22 M parametri → **DINO small / student**
- `vit_large_patch16_dinov3.lvd1689m` — ~303 M parametri → **DINO large (teacher)**

### Cos'è EVA

EVA02-large è un ViT grande (304 M parametri) preaddestrato con masked image modeling.
Differenza principale rispetto a DINO: usa patch 14×14 e viene addestrato a **448 px** invece di 224 px,
quindi vede le immagini ad alta risoluzione. Identificato in timm come
`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`.

Nota: internamente sia DINOv3 che EVA usano l'architettura **EVABlock** (con ROPE attention) — è la
base comune della famiglia in timm.

### Come viene fine-tunato un modello

Il backbone viene caricato con i pesi pretrained e si aggiunge una testa di classificazione custom:

```
Immagine (224 px per DINO, 448 px per EVA)
    ↓
ViT backbone  [frozen per i primi 3 epoch, poi scongelato]
    ↓
CLS token → feature vector [batch, embed_dim]
    ↓
Head: LayerNorm → Dropout → Linear(embed_dim → 2)
    ↓
Logits [batch, 2]  →  CrossEntropy loss
```

### Cos'è la Knowledge Distillation

La Knowledge Distillation (KD, Hinton 2015) trasferisce la conoscenza da un modello grande (teacher)
a uno piccolo (student). Lo student impara sia dalle etichette vere (hard loss) sia dalle probabilità
prodotte dal teacher (soft loss).

**Perché funziona?** Il teacher non dice solo "è glaucoma" ma produce una distribuzione tipo
`[0.12, 0.88]` — comunicando quanto è incerto. Questa informazione extra aiuta lo student a
generalizzare meglio rispetto ad allenarsi solo sulle label binarie.

**Formula della loss (single teacher):**
```
Loss = α × KD_loss + (1 - α) × Hard_loss

KD_loss   = KL_div(log_softmax(student/T), softmax(teacher/T)) × T²
Hard_loss = CrossEntropy(student_logits, label_vera)

α = 0.15  →  15% distillazione, 85% label vere
T = 4.0   →  ammorbidisce le distribuzioni del teacher
```

**Dual distillation (2 teacher):**
```
Loss = λ_dino × KD_dino + λ_eva × KD_eva + (1 - λ_dino - λ_eva) × Hard_loss

λ_dino = 0.075, λ_eva = 0.075  →  totale distillazione = 15% (uguale al single)
```

Il teacher è sempre frozen in eval mode. Solo lo student viene aggiornato.

---

## Pipeline completa

Ci sono **6 esperimenti**, ognuno con il suo batch file. I primi 3 sono indipendenti,
i successivi dipendono dai risultati dei teacher.

```
[1] train_dino_large  ──┬──→ [4] distil_single   (teacher: DINO large)
[2] train_eva         ──┤──→ [5] distil_dual      (teacher: DINO large + EVA)
[3] train_dino_small  │  └──→ [6] ensemble         (DINO large + EVA combinati)
                      └─────────────────────────────
```

| # | Batch file | Cosa fa | Dipende da | Log file |
|---|---|---|---|---|
| 1 | `train_dino_large.batch` | Fine-tuna DINO large (303 M) standalone | — | `logs/train_dino_large_JOBID.out` |
| 2 | `train_eva.batch` | Fine-tuna EVA (304 M, 448 px) standalone | — | `logs/train_eva_JOBID.out` |
| 3 | `train_dino_small.batch` | Fine-tuna DINO small (22 M) standalone | — | `logs/train_dino_small_JOBID.out` |
| 4 | `distil_single.batch` | Distilla DINO large → DINO small | job 1 | `logs/distil_single_JOBID.out` |
| 5 | `distil_dual.batch` | Distilla DINO large + EVA → DINO small | job 1 + 2 | `logs/distil_dual_JOBID.out` |
| 6 | `ensemble.batch` | Eval ensemble DINO large + EVA (nessun training) | job 1 + 2 | `logs/ensemble_JOBID.out` |

`JOBID` è il numero assegnato da SLURM al momento del lancio (vedi sezione sotto).

---

## Come lanciare i training

### Comando completo (copia e incolla)

```bash
cd /raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab

# Step 1 — i 3 training indipendenti partono subito in parallelo
JID_DINO_L=$(sbatch --parsable train_dino_large.batch)
JID_EVA=$(sbatch --parsable train_eva.batch)
JID_DINO_S=$(sbatch --parsable train_dino_small.batch)

# Step 2 — la distillazione parte automaticamente quando i teacher sono pronti
JID_DS=$(sbatch --parsable --dependency=afterok:${JID_DINO_L} distil_single.batch)
JID_DD=$(sbatch --parsable --dependency=afterok:${JID_DINO_L}:${JID_EVA} distil_dual.batch)

# Step 3 — l'ensemble parte automaticamente quando DINO large + EVA sono pronti
JID_ENS=$(sbatch --parsable --dependency=afterok:${JID_DINO_L}:${JID_EVA} ensemble.batch)
```

### Come funzionano le dipendenze SLURM

`--dependency=afterok:X` significa: **aspetta che il job X finisca con successo** (exit code 0)
prima di partire. Se X fallisce, il job dipendente viene automaticamente cancellato da SLURM —
non parte mai nulla di sbagliato.

`--dependency=afterok:X:Y` = aspetta che **entrambi** X e Y abbiano finito con successo.

`--parsable` fa sì che `sbatch` stampi solo il job ID numerico, così lo catturiamo in una variabile.

Una volta lanciato tutto, **non devi fare nulla** — SLURM gestisce l'ordine automaticamente.

### Monitorare lo stato

```bash
squeue -u dalmonte_nic
```

Possibili stati:
- `RUNNING` — sta girando ora
- `PENDING (Resources)` — aspetta una GPU libera
- `PENDING (Dependency)` — aspetta che un job precedente finisca

### Verificare i risultati a fine training

I risultati sul test set REFUGE2 vengono stampati alla fine di ogni log:
```bash
# Esempio
grep "test_auc\|test_acc\|test_f1\|test_sensitivity\|test_specificity" logs/train_dino_large_4557.out
```

---

## Come funzionano i batch file

Un batch file è uno script bash con intestazione SLURM. Esempio semplificato di `train_dino_large.batch`:

```bash
#!/bin/bash
#SBATCH --job-name=train_dino_large          # nome visibile in squeue
#SBATCH --output=logs/train_dino_large_%j.out  # file di log (%j = job ID numerico)
#SBATCH --partition=prod80                   # pool di GPU da usare
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1  # 1 GPU A100 80 GB
#SBATCH --mem=64G                            # RAM CPU
#SBATCH --time=24:0:0                        # timeout massimo

uv run src/train/train.py \
    --backbone   vit_large_patch16_dinov3.lvd1689m \
    --run_name   dino_large \
    --lr         1e-4 \
    ...
```

I batch file di distillazione aggiungono anche l'**auto-detect del checkpoint**:
```bash
# Prende il checkpoint col val_auc più alto dalla directory
DINO_CKPT=$(ls checkpoints_dino_large/dino_large-*.ckpt | grep -v last | sort -t= -k3 -rn | head -1)
```
Logica: divide il nome file sul carattere `=`, prende il terzo campo (che è il valore di val_auc),
ordina numericamente in ordine decrescente, prende il primo. Fallisce con errore se la directory
è vuota, così non parte mai con un checkpoint sbagliato.

---

## Iperparametri

### Modelli standalone — DINO large, EVA, DINO small (tutti uguali tra loro)

| Parametro | Valore | Perché |
|---|---|---|
| lr | 1e-4 | standard per fine-tuning di modelli grandi pretrained |
| weight_decay | 1e-4 | regolarizzazione AdamW |
| batch_size | 32 | massimo che entra in memoria |
| max_epochs | 50 | con early stopping, raramente si arriva a 50 |
| precision | 16-mixed | fp16 per velocità e memoria, fp32 per accumulo gradiente |
| unfreeze_backbone_epoch | 3 | backbone frozen per 3 epoch per warm-up della testa |
| early stopping | patience=10, min_delta=1e-3 | si ferma se val_auc non migliora per 10 epoch |
| optimizer | AdamW | —  |
| scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | dimezza lr se val_auc stagna |

### Distillazione — single e dual (tutti uguali tra loro)

| Parametro | Valore | Perché |
|---|---|---|
| lr | 3e-4 | più alto del teacher: lo student parte da pesi ImageNet, ha più margine |
| weight_decay | 1e-4 | — |
| batch_size | 32 | — |
| max_epochs | 50 | — |
| precision | 16-mixed | necessario: teacher grandi in memoria + student |
| kd_temperature T | 4.0 | ammorbidisce le distribuzioni del teacher |
| kd_weight totale | 0.15 | 15% distillazione, 85% hard label |
| — single: kd_alpha | 0.15 | — |
| — dual: λ_dino + λ_eva | 0.075 + 0.075 | stesso totale del single per confronto equo |
| freeze_backbone_epochs | 0 | student non viene congelato |
| early stopping | patience=10, min_delta=1e-3 | — |

---

## Struttura del codice

```
src/
├── models/
│   ├── dino_v3_1.py          # DinoV3_1: usato per DINO large e DINO small
│   ├── eva_vit.py             # EvaViT: usato per EVA
│   ├── student.py             # StudentGlaucomaDistilled: student con 1 teacher
│   └── student_dual.py        # StudentGlaucomaDualDistilled: student con 2 teacher
├── datasets/
│   ├── JRAIGS.py              # dataset principale (~81k immagini)
│   ├── ACRIMA.py / LAG.py / ORIGA.py / REFUGE2.py / Fundus_Train_Val_Data.py
│   └── augmentations.py
├── train/
│   ├── train.py               # training standalone (usato da DINO large e DINO small)
│   └── train_eva.py           # training standalone EVA
├── distilation/
│   ├── distilation.py         # distillazione single-teacher
│   └── distilation_dual.py    # distillazione dual-teacher (2 dataloader separati)
├── ensemble_eval_dual_tf.py   # valutazione ensemble con transform diverse per modello
└── ensemble_eval.py           # (legacy, non usato)
```

### Perché distilation_dual.py ha 2 dataloader

DINO large vuole immagini a 224 px, EVA a 448 px. Non si può usare lo stesso dataloader
perché le normalizzazioni e le dimensioni sono diverse. La soluzione è `DualImageDataset`:
ogni `__getitem__` restituisce la stessa immagine trasformata a entrambe le risoluzioni:
- `image_a` (224 px) → student + DINO teacher
- `image_b` (448 px) → EVA teacher (solo in training, non in val/test)

---

## Checkpoint directories

| Directory | Nome file checkpoint | Usato da |
|---|---|---|
| `checkpoints_dino_large/` | `dino_large-epochXX-val_aucY.YYYY.ckpt` | `train_dino_large.batch` |
| `checkpoints_eva/` | `eva-epochXX-val_aucY.YYYY.ckpt` | `train_eva.batch` |
| `checkpoints_dino_small/` | `dino_small-epochXX-val_aucY.YYYY.ckpt` | `train_dino_small.batch` |
| `checkpoints_student/` | `student_single-epochXX-val_aucY.YYYY.ckpt` | `distil_single.batch` |
| `checkpoints_student_dual/` | `student_dual-epochXX-val_aucY.YYYY.ckpt` | `distil_dual.batch` |

Ogni directory contiene i **top-3 checkpoint per val_auc** + `last.ckpt` (l'ultimo epoch).
Il nome file include sempre il val_auc, così si vede subito quale è il migliore.
