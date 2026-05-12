# Results comparison — REFUGE2 test set

| Metric      |   DINO   |   EVA    | Ensemble | Student (dual) |
|-------------|:--------:|:--------:|:--------:|:--------------:|
| acc         |  0.9650  |  0.9550  | **0.9675** |    0.9600    |
| auc         |  0.9218  |  0.9404  |  0.9342  |  **0.9440**  |
| f1          |  0.8108  |  0.7692  | **0.8219** |    0.7778    |
| sensitivity |  0.7500  |  0.7500  |  0.7500  |    0.7000    |
| specificity |  0.9889  |  0.9778  | **0.9917** |    0.9889    |

Models: `dinov3_1_v2-epoch=10-val_auc=0.9154` · `eva_vit_v0-epoch=17-val_auc=0.9252` · `student_dual-lD0.15-lE0.15-epoch=22-val_auc=0.9249`
