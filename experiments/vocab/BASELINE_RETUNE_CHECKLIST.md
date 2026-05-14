# Baseline Re-tune Tracking (Unweighted Loss for All)

## 목표
PORT가 unweighted BCE인 것과 fair comparison 위해 LR/XGB/BiLSTM도 모두 unweighted loss로 재학습.

## 수정 필요 파일

| File | 추가할 flag |
|---|---|
| `baselines/logreg_xgb.py` | `--unweighted` (class_weight=None, scale_pos_weight=1) |
| `baselines/logreg_xgb_tuned.py` | `--unweighted` |
| `baselines/lstm.py` | `--unweighted` (pos_weight 비활성화) |
| `baselines/lstm_tuned.py` | `--unweighted` |

## 작업 분류

### CPU jobs (common partition, NOW)

| Script | 내용 | 파일 출력 (suffix `_unweighted`) |
|---|---|---|
| `slurm_lr_xgb_unweighted_tuned.sh` | HP sweep on full history (manual + MEDS) | `test_preds_manual_unweighted.parquet`, `test_preds_meds_unweighted.parquet` |
| `slurm_lr_xgb_unweighted_ctx.sh` | best config × 4 windows (7/30/90/365d) × 2 feat sets | `test_preds_{manual,meds}_unweighted_window_{N}d.parquet` |

### GPU jobs (gpu-hp, AFTER Stage 2)

| Script | 내용 | 파일 출력 |
|---|---|---|
| `slurm_bilstm_unweighted_tuned.sh` | 8-config HP grid | `lstm_unweighted_test_predictions.parquet` |
| `slurm_bilstm_unweighted_ctx.sh` | best config × 4 windows | `lstm_unweighted_test_predictions_window_{N}d.parquet` |

## Job ID 추적

| Job ID | Script | 상태 |
|---|---|---|
| TBD | LR/XGB tuned unweighted | PENDING |
| TBD | LR/XGB ctx window unweighted | PENDING |
| TBD | BiLSTM tuned unweighted | NOT submitted (Stage 2 후) |
| TBD | BiLSTM ctx window unweighted | NOT submitted (Stage 2 후) |

## Paper 업데이트 후 (결과 도착 후)

| Asset | 변경 |
|---|---|
| Table 3 (main results) | LR/XGB/BiLSTM 4 rows new |
| Fig 2 ROC/PR | BiLSTM/XGB/LR 곡선 new |
| Fig 4 context window | LR/XGB/BiLSTM 곡선 new |
| Supp Table confusion | BiLSTM/LR rows new |
| Supp Table baseline_sweep | 전체 재 |
| Supp Fig calibration | 4-model new |
| Supp Fig DCA | 4-model new |
| §3.2 baseline 단락 | "All baselines used unweighted BCE..." |
