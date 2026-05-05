# LOB Encoder Pretraining Comparison

This document tracks the Table-I-style supervised pretraining replication for
the LOB encoders discussed in the paper.

## Scope

Implemented encoders:

| model | status | notes |
|---|---|---|
| FC-LOB | implemented | Flattened `50 x 40` LOB window with `1024 -> 256 -> 64` dense stack. |
| Conv-LOB | implemented | Dilated temporal convolution baseline with pooled 64-dimensional output. |
| DeepLOB | implemented | CNN/inception feature extractor followed by LSTM temporal aggregation. |
| Attn-LOB | implemented | Existing attention encoder matching the authors' `network.py` dimensions. |

The downloaded reference repository only contains Attn-LOB and FC-LOB code. The
Conv-LOB and DeepLOB implementations are therefore based on the paper
description and the standard DeepLOB architecture family, not copied from a
missing reference implementation.

## Experiment Settings

Launcher:

```bash
cluster/submit_piroth2_pretrain_comparison.sh
```

Common settings:

| setting | value |
|---|---|
| `LOOKBACK` | 50 |
| `PRETRAIN_HORIZON` | 10 |
| `PRETRAIN_THRESHOLD` | `1e-5` |
| `PRETRAIN_STABLE_WINDOWS_ONLY` | true |
| `TORCH_EPOCHS` | 8 |
| `TORCH_BATCH_SIZE` | 2048 |
| `MAX_PRETRAIN_SAMPLES_PER_DAY` | 80000 |

Datasets:

| dataset | symbols | split | notes |
|---|---|---|---|
| synthetic | `000001`, `000858`, `002415` | 10 train / 6 test days | flow/volatility synthetic generator variant. |
| real NASDAQ | `AAPL`, `GOOGL` | 8 train / 4 test days | `REAL_EVENT_STRIDE=250`, 09:30-16:00 load window. |

Metrics written per run:

- checkpoint: `models/{model}_pretrain.pt` (`attnlob_pretrain.pt` for Attn-LOB)
- epoch history: `models/{model}_pretrain_history.csv`
- summary JSON: `models/{model}_pretrain_summary.json`
- train and held-out eval loss, accuracy, macro precision, macro recall, macro F1

Collector:

```bash
python cluster/collect_piroth2_pretrain_comparison.py --stamp 20260505_143142
```

## Submitted Runs

Submitted on Euler at `20260505_143142`.

| dataset | symbol | FC-LOB | Conv-LOB | DeepLOB | Attn-LOB |
|---|---|---:|---:|---:|---:|
| synthetic | `000001` | 65437523 | 65437524 | 65437525 | 65437526 |
| synthetic | `000858` | 65437528 | 65437530 | 65437531 | 65437533 |
| synthetic | `002415` | 65437535 | 65437537 | 65437538 | 65437539 |
| real | `AAPL` | 65437540 | 65437541 | 65437542 | 65437543 |
| real | `GOOGL` | 65437544 | 65437545 | 65437546 | 65437547 |

Initial Slurm status: all 20 jobs started running.

## Results

Pending Euler completion.
