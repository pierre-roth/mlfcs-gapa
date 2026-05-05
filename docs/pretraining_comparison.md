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

Final Slurm status: all 20 jobs completed successfully with exit code `0:0`.
Synthetic jobs took about 11-15 minutes each; real AAPL took about 36-44
minutes; real GOOGL took about 65 minutes.

## Results

| dataset | symbol | model | params | train acc | eval acc | eval macro F1 | eval loss | train samples | eval samples |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| real | AAPL | FC-LOB | 2328067 | 0.5489 | 0.5091 | 0.3445 | 0.8404 | 91348 | 20832 |
| real | AAPL | Conv-LOB | 29219 | 0.4847 | 0.4617 | 0.2106 | 0.8366 | 91348 | 20832 |
| real | AAPL | DeepLOB | 135523 | 0.4805 | 0.4617 | 0.2106 | 0.8351 | 91348 | 20832 |
| real | AAPL | Attn-LOB | 176515 | 0.4834 | 0.4617 | 0.2106 | 0.8374 | 91348 | 20832 |
| real | GOOGL | FC-LOB | 2328067 | 0.5858 | 0.5858 | 0.3998 | 0.8175 | 135437 | 33017 |
| real | GOOGL | Conv-LOB | 29219 | 0.4799 | 0.4676 | 0.2682 | 0.8522 | 135437 | 33017 |
| real | GOOGL | DeepLOB | 135523 | 0.5424 | 0.5687 | 0.3882 | 0.8290 | 135437 | 33017 |
| real | GOOGL | Attn-LOB | 176515 | 0.4781 | 0.4739 | 0.2144 | 0.8525 | 135437 | 33017 |
| synthetic | 000001 | FC-LOB | 2328067 | 0.7652 | 0.7630 | 0.4652 | 0.6346 | 440372 | 264530 |
| synthetic | 000001 | Conv-LOB | 29219 | 0.7717 | 0.7666 | 0.4391 | 0.6410 | 440372 | 264530 |
| synthetic | 000001 | DeepLOB | 135523 | 0.8686 | 0.8645 | 0.7475 | 0.3979 | 440372 | 264530 |
| synthetic | 000001 | Attn-LOB | 176515 | 0.8565 | 0.8505 | 0.7152 | 0.4609 | 440372 | 264530 |
| synthetic | 000858 | FC-LOB | 2328067 | 0.7738 | 0.7705 | 0.4365 | 0.6072 | 440546 | 264738 |
| synthetic | 000858 | Conv-LOB | 29219 | 0.7723 | 0.7694 | 0.4346 | 0.6339 | 440546 | 264738 |
| synthetic | 000858 | DeepLOB | 135523 | 0.8700 | 0.8679 | 0.7369 | 0.3812 | 440546 | 264738 |
| synthetic | 000858 | Attn-LOB | 176515 | 0.8653 | 0.8631 | 0.7263 | 0.3973 | 440546 | 264738 |
| synthetic | 002415 | FC-LOB | 2328067 | 0.7403 | 0.7063 | 0.6085 | 0.7329 | 440504 | 264389 |
| synthetic | 002415 | Conv-LOB | 29219 | 0.7152 | 0.7178 | 0.4278 | 0.7173 | 440504 | 264389 |
| synthetic | 002415 | DeepLOB | 135523 | 0.8341 | 0.8337 | 0.7521 | 0.4550 | 440504 | 264389 |
| synthetic | 002415 | Attn-LOB | 176515 | 0.8221 | 0.8250 | 0.7328 | 0.4703 | 440504 | 264389 |

## Interpretation

- On synthetic data, DeepLOB is the best supervised pretraining model on all
  three symbols by held-out accuracy and macro F1. Attn-LOB is close but does
  not win this bounded comparison.
- On real NASDAQ data, the ranking changes. FC-LOB wins on AAPL and GOOGL by
  held-out accuracy; DeepLOB is the strongest non-FC model on GOOGL. The low
  macro F1 values show that the real-label task is still much less clean than
  the synthetic one under the current `REAL_EVENT_STRIDE=250` calibration.
- Conv-LOB is much smaller and trains successfully, but it is consistently
  behind DeepLOB/Attn-LOB on synthetic data and behind FC-LOB on real data.
- These results replicate the pretraining-comparison *structure* from the
  paper, but they do not reproduce the paper's exact ranking because the data
  sources differ and Conv-LOB/DeepLOB are reimplemented from descriptions rather
  than copied from reference code.
