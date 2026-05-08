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

## Real-Data Subsampling Correction

The first real-data rows below used `REAL_EVENT_STRIDE=250`,
`EVENTS_PER_DAY_OVERRIDE=60000`, and `MAX_PRETRAIN_SAMPLES_PER_DAY=80000`.
Those settings were useful for a bounded smoke comparison, but they are not an
acceptable real-data replication result because they subsample/cap the L3
stream. They are retained only as a superseded diagnostic.

The corrected real-data rerun uses:

| setting | value |
|---|---|
| `REAL_EVENT_STRIDE` | 1 |
| `EVENTS_PER_DAY_OVERRIDE` | unset |
| `MAX_PRETRAIN_SAMPLES_PER_DAY` | unset |
| `PRETRAIN_THRESHOLD` | `1e-5` |

`PRETRAIN_THRESHOLD` has not been tuned to fit the data. It remains fixed at
the paper value for the replication run. The rerun summaries include
train/eval label counts so the paper threshold can be audited after completion.

Submitted corrected full-real rerun on Euler as `20260507_fullreal`:

| dataset | symbol | FC-LOB | Conv-LOB | DeepLOB | Attn-LOB |
|---|---|---:|---:|---:|---:|
| real | `AAPL` | 65689381 | 65689384 | 65689385 | 65689387 |
| real | `GOOGL` | 65689389 | 65689391 | 65689392 | 65689393 |

## Results

### Superseded Bounded Results

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

### Corrected Full-Real Results

The first corrected full-real grid attempt (`20260507_fullreal`, jobs
`65689381`...`65689393`) and the first threshold/class-weight sweep
(`20260507_thrsweep1`, jobs `65696145`...`65696203`) were cancelled before
completion. They used `REAL_EVENT_STRIDE=1` and no event/sample cap as intended,
but exposed two implementation bottlenecks:

- `PretrainDataset` was constructing normalized LOB windows one sample at a
  time through `DataLoader.__getitem__`, keeping jobs CPU-bound before useful
  GPU work.
- The real-data loader always built a visualization-only `depth_cube`; on
  full NASDAQ days this added a large Python loop before pretraining.

Both issues are now patched. Pretraining batches now construct normalized LOB
windows vectorized per batch, and real-data `depth_cube` construction is
disabled by default via `REAL_BUILD_DEPTH_CUBE=false` while remaining available
for visual diagnostics.

Single-job no-subsampling fix check:

| run | job | data | setting | status | elapsed | train samples | eval samples | eval acc | eval macro F1 | eval loss |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| `piroth2_pretrain_fixcheck_real_AAPL_fclob_t1e5_1day_skipcube_20260508` | 65790168 | AAPL real, first train day + first eval day | FC-LOB, `PRETRAIN_THRESHOLD=1e-5`, no class weights, `REAL_EVENT_STRIDE=1`, no event cap | completed | 13m56s | 2,365,884 | 4,088,257 | 0.7527 | 0.2863 | 0.6934 |

Label counts for the check were train `[316981, 1723640, 325263]` and eval
`[512782, 3077206, 498269]`. The high accuracy but low macro F1 confirms that
the paper threshold is dominated by the stationary class on AAPL real data after
one epoch; threshold/class-weight diagnostics are still needed. This check also
shows that one full real day is a practical unit for the current implementation.

## Threshold And Class-Imbalance Diagnostics

The paper replication uses `PRETRAIN_THRESHOLD=1e-5` unchanged. In parallel, a
diagnostic sweep is being added to check whether a different threshold is more
appropriate for the real NASDAQ data and whether class-balanced cross-entropy
improves held-out macro F1.

Implemented diagnostics:

- `cluster/scan_piroth2_pretrain_thresholds.py`: full-real price-only label
  balance scan over thresholds, with no stride or event cap.
- `PRETRAIN_CLASS_WEIGHT_MODE=balanced`: inverse-frequency class weighting for
  supervised pretraining.
- `cluster/submit_piroth2_pretrain_threshold_sweep.sh`: full-real pretraining
  sweep over thresholds and class-weighting modes.

Completed label-scan job:

| job | purpose | data | output |
|---:|---|---|---|
| 65691807 | full-real threshold label-balance scan | AAPL/GOOGL, `REAL_EVENT_STRIDE=1` | `/cluster/project/math/piroth/mlfcs-gapa/artifacts_piroth2/pretrain_threshold_scan_20260507_fullreal.csv` |

Label-balance scan summary:

| symbol | split | threshold | up frac | stationary frac | down frac | minority frac |
|---|---|---:|---:|---:|---:|---:|
| AAPL | train | `2.5e-6` | 0.2343 | 0.5307 | 0.2350 | 0.2343 |
| AAPL | train | `5e-6` | 0.2121 | 0.5750 | 0.2129 | 0.2121 |
| AAPL | train | `1e-5` | 0.1489 | 0.7022 | 0.1489 | 0.1489 |
| AAPL | eval | `2.5e-6` | 0.2387 | 0.5238 | 0.2376 | 0.2376 |
| AAPL | eval | `5e-6` | 0.2152 | 0.5705 | 0.2143 | 0.2143 |
| AAPL | eval | `1e-5` | 0.1474 | 0.7054 | 0.1472 | 0.1472 |
| GOOGL | train | `2.5e-6` | 0.3395 | 0.3209 | 0.3396 | 0.3209 |
| GOOGL | train | `5e-6` | 0.2790 | 0.4422 | 0.2788 | 0.2788 |
| GOOGL | train | `1e-5` | 0.1862 | 0.6281 | 0.1856 | 0.1856 |
| GOOGL | eval | `2.5e-6` | 0.3297 | 0.3409 | 0.3294 | 0.3294 |
| GOOGL | eval | `5e-6` | 0.2527 | 0.4951 | 0.2522 | 0.2522 |
| GOOGL | eval | `1e-5` | 0.1613 | 0.6774 | 0.1612 | 0.1612 |

Rejected thresholds: `0` has no stationary class; `2e-5` leaves only about
5-6% directional minority; `5e-5` and `1e-4` collapse almost entirely to the
stationary class.

The original threshold/class-weight training sweep `20260507_thrsweep1` was
cancelled for the same preprocessing reason as the fixed-threshold grid. Do not
restart the full grid blindly. The next replication step should be either a
one-day architecture/threshold/class-weight comparison, or a small day-count
scaling test, using the patched loader/training path and no event-level
subsampling.
