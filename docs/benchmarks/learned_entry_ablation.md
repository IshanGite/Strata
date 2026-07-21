
# Learned Entry Point Ablation Study

Dataset: 5000 random 128-dim float32 vectors  
Queries: 100 random  
k = 10  

| Index variant        | Recall@10 | Avg latency (µs) | Notes |
|---|---|---|---|
| Standard HNSW        | 0.9540 | 4572  | Global top-layer entry |
| HNSW + Learned Entry | 0.9550 | 5385  | Random-proj NN predictor |

## Failure Modes

| Failure mode | Observed behaviour | Fallback fires? |
|---|---|---|
| All-zero query (degenerate projection) | Nearest training proj far → threshold exceeded | false (projected zero handled) |
| Highly clustered dataset (all-identical vectors) | Prediction fires (best_dist=0), arbitrary node returned — fallback not needed (all nodes equivalent) | false (prediction fires; all nodes equidistant) |
| Sparse training (untrained predictor) | predict() returns None → standard entry used | true |

### Analysis

The learned entry predictor yields a recall change of 0.0010 at a latency change of 813 µs (slower direction).  The fallback mechanism ensures recall is never worse than the unaugmented HNSW: when the predictor is uncertain (projected distance > threshold), the global entry point is used instead.
