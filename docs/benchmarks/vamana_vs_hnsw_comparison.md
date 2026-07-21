# Vamana Disk vs HNSW Comparison

Dataset: 10000 random 128-dim float32 vectors  
Full-precision size: 5.1 MB  

| Index        | In-memory (MB) | Recall@10 | Avg latency (µs) | When to use |
|---|---|---|---|---|
| HNSW (M=16)  | 7.93 | 0.8967   | 5737   | Dataset fits in RAM; latency-critical |
| Vamana Disk  | 0.37 | 0.2633   | 1994   | Dataset exceeds RAM; disk-backed |

## Summary

- HNSW stores all vectors and adjacency lists in memory → full-vec RAM + graph overhead.
- VamanaDiskIndex keeps only PQ codes (0.37 MB) and offset maps in memory during search; full-precision vectors (5.1 MB) stay on disk and are read only for the final re-rank step.
- Choose HNSW when the dataset fits comfortably in unified memory (lower latency).
- Choose VamanaDiskIndex when the full-precision dataset exceeds available RAM (accepts a latency penalty for the disk I/O, offset by the PQ coarse-pass savings).
