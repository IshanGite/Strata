# Strata Design Decisions

This document outlines the core architectural and algorithmic choices made in Strata, detailing the alternatives considered and the rationale behind each decision.

## 1. Vector Indexing: Vamana vs. HNSW

### Context
Strata needs an indexing algorithm for fast, high-recall approximate nearest neighbor (ANN) search over millions of dense vectors.

### Alternatives
- **HNSW (Hierarchical Navigable Small World):** The industry standard (used by FAISS, Weaviate, Qdrant, Milvus). It uses a multi-layered graph for fast logarithmic search but consumes significant memory (bidirectional edges on multiple layers).
- **Vamana (DiskANN):** A single-layer graph algorithm tailored for out-of-core (disk-backed) memory. It builds an RNG (Relative Neighborhood Graph) and optimizes for lower memory overhead and highly sequential disk reads.

### Decision
**Vamana** was selected for Strata's primary on-disk index, with a fallback to HNSW for in-memory partitions.
- **Why:** Vector embeddings are extremely memory intensive. HNSW's memory footprint grows impractically large when scaling to billions of vectors. Vamana allows Strata to spill indexes to SSDs while maintaining high recall and acceptable latency, drastically reducing the total cost of ownership (TCO) for cluster operators.

## 2. Sharding Strategy: Hash vs. Range

### Context
Data must be partitioned across nodes in the Raft cluster to scale out storage and compute.

### Alternatives
- **Hash Sharding:** Keys are hashed, and the hash determines the shard. Leads to perfectly even data distribution and avoids hotspots.
- **Range Sharding:** Keys are ordered sequentially, and contiguous blocks of keys form a shard. Allows for efficient range queries.

### Decision
**Hash Sharding** is utilized for vector data.
- **Why:** Vector similarity search (k-NN) inherently requires a scatter-gather query pattern across all data since nearest neighbors could reside anywhere in the dataset. Range sharding offers no optimization for vector search unless vectors are pre-clustered (like IVF), which adds unacceptable write-time overhead. Hash sharding guarantees evenly distributed data and query load, preventing stragglers during the scatter-gather fan-out.

## 3. Clock Synchronization: HLC vs. TrueTime

### Context
Cross-shard distributed transactions require globally comparable timestamps to provide Serializability.

### Alternatives
- **Google TrueTime:** Hardware-assisted clocks (atomic/GPS) providing strict bounds on uncertainty. Offers external consistency (Strict Serializability) but requires specialized hardware.
- **Hybrid Logical Clocks (HLC):** Combines physical NTP clocks with a logical counter.

### Decision
**Hybrid Logical Clocks (HLC)**.
- **Why:** Strata is designed to be an open-source system deployable on commodity hardware in standard cloud environments (AWS, GCP, bare metal) where hardware-assisted bounded clocks are unavailable. HLCs provide strong Serializability without external consistency, which is an acceptable tradeoff for most open-source deployments.

## 4. Distributed Transactions: Percolator vs. Basic 2PC

### Context
To support atomic multi-shard writes and secondary indexing updates, a distributed commit protocol is required.

### Alternatives
- **Basic Two-Phase Commit (2PC):** Requires a heavy coordinator state, blocks indefinitely on coordinator failure, and suffers from read-write blocking.
- **Percolator-style 2PC:** Decentralizes the transaction state into the data cells themselves using locks and write intent records.

### Decision
**Percolator-style 2PC**.
- **Why:** Percolator elegantly maps to our Raft-backed LSM tree. By storing locks alongside data (`lock_key`, `write_key`), we remove the need for a centralized transaction manager, eliminating a massive scalability bottleneck and single point of failure. It natively provides Snapshot Isolation (upgraded to Serializability via conflict checking) and supports lock-free MVCC reads.
