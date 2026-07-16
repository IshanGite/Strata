# Strata Sharding Architecture

This document describes the design and rationale behind `strata-sharding`.

## 1. Shard Assignment: Range-Based vs. Consistent Hash

We chose **Range-Based Shard Assignment** over consistent hashing.

### Justification
1. **Ordered Scan Efficiency:** Range-based assignment stores lexicographically contiguous ranges of keys in a single shard. This allows range query scans (such as seeking over ranges of vector IDs or user-defined prefixes) to be executed as highly efficient local database scans. Consistent-hash-based assignment distributes contiguous keys across random shards, requiring cluster-wide scatter-gather operations for scans.
2. **Deterministic Splitting:** Splitting key ranges is highly deterministic and natural: a range `[start, end)` is split into two halves `[start, split_key)` and `[split_key, end)` at a specific pivot key.
3. **Database Rationale:** Real-world state-of-the-art vector/key-value storage engines (such as CockroachDB, TiKV, and Google Spanner) leverage range-based partitioning for ordering and transactional boundary scaling, making it the preferred pattern for modern databases.

---

## 2. Dynamic Membership via Joint Consensus

Config changes are executed in two phases following the Raft paper:
1. **Joint Configuration ($C_{\text{old,new}}$):** The leader proposes a log entry containing the union of the old configuration $C_{\text{old}}$ and new configuration $C_{\text{new}}$. Quorums during this phase require independent majorities from both groups to agree on commits and elections.
2. **Stable Configuration ($C_{\text{new}}$):** Once $C_{\text{old,new}}$ commits, the leader transitions the group to $C_{\text{new}}$, updating all peer states.

---

## 3. Shard Splitting, Merging, and Rebalancing

- **Splitting:** Initiated when a shard's key counts/storage footprint cross a threshold. Keys $\ge \text{split\_key}$ are local-dumped to a binary file, deleted from the parent state machine, and a new child Raft group is initialized.
- **Merging:** Underloaded adjacent ranges are coalesced. The donor dump-appends its entries to the target state machine and the donor Raft group shuts down.
- **Rebalancer:** A greedy rebalancer computes load discrepancies among physical nodes. If the discrepancy exceeds a threshold, it generates replica `Move` actions (via Joint Consensus) to balance node load. Moves are only scheduled if they strictly decrease the absolute load imbalance.
