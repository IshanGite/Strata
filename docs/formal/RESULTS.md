# Formal Verification Results

This document summarizes the TLC model checker results for the TLA+ specifications of the Strata database consensus and transaction protocols.

## Checked Specifications

### 1. Raft Core (`RaftCore.tla`)
- **Properties Checked**: 
  - `ElectionSafety`: At most one leader per term.
  - `LogMatching`: If two logs have an entry with the same index and term, they are identical up to that index.
  - `StateMachineSafety`: If a server has applied a log entry at a given index, no other server will ever apply a different log entry for the same index.
- **Model Bounds**: 3 Servers, 2 Values, Max Term = 3, Max Log Length = 3.
- **Result**: Checked without invariant violations.

### 2. Joint Consensus (`JointConsensus.tla`)
- **Properties Checked**:
  - `Safety`: No concurrent leaders with different configurations can be elected in the same term. Disjoint active quorums never simultaneously elect a leader.
- **Model Bounds**: 5 Servers (C1 = 3 servers, C2 = 3 servers, intersection = 1 server).
- **Result**: Checked without invariant violations.

### 3. Transaction Protocol (`TxnProtocol.tla`)
- **Properties Checked**:
  - `Atomicity`: No transaction is partially visible. If any key updated by a transaction is seen as committed, all other keys involved in the transaction are also seen as committed (by resolving via the primary lock).
- **Model Bounds**: 2 Keys, 2 Transactions.
- **Result**: Checked without invariant violations.

## Limitations of the Model

The model checking provides high confidence in the design within the tested bounds, but the following limitations apply:
- **State Space Bound**: The models are bounded to small numbers of servers (3-5), terms, and log entries to keep checking times reasonable. Unbounded log lengths or terms are not verified.
- **Network Model**: The network is modeled as a simple asynchronous message pool where messages can be reordered or delayed. Real-world Byzantine faults, arbitrary byte corruption, and complex network partitions (beyond simple message loss/delay) are not covered.
- **Implementation Discrepancy**: This verifies the *specification*, not the Rust codebase directly. Bugs might still exist in the Rust implementation due to memory handling, async task scheduling, or logical errors not present in the idealized TLA+ model.
