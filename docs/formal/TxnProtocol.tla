------------------------------ MODULE TxnProtocol ------------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Key, TxnId

VARIABLES store, txns

vars == <<store, txns>>

\* store[k] = [value |-> ..., lock |-> ...]
\* txns[t] = [status |-> "Pending" | "Committed" | "Aborted", primary |-> k, secondaries |-> {k...}]

Init ==
    /\ store = [k \in Key |-> [value |-> 0, lock |-> "None"]]
    /\ txns = [t \in TxnId |-> [status |-> "Pending", primary |-> "None", secondaries |-> {}]]

StartTxn(t, p, secs) ==
    /\ txns[t].status = "Pending"
    /\ txns[t].primary = "None"
    /\ p \notin secs
    /\ txns' = [txns EXCEPT ![t].primary = p, ![t].secondaries = secs]
    /\ UNCHANGED <<store>>

Prewrite(t, k) ==
    /\ txns[t].status = "Pending"
    /\ txns[t].primary /= "None"
    /\ k \in {txns[t].primary} \cup txns[t].secondaries
    /\ store[k].lock = "None"
    /\ store[k].value = 0
    /\ store' = [store EXCEPT ![k].lock = t]
    /\ UNCHANGED <<txns>>

CommitPrimary(t) ==
    /\ txns[t].status = "Pending"
    /\ txns[t].primary /= "None"
    /\ store[txns[t].primary].lock = t
    /\ \A k \in txns[t].secondaries: store[k].lock = t
    /\ store' = [store EXCEPT ![txns[t].primary].lock = "None", ![txns[t].primary].value = t]
    /\ txns' = [txns EXCEPT ![t].status = "Committed"]

CommitSecondary(t, k) ==
    /\ txns[t].status = "Committed"
    /\ k \in txns[t].secondaries
    /\ store[k].lock = t
    /\ store' = [store EXCEPT ![k].lock = "None", ![k].value = t]
    /\ UNCHANGED <<txns>>

Abort(t) ==
    /\ txns[t].status = "Pending"
    /\ txns[t].primary /= "None"
    /\ txns' = [txns EXCEPT ![t].status = "Aborted"]
    /\ UNCHANGED <<store>>

Rollback(t, k) ==
    /\ txns[t].status = "Aborted"
    /\ store[k].lock = t
    /\ store' = [store EXCEPT ![k].lock = "None"]
    /\ UNCHANGED <<txns>>

Next ==
    \/ \E t \in TxnId, p \in Key, secs \in SUBSET Key: StartTxn(t, p, secs)
    \/ \E t \in TxnId, k \in Key: Prewrite(t, k)
    \/ \E t \in TxnId: CommitPrimary(t)
    \/ \E t \in TxnId, k \in Key: CommitSecondary(t, k)
    \/ \E t \in TxnId: Abort(t)
    \/ \E t \in TxnId, k \in Key: Rollback(t, k)
    \/ (\A t \in TxnId: txns[t].status \in {"Committed", "Aborted"}) /\ UNCHANGED vars

\* State machine safety for atomicity: No partial visibility
\* A transaction is either fully visible or not visible at all.
\* Since TLC can explore all interleavings, we formulate the property as:
\* If any key has value `t` (meaning `t` committed and wrote to it),
\* then for all other keys that `t` intended to write to, their value must also be `t`,
\* OR if we read it, we would resolve its lock to `t` by checking the primary.
\* Actually, the safety property is simpler:
\* We define what a read would return.
Read(k) ==
    IF store[k].lock = "None" THEN store[k].value
    ELSE IF txns[store[k].lock].status = "Committed" THEN store[k].lock
    ELSE store[k].value

Atomicity ==
    \A t \in TxnId:
        txns[t].status = "Committed" =>
            \A k \in {txns[t].primary} \cup txns[t].secondaries:
                Read(k) = t

Safety == Atomicity

=============================================================================
