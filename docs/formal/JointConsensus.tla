----------------------------- MODULE JointConsensus -----------------------------
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS Server, C1, C2

VARIABLES config, state, currentTerm, messages, votedFor

vars == <<config, state, currentTerm, messages, votedFor>>

Init ==
    /\ config = [i \in Server |-> C1]
    /\ state = [i \in Server |-> "Follower"]
    /\ currentTerm = [i \in Server |-> 1]
    /\ votedFor = [i \in Server |-> "None"]
    /\ messages = {}

Send(m) == messages' = messages \cup {m}

Quorums(C) == {Q \in SUBSET C : Cardinality(Q) * 2 > Cardinality(C)}
JointQuorums(C_old, C_new) == {Q1 \cup Q2 : Q1 \in Quorums(C_old), Q2 \in Quorums(C_new)}

ActiveQuorums(c) == 
    IF c = C1 THEN Quorums(C1)
    ELSE IF c = C2 THEN Quorums(C2)
    ELSE JointQuorums(C1, C2)

\* Simplified leader election
BecomeLeader(i, Q) ==
    /\ state[i] \in {"Follower", "Candidate"}
    /\ Q \in ActiveQuorums(config[i])
    /\ \A j \in Q: 
        \E m \in messages:
            /\ m.type = "Vote"
            /\ m.dest = i
            /\ m.source = j
            /\ m.term = currentTerm[i]
    /\ state' = [state EXCEPT ![i] = "Leader"]
    /\ UNCHANGED <<config, currentTerm, messages, votedFor>>

StartElection(i) ==
    /\ state' = [state EXCEPT ![i] = "Candidate"]
    /\ currentTerm' = [currentTerm EXCEPT ![i] = currentTerm[i] + 1]
    /\ votedFor' = [votedFor EXCEPT ![i] = i]
    /\ messages' = messages \cup {[type |-> "Vote", source |-> i, dest |-> i, term |-> currentTerm[i] + 1]}
    /\ UNCHANGED <<config>>

ReceiveVoteRequest(i, j, term) ==
    /\ term >= currentTerm[i]
    /\ \/ term > currentTerm[i]
       \/ votedFor[i] \in {"None", j}
    /\ currentTerm' = [currentTerm EXCEPT ![i] = term]
    /\ votedFor' = [votedFor EXCEPT ![i] = j]
    /\ state' = [state EXCEPT ![i] = IF term > currentTerm[i] THEN "Follower" ELSE state[i]]
    /\ messages' = messages \cup {[type |-> "Vote", source |-> i, dest |-> j, term |-> term]}
    /\ UNCHANGED <<config>>

StartConfigChange(i) ==
    /\ state[i] = "Leader"
    /\ config[i] = C1
    /\ config' = [config EXCEPT ![i] = C1 \cup C2] \* joint
    /\ messages' = messages \cup {[type |-> "AppendConfig", config |-> C1 \cup C2, dest |-> j, term |-> currentTerm[i]] : j \in Server}
    /\ UNCHANGED <<state, currentTerm, votedFor>>

CommitConfigChange(i) ==
    /\ state[i] = "Leader"
    /\ config[i] = C1 \cup C2
    /\ config' = [config EXCEPT ![i] = C2]
    /\ messages' = messages \cup {[type |-> "AppendConfig", config |-> C2, dest |-> j, term |-> currentTerm[i]] : j \in Server}
    /\ UNCHANGED <<state, currentTerm, votedFor>>

ReceiveConfig(i, m) ==
    /\ m.type = "AppendConfig"
    /\ m.dest = i
    /\ m.term >= currentTerm[i]
    /\ config' = [config EXCEPT ![i] = m.config]
    /\ currentTerm' = [currentTerm EXCEPT ![i] = m.term]
    /\ state' = [state EXCEPT ![i] = "Follower"]
    /\ votedFor' = [votedFor EXCEPT ![i] = IF m.term > currentTerm[i] THEN "None" ELSE votedFor[i]]
    /\ UNCHANGED <<messages>>

Next ==
    \/ \E i \in Server: StartElection(i)
    \/ \E i, j \in Server: ReceiveVoteRequest(i, j, currentTerm[j] + 1) \* simplified trigger
    \/ \E i \in Server, Q \in SUBSET Server: BecomeLeader(i, Q)
    \/ \E i \in Server: StartConfigChange(i)
    \/ \E i \in Server: CommitConfigChange(i)
    \/ \E i \in Server, m \in messages: ReceiveConfig(i, m)

Safety == 
    \A i, j \in Server:
        (state[i] = "Leader" /\ state[j] = "Leader" /\ currentTerm[i] = currentTerm[j]) => (i = j)

=============================================================================
