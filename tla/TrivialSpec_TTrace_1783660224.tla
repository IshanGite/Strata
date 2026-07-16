---- MODULE TrivialSpec_TTrace_1783660224 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, TrivialSpec

_expression ==
    LET TrivialSpec_TEExpression == INSTANCE TrivialSpec_TEExpression
    IN TrivialSpec_TEExpression!expression
----

_trace ==
    LET TrivialSpec_TETrace == INSTANCE TrivialSpec_TETrace
    IN TrivialSpec_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        x = (5)
    )
----

_init ==
    /\ x = _TETrace[1].x
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ x  = _TETrace[i].x
        /\ x' = _TETrace[j].x

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("TrivialSpec_TTrace_1783660224.json", _TETrace)

=============================================================================

 Note that you can extract this module `TrivialSpec_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `TrivialSpec_TEExpression.tla` file takes precedence 
  over the module `TrivialSpec_TEExpression` below).

---- MODULE TrivialSpec_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, TrivialSpec

expression == 
    [
        \* To hide variables of the `TrivialSpec` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        x |-> x
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_xUnchanged |-> x = x'
        
        \* Format the `x` variable as Json value.
        \* ,_xJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(x)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_xModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].x # _TETrace[s-1].x
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE TrivialSpec_TETrace ----
\*EXTENDS IOUtils, TLC, TrivialSpec
\*
\*trace == IODeserialize("TrivialSpec_TTrace_1783660224.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE TrivialSpec_TETrace ----
EXTENDS TLC, TrivialSpec

trace == 
    <<
    ([x |-> 0]),
    ([x |-> 1]),
    ([x |-> 2]),
    ([x |-> 3]),
    ([x |-> 4]),
    ([x |-> 5])
    >>
----


=============================================================================

---- CONFIG TrivialSpec_TTrace_1783660224 ----

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Fri Jul 10 10:40:24 IST 2026