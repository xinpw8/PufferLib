# ocean/rek

There is no REK environment here yet, on purpose.

What was here before was a fighting-game move state machine over a scalar
balance model: frame envelopes, circular hit volumes, hand-picked mass
exponents, invented friction and get-up timing. It was fast and internally
consistent, and it was a new game inspired by REK's controls rather than a
reconstruction of REK's transition function. A policy trained against it has no
reason to transfer. It is quarantined on the `rek_proxy` branch and is not a
training target.

For RL purposes a clone has to reproduce

    (s_{t+1}, r_t, d_t, e_t) = F(s_t, a_t^0, a_t^1, seed)

matching REK on action semantics, control and physics tick rates, input latency
and buffering, articulated dynamics, contact detection, balance and recovery,
hit/score/fall/round events, per-robot parameters, and reset and randomisation
distributions. The renderer does not need parity; the transition function does.

So the work starts with evidence rather than with another simulator. See
[`evidence/README.md`](evidence/README.md) for the sequence, what is tooled, and
what is still blocked on access to the installed build.

`tools/arm64/` and `tools/spark_train.sh` are infrastructure and survive
unchanged: they are about building and running PufferLib envs on aarch64 and on
the DGX Spark, and depend on nothing about REK.
