# Memory-bounded verification

On 2026-09-09 at 10:20:51 BST, kernel logs confirmed a **global OOM** that
killed the invoking Codex process, not a segmentation fault in inference.
Its anonymous RSS was about 9.3 GiB. Concurrent native/reference/comparison
processes used about 5.2 GiB; existing Trellis and shared-world servers used
about 6.0 and 3.1 GiB respectively, plus other applications. There was no swap.
Overlapping full-model verification with that little headroom was unsafe.
The interrupted optimized CPU run left its complete-sized binary output but
no comparison report or verified clean producer exit; it is not accepted
sanitizer or numerical evidence merely because the output file exists.

## Native jobs

`scripts/run_bounded.py` is an optional Linux/systemd development utility, not
an inference dependency. It wraps the **entire** native runner/comparison job:

```sh
python scripts/run_bounded.py --report generated/budget-run.json -- COMMAND ARG...
```

Defaults are a 6 GiB `memory.max`, 5 GiB `memory.high`, no swap, and a requirement
for at least 10 GiB additional `MemAvailable` headroom before execution. These
values are configurable with `--memory-mib` and `--reserve-mib`; they are not
claims that every workload fits that budget. Reports must be new files.
Unlike a virtual-address-space limit, the cgroup cap permits ASan's large
shadow-memory reservation.

The script fails closed if headroom, a systemd user scope, cgroup v2 controls,
or the serial lock are unavailable. The inside monitor verifies the actual
kernel limits before launching the command. `OOMPolicy=kill` sets
`memory.oom.group=1`, so an OOM terminates the whole job rather than leaving
its subprocesses behind. This follows the installed systemd scope manual;
see the [upstream scope documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.scope.html).

The lock is held inside the capped job. It therefore survives loss of its
invoking agent while inference is still running. Competing jobs fail before
launching a model. Successful completion records the command return code,
elapsed time, cgroup peak RAM and memory events. An incomplete inside record
is explicitly not proof of a clean exit. A nonzero parity exit remains nonzero.

The headroom check is a snapshot, not a physical reservation: unrelated jobs
can still grow. GPU-driver allocations and daemon-owned subprocesses may need
separate accounting. Keep heavy verification serialized, inspect current RAM,
and do not stop unrelated services without authorization.

## Original Docker captures

Docker's daemon creates containers outside a client's scope. The wrapper's
memory limit does **not** constrain those containers. Reference captures must
also use explicit Docker `--memory` and `--memory-swap` caps, be serialized
with native jobs, and leave the same host headroom. The left-hand original
CPU/CUDA image captures completed with a 4 GiB container cap and no swap.
Do not count a Docker-client cgroup as a container memory bound.

## Guard verification

A disposable process touching 256 MiB inside a verified 64 MiB scope exited
with signal 9. At 10:37:34 BST the kernel recorded `CONSTRAINT_MEMCG`, naming
only that test scope; its two Python processes were killed as a group. This
was an intentional bounded test, distinct from the earlier global OOM.
A subsequent 128 MiB smoke job acquired the lock, exited zero, peaked at
12,316,672 bytes and recorded zero OOM events. An attempted overlapping job
was rejected by the lock before execution.

Local ignored evidence is in `generated/diagnostics/budget-contained-oom.json`
and `budget-after-oom.json`. The initial smoke attempt rejected an unsupported
systemd property before starting a command; the implementation now uses the
documented `OOMPolicy=kill` setting and verifies the resulting cgroup value.

`optimized-sanitizers` is a separate CPU build with `-O2 -g`, no `NDEBUG` or
fast-math, and ASan/UBSan retained in both native code and GGML. All 42 normal
tests pass. ELF inspection confirms the executable links `libasan.so.8` and
`libubsan.so.1`, and the CPU module calls both sanitizers. Optimization is not
itself a parity result; full-model comparisons are still required.
