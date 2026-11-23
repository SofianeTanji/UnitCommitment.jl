# Convex hull pricing: optimal dual values

Computes `max_π L(π)` (the convex hull pricing dual of
`UnitCommitment.ConvexHullOracle`) for the UnitCommitment.jl benchmark
instances listed in `instances.txt`, by Dantzig–Wolfe column generation with
Wentges smoothing and parallel pricing (`dw.jl`), up to a relative gap of 1e-6.

Not included: RTS-GMLC and pglib-uc/ca/Scenario400 (time-varying minimum
power, not supported by the default thermal formulation), transmission
expansion instances, MATPOWER case300, case89pegase and larger cases (too
large with eager contingency constraints), and the MATPOWER instances on which
HiGHS returned OTHER_ERROR (case30 and 46 days of case14). MATPOWER cases are
run for 2017-01-01 only, except case14 (319 days, at the end of
`instances.txt`).

## Running on a CECI cluster (NIC6)

On a login node (compute nodes may have no internet access):

```bash
# Julia, through juliaup (installed in ~/.juliaup, added to the PATH by ~/.bashrc)
curl -fsSL https://install.julialang.org | sh -s -- -y
source ~/.bashrc
juliaup add 1.12 && juliaup default 1.12

git clone -b convex-hull-lmp-benchmark https://github.com/SofianeTanji/UnitCommitment.jl.git
cd UnitCommitment.jl/benchmark/convexhull
./download.sh                                            # instances/
module load releases/2025b Gurobi/13.0.1-GCCcore-14.3.0
GUROBI_JL_USE_GUROBI_JLL=false julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build("Gurobi"); Pkg.precompile()'
mkdir -p logs
sbatch --array=1-$(wc -l < instances.txt)%7 submit.sh
```

Each job uses 8 cores, 1900 MB per core (the NIC6 maximum) and at most 2
days. Everything is solved by Gurobi (HiGHS is not used), and each Gurobi
environment takes a license token: `dw.jl` uses one per thread, 8 per job, so
`%7` keeps the array within 60 tokens. When all jobs are done:

```bash
julia --project=. collect.jl
```

## Output

- `results/<name>.json`, one per instance: status (`optimal`, `time_limit` or
  `error`), lower and upper bounds on `max_π L(π)`, relative gap, LP
  relaxation value, `L` at the LP relaxation duals, iterations, times, the
  coupling row keys, the LP duals, the convex hull prices (the prices of the
  best lower bound) and the bounds at each iteration.
- `results.csv`: one row per instance with the scalar fields.
- `logs/<job>_<index>.out`: Slurm logs.

## Benchmark of the methods

`methods/` holds the methods that maximize `L` (subgradient, bundle, smoothing
and column generation), with the interface described in `methods/common.jl`.
`configs.jl` lists the 82 benchmarked configurations, each with at most one
tuned parameter, and the tuning instances: one per dataset (or-lib,
pglib-uc/ca and pglib-uc/ferc), the one with the median gap
`(L* - L(π0)) / |L*|`. The other instances on which `L(π0)` is not within
1e-6 of `L*` (79) are the evaluation instances. Every run starts from the LP
relaxation duals and lasts 300 s, or until `L` is within 1e-6 of `L*`.

The tuning: each tuned configuration runs on each tuning instance for an
11-value coarse grid of its parameter, then for 4 more values around the best
one. Then the evaluation runs with the selected values, for all the
configurations (trust region variants and CRG included).

The tuning and evaluation instances depend on `L*`, so the steps run after the
Gurobi references (see Solver). The traces of the earlier HiGHS runs are kept
in `traces_old/`, only to estimate the job times:

```bash
mv traces traces_old
julia --project=. plan.jl coarse        # tasks/coarse.txt, tasks/coarse_mip.txt
# submit the two arrays with the caps that plan.jl prints, e.g.
sbatch --array=1-$(wc -l < tasks/coarse.txt)%52 bench.sh tasks/coarse.txt traces/tune
sbatch --array=1-$(wc -l < tasks/coarse_mip.txt)%1 bench.sh tasks/coarse_mip.txt traces/tune
# when they are done:
julia --project=. plan.jl fine          # tasks/fine.txt, tasks/fine_mip.txt, idem
# when they are done:
julia --project=. plan.jl select        # tuned.json, and prints the selected values
julia --project=. plan.jl eval          # tasks/eval.txt, tasks/eval_mip.txt, idem with traces/eval
```

Gurobi tokens: the oracle solves the thermal unit blocks by dynamic
programming on all the threads, without Gurobi, so a job of `bench.sh` uses
one Gurobi environment (one token), except the jobs of `fgm_ch`, whose pricing
MILPs need one environment per thread (8 tokens). `plan.jl` therefore writes
the jobs of `fgm_ch` in a separate file (`*_mip.txt`), sorts each file by
decreasing estimated time (the longest first), and prints the caps `%M` of the
two arrays that stay within 60 tokens (`TOKENS=<n>` for another budget) and
minimize the estimated makespan, with both arrays running at the same time.
When the `*_mip.txt` array is done, its tokens can go to the other one:
`scontrol update JobId=<array job id> ArrayTaskThrottle=60`.

A regex after the step restricts it to the matching labels, with task files
suffixed accordingly; `select` then only updates their entries of
`tuned.json` (e.g. `plan.jl coarse -tr` for the trust region variants only).

Each job uses 8 cores and at most 4 hours. The traces are in
`traces/<stage>/<run>/<name>.json`: the parameters, the status (`target`,
`time_limit`, `optimal`, `unbounded`, `iterations` or `error`), and the time,
`L`, the certified bound and the method's upper bound at each evaluation.

## Solver

Gurobi solves the blocks and masters of `dw.jl` and of the methods. HiGHS is
not used anymore: the presolve of HiGHS 1.15.1 can return a suboptimal block
solution with status `OPTIMAL` and the same bound: on the unit g42 of
`or-lib/50_0_1_w`, at the best prices of the subgradient run, it reports
-9037.8 where Gurobi (and HiGHS without presolve) find -9584.7. The HiGHS
bounds on `L` are then above `L`; in the old HiGHS traces, the certified
bounds of some runs exceed the optimal value. The optimal values of the
development instances are recomputed with Gurobi to the relative gap 1e-8,
warm-started from the convex hull prices of the HiGHS run, together with the
test instances (see below; those without a HiGHS result start cold):

```bash
mv results results_highs
grep -E '^(or-lib|pglib-uc/(ca|ferc))/' instances.txt > instances_dev.txt   # 82 instances
cat instances_dev.txt instances_test.txt > instances_ref.txt               # 170 instances
julia --project=. order.jl instances_ref.txt results_highs                 # longest first
sbatch --array=1-$(wc -l < instances_ref.txt)%7 submit.sh instances_ref.txt 1e-8 results_highs
```

The Gurobi block MIPs are solved without presolve, which took most of their
time. The bundle masters are solved by the dual simplex, warm-started from the
last basis, instead of the default barrier (see `methods/bundle.jl`). In the
methods (not in `dw.jl`, which stays an independent MIP reference), the
thermal unit blocks are solved by dynamic programming (`methods/dp.jl`,
`ORACLE=mip` to turn it off): exact without the ramping constraints, and with
them by the convex value function of Frangioni and Gentile
(`methods/dp_ramp.jl`), for one or several scenarios, with or without an
investment variable, and with the flexiramp of `WanHob2016` (none of the
instances has one). Every solution is checked against every block constraint,
with the MIP as fallback. The traces record `dp_calls`, `dp_ramp` and
`dp_fallbacks`.

## Held-out test instances

The 82 instances above are the development set. The test set is kept aside
and is only run on the methods selected on the development set:
`instances_test.txt` lists 80 randomized variants (2 per instance) of the
pglib-uc/ca and pglib-uc/ferc instances, made by `make_test.jl` with the
randomization of `UnitCommitment.XavQiuAhm2021` (costs, load profile), the
peak load being drawn around the peak of the original instance. Each variant
is checked against `UnitCommitment.randomize!` with the same seed. It also
lists the 8 Belgian instances of ConvexHullPricing.jl (`data/belgian`),
converted by `make_belgian.jl` (one bus, 68 units, 96 periods, UnitCommitment.jl
defaults for the parameters the data do not give, initial conditions from
`UnitCommitment.generate_initial_conditions!`). The converted Belgian
instances are in the repository (`instances/belgian/`, 36 KB); the randomized
variants are made on the cluster, after `./download.sh`:

```bash
julia --project=. make_test.jl          # instances/pglib-uc/{ca,ferc}-rand/, keeps the Belgian lines of instances_test.txt
```

The optimal values of the 88 test instances are computed with those of the
development instances (see Solver).
