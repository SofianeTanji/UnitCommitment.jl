# Tuning and evaluation plan of the configurations of configs.jl. Each line of
# a task file is the instance and the runs of one job of bench.sh (see
# run.jl).
#
# Usage: julia --project=. plan.jl <step> [<label regex>], with the steps in
# this order:
#
#   coarse  tasks/coarse.txt: the coarse grid of every tuned configuration, on
#           each tuning instance
#   fine    tasks/fine.txt: the fine grid around the best coarse value, from
#           the traces in traces/tune (skips, and lists, the configurations
#           without their 11 coarse runs)
#   select  tuned.json: the best value of each configuration on the tuning
#           instance of each dataset, from all the traces in traces/tune
#   eval    tasks/eval.txt: every configuration on each benchmark instance
#           that is not a tuning instance, at most RUNS_PER_JOB runs per job
#
# With a regex, each step only covers the configurations whose label matches
# it (e.g. '-tr' for the trust region variants): the task files get the
# suffix _<letters and digits of the regex> (tasks/coarse_tr.txt), and select
# updates their entries of tuned.json, keeping the others.
#
# Gurobi tokens: the jobs of fgm_ch, whose pricing MILPs are not solved by the
# dynamic program, use one Gurobi environment (license token) per thread,
# THREADS; the other jobs use one (see methods/common.jl and bench.sh). Each
# task file is therefore split: <file>.txt for the jobs with one token and
# <file>_mip.txt for those of fgm_ch. Within a file, the jobs are sorted by
# decreasing estimated time (the longest first, which shortens the makespan
# when the number of running jobs is capped): the build time of the oracle,
# 20 s of warm-up per method, and for each run its time in the earlier traces
# (traces/*/ or traces_old/*/<run or label>/<name>.json), at most 300 s, or
# 300 s without one.
# The step prints, for a budget of TOKENS tokens (60 by default, environment
# variable), the caps %M of the two job arrays that use at most TOKENS tokens
# and run at most MAX_JOBS jobs at a time (48 by default: the CPU limit per
# user of NIC6, 384 cores, is 48 jobs of 8 cores), and minimize the estimated
# makespan max(job-hours / cap).
#
# The best value is the one with the smallest final relative gap
# (L* - best bound) / |L*|, counting gaps below 1e-6 as 1e-6, and then the
# shortest run.

using JSON
include("configs.jl")

const TOL = 1e-6

function score(file)
    r = JSON.parsefile(file)
    r["status"] == "error" && return (Inf, Inf)
    bounds = [something(b, -Inf) for b in r["bound"]]
    isempty(bounds) && return (Inf, Inf)
    gap = (r["L_star"] - maximum(bounds)) / abs(r["L_star"])
    return (max(gap, TOL), r["time"][end])
end

# Value and score of each tuning run of configuration c on instance name
function tuning_runs(c::Config, name)
    runs = Tuple{Float64,Tuple{Float64,Float64}}[]
    isdir("traces/tune") || return runs
    for dir in readdir("traces/tune")
        startswith(dir, c.label * "@") || continue
        file = joinpath("traces/tune", dir, "$name.json")
        isfile(file) && push!(runs, (parse(Float64, split(dir, '@')[2]), score(file)))
    end
    return runs
end

best_value(c, name) = argmin(last, tuning_runs(c, name))[1]

const SELECTED = length(ARGS) >= 2 ? filter(c -> occursin(Regex(ARGS[2]), c.label), CONFIGS) : CONFIGS

const SUFFIX = length(ARGS) >= 2 ? "_" * replace(ARGS[2], r"[^A-Za-z0-9]" => "") : ""

tuned_configs() = filter(c -> c.tuned !== nothing, SELECTED)

const THREADS = 8                      # cpus-per-task of bench.sh
const TOKENS = parse(Int, get(ENV, "TOKENS", "60"))
const MAX_JOBS = parse(Int, get(ENV, "MAX_JOBS", "48"))
# NIC6 accepts at most 400 submitted jobs per user (MaxSubmitPU, running and
# pending): with 27 runs per job, the 81 configurations other than fgm_ch make
# 3 jobs per evaluation instance (237 jobs, at most 27 × 300 s ≈ 2.3 h each,
# within the 4 h of bench.sh), plus the 79 jobs of fgm_ch
const RUNS_PER_JOB = 27
const MAX_SUBMIT = 400

label_of(run) = String(split(run, '@')[1])
is_mip(line) = any(r -> CONFIG[label_of(r)].method == "fgm_ch", split(line)[2:end])

# Time of a run in the earlier traces, at most 300 s
function trace_time(name, run)
    for root in ("traces", "traces_old"), stage in ("eval_gurobi", "eval", "tune"),
        dir in unique([run, label_of(run)])
        file = joinpath(root, stage, dir, "$name.json")
        isfile(file) || continue
        t = try JSON.parsefile(file)["time"] catch; nothing end
        t isa Vector && !isempty(t) && return min(Float64(t[end]), 300.0)
    end
    return 300.0
end

function estimate(line)
    name, runs = split(line)[1], split(line)[2:end]
    file = joinpath("results", "$name.json")
    build = isfile(file) ? Float64(get(JSON.parsefile(file), "build_time", 60.0)) : 60.0
    methods = unique(CONFIG[label_of(r)].method for r in runs)
    return build + 20 * length(methods) + sum(r -> trace_time(name, r), runs)
end

function write_tasks(file, lines)
    mkpath(dirname(file))
    files = [(file, filter(!is_mip, lines), 1), (replace(file, ".txt" => "_mip.txt"), filter(is_mip, lines), THREADS)]
    hours = Dict{String,Float64}()
    for (f, ls, _) in files
        isempty(ls) && continue
        t = estimate.(ls)
        order = sortperm(t; rev = true)
        open(io -> foreach(l -> println(io, l), ls[order]), f, "w")
        hours[f] = sum(t) / 3600
        println("$f: $(length(ls)) jobs, estimated $(round(hours[f]; digits = 1)) job-hours")
    end
    # caps (one-token jobs, MIP jobs) within the budget, with the least makespan
    (f1, _, _), (f2, _, _) = files
    h1, h2 = get(hours, f1, 0.0), get(hours, f2, 0.0)
    best = (Inf, 0, 0)
    for m in (h2 > 0 ? (1:TOKENS÷THREADS) : 0:0)
        c = h1 > 0 ? min(TOKENS - THREADS * m, MAX_JOBS - m) : 0
        c < (h1 > 0) && continue
        makespan = max(h1 > 0 ? h1 / c : 0.0, h2 > 0 ? h2 / m : 0.0)
        makespan < best[1] && (best = (makespan, c, m))
    end
    makespan, c, m = best
    h1 > 0 && println("  sbatch --array=1-$(countlines(f1))%$c bench.sh $f1 ...   (1 token per job)")
    h2 > 0 && println("  sbatch --array=1-$(countlines(f2))%$m bench.sh $f2 ...   ($THREADS tokens per job)")
    println("  estimated makespan $(round(makespan; digits = 1)) h with $TOKENS tokens and $MAX_JOBS jobs")
    n = sum(countlines(f) for (f, _, _) in files if haskey(hours, f))
    n > MAX_SUBMIT && println("  WARNING: $n jobs, more than the $MAX_SUBMIT submitted jobs allowed per user")
end

function main(step)
    if step == "coarse"
        lines = [
            join([name; ["$(c.label)@$v" for v in coarse_grid(c)]], " ") for
            name in TUNING for c in tuned_configs()
        ]
        write_tasks("tasks/coarse$SUFFIX.txt", lines)
    elseif step == "fine"
        lines = String[]
        for name in TUNING, c in tuned_configs()
            done = first.(tuning_runs(c, name))
            if length(done) < 11
                println("skipped $(c.label) on $name: $(length(done)) of 11 coarse runs")
                continue
            end
            runs = ["$(c.label)@$v" for v in fine_grid(c, best_value(c, name))]
            isempty(runs) || push!(lines, join([name; runs], " "))
        end
        write_tasks("tasks/fine$SUFFIX.txt", lines)
    elseif step == "select"
        tuned = length(ARGS) >= 2 && isfile("tuned.json") ? JSON.parsefile("tuned.json") : Dict()
        for name in TUNING
            entries = get!(tuned, dataset(name), Dict())
            merge!(entries, Dict(c.label => best_value(c, name) for c in tuned_configs()))
        end
        open(io -> JSON.print(io, tuned, 2), "tuned.json", "w")
        for name in TUNING, c in tuned_configs()
            v = best_value(c, name)
            gap, seconds = Dict(tuning_runs(c, name))[v]
            println(rpad(dataset(name), 16), rpad(c.label, 28), rpad("$(c.tuned) = $v", 22),
                "gap $(round(gap; sigdigits = 3)) in $(round(seconds; digits = 1)) s")
        end
    elseif step == "eval"
        names = setdiff(benchmark_instances(; tol = TOL), TUNING)
        labels = [c.label for c in SELECTED]
        mip = filter(l -> CONFIG[l].method == "fgm_ch", labels)
        lines = [
            join([name; chunk], " ") for name in names for
            group in (setdiff(labels, mip), mip) for chunk in Iterators.partition(group, RUNS_PER_JOB)
        ]
        println("$(length(names)) instances, $(length(labels)) configurations")
        write_tasks("tasks/eval$SUFFIX.txt", lines)
    else
        error("unknown step $step")
    end
end

main(ARGS[1])
