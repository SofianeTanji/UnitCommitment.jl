# Analysis of the evaluation runs: checks the traces, then writes the tables and
# curves of the comparison of the configurations to <out dir>:
#
#   runs.csv         one row per run: final gap, time to the target, gap at
#                    CHECKPOINT seconds, and for the methods with an upper bound
#                    U the time at which U - L_best certifies the target
#   summary.csv      per dataset and over all instances: instances solved,
#                    capped geometric mean of the time to the target (TIME_LIMIT
#                    when not reached), speedup relative to BASELINE, mean final
#                    gap and mean gap at CHECKPOINT seconds
#   curves.csv       mean gap over the instances of each dataset, on a
#                    logarithmic time grid
#   switches.csv     speedup of each switch of the bundle methods (k4 or block
#                    instead of agg, preconditioning, inexact oracle): geometric
#                    mean, over the pairs of configurations that differ only by
#                    this switch, of the ratio of their capped geometric means
#   sensitivity.csv  worst-to-best ratio of the final gap over the coarse grid,
#                    on the tuning instance of each dataset (from traces/tune)
#
# The gap of a run at time t is (L* - L_best(t)) / |L*|, where L_best(t) is the
# best certified bound found by t, and L(π0) before the first evaluation. A run
# reaches the target when its gap is at most its tolerance (1e-6).
#
# Usage: julia --project=. analyze.jl [<traces dir> [<out dir>]], by default
# traces/eval and analysis.

using JSON, Statistics
include("configs.jl")

const TIME_LIMIT = 300.0
const CHECKPOINT = 100.0
const BASELINE = "blm-agg"
const GRID = round.(10.0 .^ range(-1, log10(TIME_LIMIT), 50); sigdigits = 4)

struct Run
    name::String
    dataset::String
    label::String
    status::String
    tol::Float64
    gap0::Float64
    time::Vector{Float64}
    gap::Vector{Float64}       # gap of the best bound by each evaluation
    certified::Vector{Float64} # (U - L_best) / |L*| at each evaluation
end

function Run(file, reference)
    r = JSON.parsefile(file)
    L_star = r["L_star"]
    relative = x -> x / abs(L_star)
    gap0 = relative(L_star - reference[r["name"]]["dual_value_at_lp_duals"])
    time = Float64.(get(r, "time", Float64[]))
    best = accumulate(max, [something(b, -Inf) for b in get(r, "bound", [])]; init = -Inf)
    upper = [something(u, Inf) for u in get(r, "upper", [])]
    return Run(
        r["name"],
        dataset(r["name"]),
        r["label"],
        r["status"],
        r["tol"],
        gap0,
        time,
        min.(relative.(L_star .- best), gap0),
        relative.(upper .- best),
    )
end

function gap_at(r::Run, t)
    k = searchsortedlast(r.time, t)
    return k == 0 ? r.gap0 : r.gap[k]
end

final_gap(r::Run) = gap_at(r, Inf)

first_time(r::Run, reached) = (k = findfirst(reached); k === nothing ? nothing : r.time[k])
time_to_target(r::Run) = first_time(r, r.gap .<= r.tol)
time_certified(r::Run) = first_time(r, r.certified .<= r.tol)

capped(t) = t === nothing ? TIME_LIMIT : min(t, TIME_LIMIT)
geomean(x) = exp(mean(log.(x)))

function write_csv(file, header, rows)
    open(file, "w") do io
        println(io, join(header, ","))
        foreach(row -> println(io, join(something.(row, ""), ",")), rows)
    end
    println("$file: $(length(rows)) rows")
end

# Reads every run, and prints what is missing or suspect: missing traces, runs
# that ended in an error, gaps below -1e-6 (L above L*), and tuned values that
# differ from tuned.json
function read_runs(dir)
    names = setdiff(benchmark_instances(), TUNING)
    reference = Dict(n => JSON.parsefile(joinpath("results", "$n.json")) for n in names)
    labels = filter(l -> haskey(CONFIG, l), readdir(dir))
    tuned = isfile("tuned.json") ? JSON.parsefile("tuned.json") : nothing
    runs, missing_runs = Run[], String[]
    for label in labels, name in names
        file = joinpath(dir, label, "$name.json")
        if !isfile(file)
            push!(missing_runs, "$label $name")
            continue
        end
        r = Run(file, reference)
        push!(runs, r)
        r.status == "error" && println("error: $label $name")
        minimum(r.gap; init = 0.0) < -1e-6 &&
            println("gap below -1e-6: $label $name ($(minimum(r.gap)))")
        c = CONFIG[label]
        if tuned !== nothing && c.tuned !== nothing
            value = JSON.parsefile(file)["tuned_value"]
            value == tuned[r.dataset][label] ||
                println("tuned value $value, not $(tuned[r.dataset][label]): $label $name")
        end
    end
    println("$(length(names)) instances, $(length(labels)) configurations, ",
        "$(length(runs)) runs, $(length(missing_runs)) missing")
    foreach(m -> println("missing: $m"), missing_runs)
    return runs, labels
end

# The groups of runs of each configuration: per dataset, and all instances
groups(runs, labels) = [
    (d, l, filter(r -> r.label == l && (d == "all" || r.dataset == d), runs)) for
    d in [unique(dataset.(TUNING)); "all"] for l in labels
]

function summary(runs, labels)
    times = Dict((d, l) => geomean(capped.(time_to_target.(g))) for (d, l, g) in groups(runs, labels) if !isempty(g))
    rows = []
    for (d, l, g) in groups(runs, labels)
        isempty(g) && continue
        base = get(times, (d, BASELINE), nothing)
        push!(rows, (
            d, l, count(r -> time_to_target(r) !== nothing, g), length(g),
            round(times[(d, l)]; sigdigits = 4),
            base === nothing ? nothing : round(base / times[(d, l)]; sigdigits = 3),
            round(mean(final_gap.(g)); sigdigits = 3),
            round(mean(gap_at.(g, CHECKPOINT)); sigdigits = 3),
        ))
    end
    return times, rows
end

# Bundle configurations by (base, cuts, preconditioning, inexact)
function bundle_key(label)
    c = CONFIG[label]
    haskey(c.fixed, :cuts) || return nothing
    base = first(split(label, r"-(agg|k4|block)"))
    return (base, c.fixed[:cuts], c.fixed[:precondition], c.fixed[:inexact] > 0)
end

function switches(times, labels)
    config = Dict(k => l for l in labels for k in (bundle_key(l),) if k !== nothing)
    changes = [
        ("k4 instead of agg", k -> k[2] == 1, k -> (k[1], 4, k[3], k[4])),
        ("block instead of agg", k -> k[2] == 1, k -> (k[1], 0, k[3], k[4])),
        ("preconditioning", k -> !k[3], k -> (k[1], k[2], true, k[4])),
        ("inexact oracle", k -> !k[4], k -> (k[1], k[2], k[3], true)),
    ]
    rows = []
    for d in [unique(dataset.(TUNING)); "all"], (name, off, on) in changes
        bases = unique(first.(keys(config)))
        for base in [bases; "all"]
            ratios = [
                times[(d, config[k])] / times[(d, config[on(k)])] for
                k in keys(config) if off(k) && (base == "all" || k[1] == base) &&
                haskey(config, on(k)) && haskey(times, (d, config[k])) &&
                haskey(times, (d, config[on(k)]))
            ]
            isempty(ratios) ||
                push!(rows, (d, name, base, round(geomean(ratios); sigdigits = 3), length(ratios)))
        end
    end
    return rows
end

function sensitivity()
    isdir("traces/tune") || return []
    reference = Dict(n => JSON.parsefile(joinpath("results", "$n.json")) for n in TUNING)
    rows = []
    for name in TUNING, c in filter(c -> c.tuned !== nothing, CONFIGS)
        gaps = Float64[]
        for v in coarse_grid(c)
            file = joinpath("traces/tune", "$(c.label)@$v", "$name.json")
            isfile(file) || continue
            r = Run(file, reference)
            r.status == "error" || push!(gaps, max(final_gap(r), r.tol))
        end
        isempty(gaps) && continue
        push!(rows, (dataset(name), c.label, round(maximum(gaps) / minimum(gaps); sigdigits = 3), length(gaps)))
    end
    return rows
end

function main(dir, out)
    mkpath(out)
    runs, labels = read_runs(dir)
    write_csv(
        joinpath(out, "runs.csv"),
        ["dataset", "name", "label", "status", "final_gap", "time_to_target",
            "gap_at_$(Int(CHECKPOINT))s", "time_certified"],
        [(r.dataset, r.name, r.label, r.status, final_gap(r), time_to_target(r),
            gap_at(r, CHECKPOINT), time_certified(r)) for r in runs],
    )
    times, rows = summary(runs, labels)
    write_csv(
        joinpath(out, "summary.csv"),
        ["dataset", "label", "solved", "instances", "geomean_time", "speedup_vs_$BASELINE",
            "mean_final_gap", "mean_gap_at_$(Int(CHECKPOINT))s"],
        rows,
    )
    write_csv(
        joinpath(out, "curves.csv"),
        ["dataset", "label", "time", "mean_gap"],
        [(d, l, t, mean(gap_at.(g, t))) for (d, l, g) in groups(runs, labels) if d != "all" && !isempty(g) for t in GRID],
    )
    write_csv(
        joinpath(out, "switches.csv"),
        ["dataset", "switch", "base", "speedup", "pairs"],
        switches(times, labels),
    )
    write_csv(
        joinpath(out, "sensitivity.csv"),
        ["dataset", "label", "worst_to_best", "values"],
        sensitivity(),
    )
    return
end

main(get(ARGS, 1, "traces/eval"), get(ARGS, 2, "analysis"))
