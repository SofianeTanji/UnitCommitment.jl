# Tuning and evaluation plan of the configurations of configs.jl. Each line of
# a task file is the instance and the runs of one job of bench.sh (see
# run.jl).
#
# Usage: julia --project=. plan.jl <step>, with the steps in this order:
#
#   coarse  tasks/coarse.txt: the coarse grid of every tuned configuration, on
#           each tuning instance
#   fine    tasks/fine.txt: the fine grid around the best coarse value, from
#           the traces in traces/tune
#   select  tuned.json: the best value of each configuration on the tuning
#           instance of each dataset, from all the traces in traces/tune
#   eval    tasks/eval.txt: every configuration on each benchmark instance
#           that is not a tuning instance, 12 runs per job
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

tuned_configs() = filter(c -> c.tuned !== nothing, CONFIGS)

function write_tasks(file, lines)
    mkpath(dirname(file))
    open(io -> foreach(l -> println(io, l), lines), file, "w")
    println("$file: $(length(lines)) jobs")
end

function main(step)
    if step == "coarse"
        lines = [
            join([name; ["$(c.label)@$v" for v in coarse_grid(c)]], " ") for
            name in TUNING for c in tuned_configs()
        ]
        write_tasks("tasks/coarse.txt", lines)
    elseif step == "fine"
        lines = String[]
        for name in TUNING, c in tuned_configs()
            done = first.(tuning_runs(c, name))
            length(done) == 11 || error("$(c.label) on $name: $(length(done)) coarse runs")
            runs = ["$(c.label)@$v" for v in fine_grid(c, best_value(c, name))]
            isempty(runs) || push!(lines, join([name; runs], " "))
        end
        write_tasks("tasks/fine.txt", lines)
    elseif step == "select"
        tuned = Dict(
            dataset(name) => Dict(c.label => best_value(c, name) for c in tuned_configs()) for
            name in TUNING
        )
        open(io -> JSON.print(io, tuned, 2), "tuned.json", "w")
        for name in TUNING, c in tuned_configs()
            v = best_value(c, name)
            gap, seconds = Dict(tuning_runs(c, name))[v]
            println(rpad(dataset(name), 16), rpad(c.label, 28), rpad("$(c.tuned) = $v", 22),
                "gap $(round(gap; sigdigits = 3)) in $(round(seconds; digits = 1)) s")
        end
    elseif step == "eval"
        names = setdiff(benchmark_instances(; tol = TOL), TUNING)
        labels = [c.label for c in CONFIGS]
        lines = [
            join([name; chunk], " ") for name in names for
            chunk in Iterators.partition(labels, 12)
        ]
        println("$(length(names)) instances, $(length(labels)) configurations")
        write_tasks("tasks/eval.txt", lines)
    else
        error("unknown step $step")
    end
end

main(ARGS[1])
