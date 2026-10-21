# Runs configurations of configs.jl on one instance, one after the other, from
# the LP relaxation duals, each until the time limit or until L(π) is within
# the relative gap 1e-6 of the optimal value found by dw.jl. The time to build
# the oracle is excluded, and each method is first run once for 20 s, not
# recorded, so that compilation is not timed either.
#
# Usage: julia -t <threads> --project=. run.jl <output dir> <name> <time limit (s)> <run> [<run> ...]
#
# A run is `<label>@<value>`, with the relative value of the tuned parameter
# (see configs.jl), or `<label>`, with the value of tuned.json for the dataset
# of the instance (for configurations with a tuned parameter). Reads
# instances/<name>.json.gz and results/<name>.json; writes
# <output dir>/<run>/<name>.json. A run that fails is recorded with status
# "error" and the next runs go on.

using UnitCommitment, HiGHS, JSON
include("methods/common.jl")
include("methods/subgradient.jl")
include("methods/smoothing.jl")
include("methods/bundle.jl")
include("methods/primal.jl")
include("configs.jl")

const METHODS = Dict(
    "subgradient" => subgradient,
    "subgradient_last" => subgradient_last,
    "polyak_estimated" => polyak_estimated,
    "d_adaptation" => d_adaptation,
    "dowg" => dowg,
    "fgm_ch" => fgm_ch,
    "blm" => blm,
    "bplm" => bplm,
    "bpm" => bpm,
    "cg" => cg,
)

const TOL = 1e-6
const WARMUP = 20.0

# Parameters of a run: the fixed ones, the tuned one times its scale, and for
# subgradient_last the number of iterations N that fills the time limit, from
# the time of one evaluation of L (the second one, after compilation)
function parameters(c::Config, value, oracle, reference, time_limit)
    p = copy(c.fixed)
    if c.tuned !== nothing
        P = maximum(abs, oracle.π0)
        F = abs(reference["dual_value_at_lp_duals"])
        scale = Dict(:one => 1.0, :P => P, :F => F, :P2F => P^2 / F)[c.scale]
        p[c.tuned] = value * scale
    end
    if c.method == "subgradient_last"
        seconds = map(1:2) do _
            start = time()
            evaluate!(Trace(oracle.π0), oracle, oracle.π0, Stop(; time_limit = Inf))
            return time() - start
        end
        p[:N] = max(1, floor(Int, time_limit / seconds[2]))
    end
    return p
end

function main(out, name, time_limit, runs)
    reference = JSON.parsefile(joinpath("results", "$name.json"))
    reference["status"] == "optimal" || error("$name: no optimal value")
    L_star = reference["lower_bound"]
    tuned = isfile("tuned.json") ? JSON.parsefile("tuned.json") : Dict()
    optimizer = optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "mip_rel_gap" => 1e-9,
        "threads" => 1,
    )
    instance = UC.read(
        joinpath("instances", "$name.json.gz");
        extensions = [UC.ShiftFactorsTransmissionExt(lazy = false)],
    )
    oracle = UC.ConvexHullOracle(instance; optimizer)
    stop = Stop(; time_limit, target = L_star - TOL * abs(L_star))
    warm = Set{String}()
    for run in runs
        label, value = occursin('@', run) ? split(run, '@') : (run, nothing)
        c = CONFIG[label]
        value =
            c.tuned === nothing ? nothing :
            value === nothing ? tuned[dataset(name)][label] : parse(Float64, value)
        method = METHODS[c.method]
        result = Dict{String,Any}(
            "name" => name,
            "label" => label,
            "method" => c.method,
            "tuned_value" => value,
            "tol" => TOL,
            "threads" => Threads.nthreads(),
            "L_star" => L_star,
        )
        try
            p = parameters(c, value, oracle, reference, time_limit)
            result["parameters"] = Dict(string(k) => v for (k, v) in p)
            if !(c.method in warm)
                method(oracle, oracle.π0, Stop(; time_limit = WARMUP); p...)
                push!(warm, c.method)
            end
            trace = method(oracle, oracle.π0, stop; p...)
            merge!(
                result,
                Dict(
                    "status" => trace.status,
                    "time" => trace.time,
                    "value" => trace.value,
                    "bound" => trace.bound,
                    "upper" => trace.upper,
                    "best_prices" => trace.best_π,
                ),
            )
        catch e
            e isa InterruptException && rethrow()
            result["status"] = "error"
            result["error"] = sprint(showerror, e)
        end
        println(run, ": ", result["status"])
        flush(stdout)
        file = joinpath(out, run, "$name.json")
        mkpath(dirname(file))
        open(io -> JSON.print(io, result), file, "w")
    end
    return
end

main(ARGS[1], ARGS[2], parse(Float64, ARGS[3]), ARGS[4:end])
