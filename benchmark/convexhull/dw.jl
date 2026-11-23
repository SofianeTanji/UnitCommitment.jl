# Optimal value of the convex hull pricing problem max_π L(π) of one instance,
# by Dantzig–Wolfe column generation on UnitCommitment.ConvexHullOracle, with
# Wentges smoothing of the master duals.
#
# The master is the LP of the system block (buses, network, shortfalls) with
# the coupling rows, plus one convex combination of columns per unit. Penalized
# slacks on the coupling rows keep the first masters feasible. Its value z is
# an upper bound on max L. The columns are found at the smoothed prices
#
#     π = α π_center + (1 - α) π_master,
#
# where π_center are the best prices so far; L(π) is evaluated there, which
# gives the lower bound. If no column improves the master at π_master
# (mispricing), the pricing is repeated at π_master. All blocks are solved in
# parallel (julia -t <threads>). The prices with the best lower bound are the
# convex hull prices.
#
# The blocks and the master are solved by Gurobi (see methods/common.jl), with
# one environment per thread. With a warm start file (the result of an
# earlier run), the blocks are also priced at its convex hull prices before
# the first master, and these prices are the first center if they give a
# better bound than the LP duals; only the prices are reused, and all the
# bounds are recomputed.
#
# Usage: julia -t <threads> --project=. dw.jl <name> <instances dir> <results dir> <time limit (s)> [<tol> [<warm start dir>]]
# Reads <instances dir>/<name>.json.gz (and <warm start dir>/<name>.json) and
# writes <results dir>/<name>.json. Stops at the relative gap <tol> (default
# 1e-6).

using UnitCommitment, JuMP, JSON, SparseArrays
include("methods/common.jl")

function dw(path; time_limit, tol = 1e-6, α = 0.5, warm = nothing)
    start = time()
    instance = UC.read(
        path;
        extensions = [UC.ShiftFactorsTransmissionExt(lazy = false)],
    )
    oracle = use_gurobi_envs!(UC.ConvexHullOracle(instance; optimizer = mip_optimizer()))
    build_time = time() - start
    n = length(oracle.keys)
    blocks = oracle.blocks
    units = 1:(length(blocks)-1)
    sys = blocks[end]
    lp_value = sum(value(v -> b.x_lp[v], b.cost) for b in blocks)

    # Solves all blocks at π; returns their results and the certified lower
    # bound on L(π)
    function price(π)
        results = Vector{Any}(undef, length(blocks))
        function solve(b)
            r = UC.solve_block!(oracle, b, π)
            results[b] = (; r..., cost = value(blocks[b].cost))
            return
        end
        K = length(GUROBI_ENVS) # one task per environment, see methods/common.jl
        @sync for k in 1:K
            Threads.@spawn foreach(solve, k:K:length(blocks))
        end
        return results, sum(r.bound for r in results)
    end

    # Master: a copy of the system block model, with the coupling rows. The
    # oracle deleted these rows but left them registered, which copy_model
    # does not accept.
    for name in UC._CH_ROWS
        haskey(object_dictionary(sys.model), name) &&
            unregister(sys.model, name)
    end
    master, ref = copy_model(sys.model)
    set_optimizer(master, lp_optimizer())
    M = 1e3 * (1 + maximum(abs, oracle.π0))
    s_plus = @variable(master, [1:n], lower_bound = 0)
    s_minus = @variable(master, [1:n], lower_bound = 0)
    g = [AffExpr() for _ in 1:n]
    for (i, gi) in zip(sys.rows, sys.g)
        add_to_expression!(g[i], ref[gi])
    end
    rows = [
        oracle.lower[i] == 0 ? @constraint(master, g[i] + s_plus[i] >= 0) :
        @constraint(master, g[i] + s_plus[i] - s_minus[i] == 0) for i in 1:n
    ]
    convexity = [@constraint(master, 0 == 1) for _ in units]
    @objective(master, Min, ref[sys.cost] + M * (sum(s_plus) + sum(s_minus)))

    # Adds the unit columns with a negative reduced cost at (π, μ), or all of
    # them if μ is nothing; returns the number of columns added
    function add_columns!(results, π, μ)
        added = 0
        for b in units
            r = results[b]
            if μ !== nothing
                reduced_cost =
                    r.cost - sum(
                        π[i] * v for (i, v) in zip(findnz(r.a)...);
                        init = 0.0,
                    ) - μ[b]
                reduced_cost < -1e-9 * max(1.0, abs(r.cost)) || continue
            end
            λ = @variable(master, lower_bound = 0)
            set_normalized_coefficient(convexity[b], λ, 1.0)
            for (i, v) in zip(findnz(r.a)...)
                set_normalized_coefficient(rows[i], λ, v)
            end
            set_objective_coefficient(master, λ, r.cost)
            added += 1
        end
        return added
    end

    results, lower = price(oracle.π0)
    L0 = lower
    center = oracle.π0
    add_columns!(results, nothing, nothing)
    warm_bound = nothing
    if warm !== nothing
        string.(oracle.keys) == warm["keys"] || error("warm start: other coupling rows")
        π_warm = Float64.(warm["convex_hull_prices"])
        results, warm_bound = price(π_warm)
        add_columns!(results, nothing, nothing)
        warm_bound > lower && ((lower, center) = (warm_bound, π_warm))
    end

    status = "time_limit"
    upper, gap, iterations, pricings, artificials = Inf, Inf, 0, 1, NaN
    history = []
    while time() - start < time_limit
        iterations += 1
        JuMP.optimize!(master)
        termination_status(master) == OPTIMAL ||
            error("master: $(termination_status(master))")
        upper = objective_value(master)
        π_master = dual.(rows)
        μ = dual.(convexity)
        artificials = sum(value.(s_plus)) + sum(value.(s_minus))
        gap = (upper - lower) / max(1.0, abs(upper))
        push!(
            history,
            (; iteration = iterations, time = time() - start, lower, upper),
        )
        if gap <= tol
            status = "optimal"
            break
        end
        for π in (α .* center .+ (1 - α) .* π_master, π_master)
            results, L = price(π)
            pricings += 1
            if L > lower
                lower, center = L, π
            end
            add_columns!(results, π_master, μ) > 0 && break
        end
    end
    return Dict(
        "status" => status,
        "lower_bound" => lower,
        "upper_bound" => upper,
        "gap" => gap,
        "iterations" => iterations,
        "pricings" => pricings,
        "artificials" => artificials,
        "lp_value" => lp_value,
        "dual_value_at_lp_duals" => L0,
        "units" => length(units),
        "prices" => n,
        "threads" => Threads.nthreads(),
        "solver" => "gurobi",
        "tol" => tol,
        "warm_start_bound" => warm_bound,
        "build_time" => build_time,
        "total_time" => time() - start,
        "keys" => [string(k) for k in oracle.keys],
        "lp_duals" => oracle.π0,
        "convex_hull_prices" => center,
        "history" => history,
    )
end

function main(name, instances_dir, results_dir, time_limit, tol = 1e-6, warm_dir = nothing)
    out = joinpath(results_dir, "$name.json")
    mkpath(dirname(out))
    warm_file = warm_dir === nothing ? nothing : joinpath(warm_dir, "$name.json")
    warm = warm_file !== nothing && isfile(warm_file) ? JSON.parsefile(warm_file) : nothing
    warm !== nothing && get(warm, "status", "") == "error" && (warm = nothing)
    result = try
        dw(joinpath(instances_dir, "$name.json.gz"); time_limit, tol, warm)
    catch e
        Dict("status" => "error", "error" => sprint(showerror, e))
    end
    result["name"] = name
    open(io -> JSON.print(io, result), out, "w")
    return
end

main(ARGS[1:3]..., parse.(Float64, ARGS[4:min(end, 5)])..., ARGS[6:end]...)
