# Dantzig-Wolfe column generation. The master is the LP of the system block
# (buses, network, shortfalls) with the coupling rows, plus convex combinations
# of unit columns: one convexity row per unit with `multicut = true`, or a
# single one, whose columns are the sums of the unit columns found at one
# price, with `multicut = false` (the primal views of the disaggregated and
# aggregated cutting-plane methods). Penalized slacks on the coupling rows keep
# the first masters feasible. The master value is an upper bound on max L.
#
# The columns are priced at π = α π_center + (1 - α) π_master, where π_center
# are the best prices so far (Wentges smoothing; α = 0 is plain column
# generation). If no column improves the master at π_master (mispricing), the
# pricing is repeated at π_master.
function cg(oracle, π0, stop::Stop; multicut = true, α = 0.0)
    trace = Trace(π0)
    n = length(oracle.keys)
    blocks = oracle.blocks
    units = 1:(length(blocks)-1)
    sys = blocks[end]
    # The oracle deleted the coupling rows from the system block but left them
    # registered, which copy_model does not accept
    for name in UC._CH_ROWS
        haskey(object_dictionary(sys.model), name) &&
            unregister(sys.model, name)
    end
    master, ref = copy_model(sys.model)
    set_optimizer(master, lp_optimizer())
    M = 1e3 * (1 + maximum(abs, π0))
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
    convexity = [@constraint(master, 0 == 1) for _ in (multicut ? units : 1:1)]
    @objective(master, Min, ref[sys.cost] + M * (sum(s_plus) + sum(s_minus)))

    # Adds the columns of the pricing results r with a negative reduced cost
    # at (π, μ), or all of them if μ is nothing; returns how many were added
    function add_columns!(r, π, μ)
        columns =
            multicut ? [(b, r.blocks[b].cost, r.blocks[b].a) for b in units] :
            [(
                1,
                sum(r.blocks[b].cost for b in units),
                sum(r.blocks[b].a for b in units),
            )]
        added = 0
        for (k, cost, a) in columns
            if μ !== nothing
                reduced_cost = cost - dot(π, a) - μ[k]
                reduced_cost < -1e-9 * max(1.0, abs(cost)) || continue
            end
            λ = @variable(master, lower_bound = 0)
            set_normalized_coefficient(convexity[k], λ, 1.0)
            for (i, v) in zip(findnz(a)...)
                set_normalized_coefficient(rows[i], λ, v)
            end
            set_objective_coefficient(master, λ, cost)
            added += 1
        end
        return added
    end

    r = evaluate!(trace, oracle, π0, stop)
    r.stop && return trace
    add_columns!(r, nothing, nothing)
    while true
        JuMP.optimize!(master)
        termination_status(master) == OPTIMAL ||
            error("master: $(termination_status(master))")
        upper = objective_value(master)
        trace.U = min(trace.U, upper)
        π_master = dual.(rows)
        μ = dual.(convexity)
        if upper - maximum(trace.bound) <= 1e-12 * abs(upper)
            trace.status = "optimal"
            break
        end
        added = 0
        for π in unique([α .* trace.best_π .+ (1 - α) .* π_master, π_master])
            r = evaluate!(trace, oracle, π, stop)
            r.stop && return trace
            added = add_columns!(r, π_master, μ)
            added > 0 && break
        end
        if added == 0
            trace.status = "optimal"
            break
        end
    end
    return trace
end
