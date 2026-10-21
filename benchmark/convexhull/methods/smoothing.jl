# Fast gradient method on the smoothing of L over the convex hull of each
# block (FGM-CH):
#
#     L_σ(π) = Σ_b min { cost_b(x) - π' g_b(x) + (σ/2) ‖x - x_lp‖² : x ∈ conv X_b },
#
# where x are all the block variables and x_lp their values in the LP
# relaxation. L_σ is smooth, with ∇L_σ(π) = -Σ_b g_b(x̄_b) at the minimizers
# x̄_b. Blocks without integer variables are solved directly as QPs, by
# Clarabel. For the other blocks, x̄_b is found by column generation: a restricted master QP over
# convex combinations of stored block solutions, and a pricing MILP at the
# linearization of the prox term, until the certified Frank-Wolfe gap is at
# most τ (or after max_rounds rounds). The columns are kept between calls.
#
# The iterates are π_{k+1} = clamp(y_k + η ∇L_σ(y_k), box) and y_{k+1} =
# π_{k+1} + (k - 1) / (k + 2) (π_{k+1} - π_k). L is evaluated at π_k only for
# the trace, outside the run time.

# Column generation state of one block with integer variables: columns V_j
# (values of the block variables x), their cost and coupling shares a_j, and
# the Gram matrix Q = V'V of the prox term
mutable struct SmoothBlock
    x::Vector{VariableRef}
    position::Dict{VariableRef,Int}
    xr::Vector{Float64}
    integer::Vector{Bool}
    master::Model
    λ::Vector{VariableRef}
    convexity::Union{Nothing,ConstraintRef}
    V::Vector{Vector{Float64}}
    cost::Vector{Float64}
    a::Vector{SparseVector{Float64,Int}}
    Q::Matrix{Float64}
end

function SmoothBlock(oracle, b)
    blk = oracle.blocks[b]
    x = collect(keys(blk.x_lp))
    master = Model(qp_optimizer())
    set_silent(master)
    return SmoothBlock(
        x,
        Dict(zip(x, eachindex(x))),
        [blk.x_lp[v] for v in x],
        [is_binary(v) || is_integer(v) for v in x],
        master,
        VariableRef[],
        nothing,
        Vector{Float64}[],
        Float64[],
        SparseVector{Float64,Int}[],
        zeros(0, 0),
    )
end

# Adds the current solution of the block model as a column
function add_column!(S::SmoothBlock, oracle, b)
    blk = oracle.blocks[b]
    v = value.(S.x)
    v[S.integer] .= round.(v[S.integer])
    any(==(v), S.V) && return false
    at_v = var -> v[S.position[var]]
    push!(S.V, v)
    push!(S.cost, value(at_v, blk.cost))
    push!(S.a, sparsevec(blk.rows, value.(at_v, blk.g), length(oracle.keys)))
    λ = @variable(S.master, lower_bound = 0)
    push!(S.λ, λ)
    if S.convexity === nothing
        S.convexity = @constraint(S.master, λ == 1)
    else
        set_normalized_coefficient(S.convexity, λ, 1.0)
    end
    k = length(S.V)
    Q = zeros(k, k)
    Q[1:k-1, 1:k-1] .= S.Q
    for j in 1:k
        Q[j, k] = Q[k, j] = dot(S.V[j], v)
    end
    S.Q = Q
    return true
end

# Share of a block without integer variables: its QP with the prox term,
# solved by Clarabel on a copy of the block model (the HiGHS QP solver fails
# on some of these blocks)
function continuous_share(S, oracle, b, π, σ)
    blk = oracle.blocks[b]
    obj = copy(blk.cost)
    for (i, gi) in zip(blk.rows, blk.g)
        add_to_expression!(obj, -π[i], gi)
    end
    prox = (σ / 2) * sum((S.x[i] - S.xr[i])^2 for i in eachindex(S.x))
    @objective(blk.model, Min, obj + prox)
    qp = MOI.instantiate(Clarabel.Optimizer; with_bridge_type = Float64)
    MOI.set(qp, MOI.Silent(), true)
    map = MOI.copy_to(qp, backend(blk.model))
    MOI.optimize!(qp)
    status = MOI.get(qp, MOI.TerminationStatus())
    status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("block $(blk.name): $status")
    x = v -> MOI.get(qp, MOI.VariablePrimal(), map[index(v)])
    return sparsevec(blk.rows, value.(x, blk.g), length(oracle.keys))
end

# Share -∇ of block b in ∇L_σ(π), i.e. g_b(x̄_b)
function smooth_share!(S, oracle, b, π, σ, τ, max_rounds)
    any(S.integer) || return continuous_share(S, oracle, b, π, σ)
    if isempty(S.V)
        UC.solve_block!(oracle, b, π)
        add_column!(S, oracle, b)
    end
    λ = Float64[]
    for iteration in 1:max_rounds
        k = length(S.V)
        linear = [S.cost[j] - dot(π, S.a[j]) for j in 1:k]
        @objective(
            S.master,
            Min,
            sum((linear[j] - σ * dot(S.xr, S.V[j])) * S.λ[j] for j in 1:k) +
            (σ / 2) * sum(S.Q[i, j] * S.λ[i] * S.λ[j] for i in 1:k, j in 1:k)
        )
        optimize!(S.master)
        λ = max.(value.(S.λ), 0.0)
        λ ./= sum(λ)
        x̄ = sum(λ[j] .* S.V[j] for j in 1:k)
        d = σ .* (x̄ .- S.xr)
        r = UC.solve_block!(
            oracle,
            b,
            π;
            extra = sum(d[i] * S.x[i] for i in eachindex(S.x)),
        )
        δ = dot(λ, linear) + dot(d, x̄) - r.bound
        δ <= τ && break
        iteration < max_rounds && add_column!(S, oracle, b) || break
    end
    return sum(λ[j] .* S.a[j] for j in eachindex(λ))
end

function fgm_ch(
    oracle,
    π0,
    stop::Stop;
    σ,
    η,
    τ = 1e-7,
    max_rounds = 50,
    radius = 1.0,
)
    trace = Trace(π0)
    lo, hi = price_box(oracle; radius)
    S = [SmoothBlock(oracle, b) for b in eachindex(oracle.blocks)]
    r = evaluate!(trace, oracle, π0, stop; timed = false)
    r.stop && return trace
    π_prev, y = π0, π0
    for k in Iterators.countfrom(1)
        shares = Vector{Any}(undef, length(S))
        Threads.@threads :greedy for b in eachindex(S)
            shares[b] = smooth_share!(S[b], oracle, b, y, σ, τ, max_rounds)
        end
        π = clamp.(y .- η .* Vector(sum(shares)), lo, hi)
        r = evaluate!(trace, oracle, π, stop; timed = false)
        r.stop && break
        y = π .+ (k - 1) / (k + 2) .* (π .- π_prev)
        π_prev = π
    end
    return trace
end
