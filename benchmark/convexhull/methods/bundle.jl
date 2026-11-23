# Bundle methods: the bundle level method (BLM), the bundle proximal level
# method (BPLM) and the proximal bundle method (BPM), on a cutting-plane model
# of L over the price box. The three share these options:
#
# - `cuts`: the blocks are grouped into `cuts` clusters, with one model per
#   cluster (1: aggregated, 0: one model per block). A block goes to the
#   cluster of the period at which its coupling coefficients are largest.
# - `precondition`: the quadratic terms use the weights w_j = clamp((rms_j +
#   1e-8)^-2, 1e-4, 1e4), where rms is an exponential moving average (factor
#   0.9) of the squared supergradients.
# - `trust_region` (BLM and BPLM): the projection and the level are computed
#   on a trust region around the center instead of the whole box, with one
#   radius per coupling row (see TrustRegion).
# - `inexact`: if positive, the MIP blocks of the next evaluation are solved
#   with the relative gap clamp(inexact · Δ / |L_best|, 1e-9, max_gap), where
#   Δ is the current gap estimate of the method. The cuts remain valid (they
#   come from feasible block solutions), and L_best uses certified bounds.
#
# A solution x_b of block b at any price gives the cut
#
#     L_b(π) ≤ cost_b(x_b) - π' a_b,   a_b = g_b(x_b),
#
# and the model of a cluster is the minimum of the cuts of its blocks, summed.
#
# With Gurobi, the masters are solved by the dual simplex (Method = 1),
# warm-started from their last basis, instead of the default barrier, which
# restarts from scratch at each solve (with one model per block, the master
# is large: a cut per block per iteration). The LP of the maximum of the
# model and the QP of the projection are two models with the same cuts, so
# that each one is warm-started from its own last solve: warm-started from
# the LP basis, the simplex of Gurobi fails on the QP (dual infeasible or
# numerical error). A solve that fails is redone with the default method.

struct Bundle
    model::Model
    π::Vector{VariableRef}
    θ::Vector{VariableRef}
    members::Vector{Vector{Int}}
    lo::Vector{Float64}
    hi::Vector{Float64}
    lp::Model                          # LP of model_max! (B.model without it)
    lpπ::Vector{VariableRef}
    lpθ::Vector{VariableRef}
end

# Solves m (B.model or B.lp), again with the default method if it fails
function solve_master!(m)
    set_attribute(m, "Method", 1)
    optimize!(m)
    if termination_status(m) != MOI.OPTIMAL
        set_attribute(m, "Method", -1)
        optimize!(m)
    end
    return
end

# Model over the box, with the clusters set from the first evaluation `r`
# (with `lp`, a second model for model_max!)
function Bundle(oracle, r, cuts; radius, lp = true)
    lo, hi = price_box(oracle; radius)
    model = Model(qp_optimizer())
    set_silent(model)
    π = @variable(model, lo[j] <= π[j = eachindex(lo)] <= hi[j])
    n_blocks = length(oracle.blocks)
    K = cuts == 0 ? n_blocks : cuts
    T = maximum(last(key[2]) for key in oracle.keys)
    cluster = map(1:n_blocks) do b
        cuts == 0 && return b
        a = r.blocks[b].a
        nnz(a) == 0 && return 1
        t = last(oracle.keys[findnz(a)[1][argmax(abs.(findnz(a)[2]))]][2])
        return clamp(cld(K * t, T), 1, K)
    end
    members = filter(!isempty, [findall(==(k), cluster) for k in 1:K])
    θ = @variable(model, [eachindex(members)])
    lp_model, lpπ, lpθ = model, π, θ
    if lp
        lp_model = Model(qp_optimizer())
        set_silent(lp_model)
        lpπ = @variable(lp_model, lo[j] <= lpπ[j = eachindex(lo)] <= hi[j])
        lpθ = @variable(lp_model, [eachindex(members)])
    end
    return Bundle(model, π, θ, members, lo, hi, lp_model, lpπ, lpθ)
end

function add_cuts!(B::Bundle, r)
    for k in eachindex(B.θ)
        c, a = 0.0, spzeros(length(B.π))
        for b in B.members[k]
            c += r.blocks[b].cost
            a += r.blocks[b].a
        end
        for (m, π, θ) in unique([(B.model, B.π, B.θ), (B.lp, B.lpπ, B.lpθ)])
            @constraint(m, θ[k] + sum(v * π[j] for (j, v) in zip(findnz(a)...); init = 0) <= c)
        end
    end
    return
end

# Maximum of the model over the box, an upper bound on max L
function model_max!(B::Bundle)
    @objective(B.lp, Max, sum(B.lpθ))
    solve_master!(B.lp)
    return objective_value(B.lp)
end

function set_region!(B::Bundle, lo, hi)
    for π in unique([B.π, B.lpπ]), j in eachindex(π)
        set_lower_bound(π[j], lo[j])
        set_upper_bound(π[j], hi[j])
    end
    return
end

# Trust region of the variable-metric level methods of the thesis (PC-BLM,
# PC-BPLM), with one radius δ_j per coupling row j: X = {π in the box :
# |π_j - c_j| ≤ δ_j} around the center c. The first step is over the whole box
# and sets every δ_j to its ∞-norm. After each step from c to π, a row is at
# the boundary if |π_j - c_j| ≥ 0.9 δ_j and the box does not cut the region
# in that direction. If the step improved L_best, δ_j doubles at the boundary
# and the counters are reset; otherwise the counter n_j of a row at the
# boundary is incremented, and δ_j is halved when it reaches 3.
mutable struct TrustRegion
    on::Bool
    δ::Vector{Float64}
    n::Vector{Int}
end

TrustRegion(on, n) = TrustRegion(on, fill(Inf, n), zeros(Int, n))

region(R::TrustRegion, B::Bundle, c) = max.(B.lo, c .- R.δ), min.(B.hi, c .+ R.δ)

function update!(R::TrustRegion, B::Bundle, c, π, improved)
    if all(isinf, R.δ)
        R.δ .= max(maximum(abs.(π .- c)), 1e-9 * (1 + maximum(abs, c)))
        return R
    end
    for j in eachindex(π)
        edge = π[j] >= c[j] ? B.hi[j] - c[j] : c[j] - B.lo[j]
        abs(π[j] - c[j]) >= 0.9 * R.δ[j] && R.δ[j] < edge || continue
        if improved
            R.δ[j] *= 2
        else
            R.n[j] += 1
            R.n[j] == 3 && (R.δ[j] /= 2; R.n[j] = 0)
        end
    end
    improved && fill!(R.n, 0)
    return R
end

# Weights of the quadratic terms
mutable struct Preconditioner
    on::Bool
    rms::Vector{Float64}
    w::Vector{Float64}
end

Preconditioner(on, n) = Preconditioner(on, ones(n), ones(n))

function update!(P::Preconditioner, g)
    P.on || return P.w
    @. P.rms = sqrt(0.9 * P.rms^2 + 0.1 * g^2)
    @. P.w = clamp((P.rms + 1e-8)^-2, 1e-4, 1e4)
    return P.w
end

inexact_gap(inexact, max_gap, Δ, L_best) =
    inexact > 0 ? clamp(inexact * Δ / abs(L_best), 1e-9, max_gap) : 1e-9

# BLM and BPLM. The next price is the projection of the center onto
# {π : model(π) ≥ ℓ}, where U is the maximum of the model. For BLM the center
# is the last price and ℓ = U - α (U - L_best). For BPLM the center is the best
# price and ℓ = L_best + β (U - L_best), with a variable β: 1 - α at the
# start, halved after each evaluation that does not improve L_best, doubled
# (up to 1 - α) after each one that does. With a fixed α, U - L_best
# overestimates the gap by a factor of about 2000 at the start (one cut, the
# whole price box) and the level is out of reach for many evaluations (17 on
# pglib-uc/ca/2015-06-01_reserves_1, 6 with the variable β); near the end, a
# small β would give tiny steps. With `trust_region`, U in the level is the
# maximum of the model over the trust region (a local bound, never recorded in
# the trace); while it is not above L_best, all the radii are doubled.
function level_method(
    oracle,
    π0,
    stop::Stop;
    proximal::Bool,
    α,
    cuts = 1,
    precondition = false,
    inexact = 0.0,
    max_gap = 1e-2,
    radius = 1.0,
    trust_region = false,
)
    trace = Trace(π0)
    P = Preconditioner(precondition, length(π0))
    R = TrustRegion(trust_region, length(π0))
    B, π, gap = nothing, π0, 1e-9
    β = 1 - α
    level_con, center = nothing, nothing
    while true
        L_prev = maximum(trace.bound; init = -Inf)
        r = evaluate!(trace, oracle, π, stop; gap)
        r.stop && break
        B === nothing && (B = Bundle(oracle, r, cuts; radius))
        add_cuts!(B, r)
        w = update!(P, r.grad)
        L_best = maximum(trace.bound)
        proximal && (β = r.bound > L_prev ? min(2β, 1 - α) : β / 2)
        R.on && center !== nothing && update!(R, B, center, π, r.bound > L_prev)
        R.on && set_region!(B, B.lo, B.hi)
        U = model_max!(B)
        trace.U = min(trace.U, U)
        Δ = U - L_best
        if Δ <= 1e-12 * abs(L_best)
            trace.status = "optimal"
            break
        end
        center = proximal ? trace.best_π : π
        U_k = U
        while R.on
            lo, hi = region(R, B, center)
            set_region!(B, lo, hi)
            U_k = model_max!(B)
            (U_k - L_best > 1e-12 * abs(L_best) || (lo == B.lo && hi == B.hi)) && break
            R.δ .*= 2
        end
        Δ_k = U_k - L_best
        level = proximal ? L_best + β * Δ_k : U_k - α * Δ_k
        # the level constraint is kept and only its right-hand side changes:
        # deleting and re-adding it loses the basis of the dual simplex (on
        # pglib-uc/ca/2015-06-01_reserves_1 with one model per block, 967
        # instead of 95 simplex iterations per projection, 3.2 times longer)
        if level_con === nothing
            level_con = @constraint(B.model, sum(B.θ) >= level)
        else
            set_normalized_rhs(level_con, level)
        end
        @objective(
            B.model,
            Min,
            0.5 * sum(w[j] * (B.π[j] - center[j])^2 for j in eachindex(center))
        )
        solve_master!(B.model)
        is_solved_and_feasible(B.model) && (π = value.(B.π))
        gap = inexact_gap(inexact, max_gap, Δ, L_best)
    end
    return trace
end

blm(oracle, π0, stop::Stop; kwargs...) =
    level_method(oracle, π0, stop; proximal = false, kwargs...)

bplm(oracle, π0, stop::Stop; kwargs...) =
    level_method(oracle, π0, stop; proximal = true, kwargs...)

# Proximal bundle method. The candidate maximizes model(π) - ‖π - y‖²_w / (2t)
# around the prox center y. With `test = :classic`, it becomes the new center
# (serious step) if L(π̃) - L(y) ≥ m (model(π̃) - L(y)); otherwise only its
# cuts are kept (null step). With `test = :ms`, the step is accepted if
# model(π̃) - L(π̃) ≤ κ ‖π̃ - y‖²_w / (2t) (Monteiro and Svaiter, 2013), and
# `accelerate` adds their momentum, with a restart when L decreases between
# accepted steps (Moreau-FGM; without it, Moreau-PP). In both cases t is
# multiplied by γ after a step accepted without a null step before it.
function bpm(
    oracle,
    π0,
    stop::Stop;
    t,
    γ = 2.0,
    test = :classic,
    m = 0.1,
    κ = 0.9,
    accelerate = false,
    restart = true,
    cuts = 1,
    precondition = false,
    inexact = 0.0,
    max_gap = 1e-2,
    radius = 1.0,
)
    test in (:classic, :ms) || error("unknown test $test")
    accelerate && test != :ms && error("acceleration needs test = :ms")
    trace = Trace(π0)
    P = Preconditioner(precondition, length(π0))
    r = evaluate!(trace, oracle, π0, stop)
    r.stop && return trace
    B = Bundle(oracle, r, cuts; radius, lp = false)
    add_cuts!(B, r)
    w = copy(update!(P, r.grad))
    t = Float64(t)
    x, L_x = π0, r.bound              # last accepted price and its value
    v, A = copy(π0), 0.0              # Monteiro-Svaiter sequence
    first_round, gap = true, 1e-9
    while true
        a = accelerate ? (t + sqrt(t^2 + 4 * t * A)) / 2 : t
        y = accelerate ? (A .* x .+ a .* v) ./ (A + a) : x
        @objective(
            B.model,
            Max,
            sum(B.θ) -
            sum(w[j] * (B.π[j] - y[j])^2 for j in eachindex(y)) / (2 * t)
        )
        solve_master!(B.model)
        π, model_π = value.(B.π), value(sum(B.θ))
        r = evaluate!(trace, oracle, π, stop; gap)
        r.stop && break
        add_cuts!(B, r)
        w_old, w = w, copy(update!(P, r.grad))
        accepted =
            test == :classic ? r.bound - L_x >= m * (model_π - L_x) :
            model_π - r.bound <=
            κ * sum(w_old .* (π .- y) .^ 2) / (2 * t) + 1e-9
        if accepted
            if accelerate
                v .+= a .* (π .- y) ./ t
                A += a
            end
            first_round && (t *= γ)
            if accelerate && restart && r.bound < L_x
                A = 0.0
                v .= π
            end
            x, L_x = π, r.bound
        end
        first_round = accepted
        gap = inexact_gap(inexact, max_gap, max(model_π - L_x, 0.0), L_x)
    end
    return trace
end
