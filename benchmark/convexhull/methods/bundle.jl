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

struct Bundle
    model::Model
    π::Vector{VariableRef}
    θ::Vector{VariableRef}
    members::Vector{Vector{Int}}
end

# Model over the box, with the clusters set from the first evaluation `r`
function Bundle(oracle, r, cuts; radius)
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
    return Bundle(model, π, θ, members)
end

function add_cuts!(B::Bundle, r)
    for k in eachindex(B.θ)
        c, a = 0.0, spzeros(length(B.π))
        for b in B.members[k]
            c += r.blocks[b].cost
            a += r.blocks[b].a
        end
        @constraint(
            B.model,
            B.θ[k] + sum(v * B.π[j] for (j, v) in zip(findnz(a)...); init = 0) <= c
        )
    end
    return
end

# Maximum of the model over the box, an upper bound on max L
function model_max!(B::Bundle)
    @objective(B.model, Max, sum(B.θ))
    optimize!(B.model)
    return objective_value(B.model)
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

# BLM and BPLM. The level is ℓ = U - α (U - L_best), where U is the maximum of
# the model; the next price is the projection of the center onto
# {π : model(π) ≥ ℓ}. For BLM the center is the last price. For BPLM it is the
# best price, and the level is only lowered once the gap U - L_best has
# decreased by the factor 1 - α.
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
)
    trace = Trace(π0)
    P = Preconditioner(precondition, length(π0))
    B, π, gap = nothing, π0, 1e-9
    level, Δ_ref = -Inf, Inf
    level_con = nothing
    while true
        r = evaluate!(trace, oracle, π, stop; gap)
        r.stop && break
        B === nothing && (B = Bundle(oracle, r, cuts; radius))
        add_cuts!(B, r)
        w = update!(P, r.grad)
        L_best = maximum(trace.bound)
        level_con === nothing || delete(B.model, level_con)
        U = model_max!(B)
        trace.U = min(trace.U, U)
        Δ = U - L_best
        if Δ <= 1e-12 * abs(L_best)
            trace.status = "optimal"
            break
        end
        new_level = U - α * Δ
        if !proximal || Δ < (1 - α) * Δ_ref
            level, Δ_ref = new_level, Δ
        else
            level = min(max(level, new_level), U)
        end
        center = proximal ? trace.best_π : π
        level_con = @constraint(B.model, sum(B.θ) >= level)
        @objective(
            B.model,
            Min,
            0.5 * sum(w[j] * (B.π[j] - center[j])^2 for j in eachindex(center))
        )
        optimize!(B.model)
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
    B = Bundle(oracle, r, cuts; radius)
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
        optimize!(B.model)
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
