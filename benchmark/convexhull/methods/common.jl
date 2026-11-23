# Shared by all methods: the stopping rule, the trace of a run, the evaluation
# of L with all blocks solved in parallel, and the price box.
#
# A method is a function
#
#     method(oracle, π0, stop::Stop; parameters...) -> Trace
#
# that maximizes L(π) over π ≥ oracle.lower, starting from π0. It evaluates L
# through `evaluate!(trace, oracle, π, stop)`, which records the evaluation in
# the trace, and it returns as soon as that call sets `stop`.

using UnitCommitment, Gurobi, JuMP, LinearAlgebra, SparseArrays
const UC = UnitCommitment

# The oracle MIPs and the models of the methods are solved by Gurobi (HiGHS is
# not used: the presolve of HiGHS 1.15.1 can return a suboptimal block solution
# with status OPTIMAL and the same bound, on the unit g42 of or-lib/50_0_1_w).
#
# Each Gurobi environment holds a license token, so all the models share
# GUROBI_ENVS environments (the environment variable, Threads.nthreads() by
# default), created once: block b of the oracle uses environment env_slot(b,
# K) (see use_gurobi_envs!), and the models of the methods, solved between
# evaluations, use the first one. An environment must not be used by two
# threads at once, so evaluate! solves the MIP blocks of an environment one
# after the other, in one task. The blocks of the dynamic program use no
# environment and are solved by all the threads, so that the methods need
# one environment (bench.sh), and dw.jl, which solves every block by MIP, one
# per thread. The connection to the token server can fail for a while (Gurobi
# error 10009 on a node of NIC6), so each environment is tried for up to 10
# minutes before the job fails.
function gurobi_env(; attempts = 20, wait = 30)
    for attempt in 1:attempts
        try
            return Gurobi.Env(; output_flag = 0)
        catch e
            attempt == attempts && rethrow()
            @warn "Gurobi environment, attempt $attempt of $attempts: retrying in $wait s" exception = e
            sleep(wait)
        end
    end
end
const GUROBI_ENVS =
    [gurobi_env() for _ in 1:parse(Int, get(ENV, "GUROBI_ENVS", string(Threads.nthreads())))]
env_slot(b, K) = mod1(b, K)

# Stops after `time_limit` seconds, or once the certified value of L(π) is at
# least `target`
Base.@kwdef struct Stop
    time_limit::Float64
    target::Float64 = Inf
end

# The evaluations of a run: time since the start, L(π) (an upper estimate when
# the MIP blocks are solved with a gap), a certified lower bound on L(π), and
# the upper bound U on max L known to the method at that time (Inf if it has
# none; methods update U); the prices of the best bound; and why the run ended
mutable struct Trace
    start::Float64
    time::Vector{Float64}
    value::Vector{Float64}
    bound::Vector{Float64}
    upper::Vector{Float64}
    U::Float64
    best_π::Vector{Float64}
    status::String
end

Trace(π0) = Trace(
    time(),
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Inf,
    copy(π0),
    "running",
)

# Evaluates L at π, with the MIP blocks solved up to the relative gap `gap`,
# and records it in the trace. With `timed = false`, the evaluation is not
# counted in the run time (for methods that need L only to be reported).
# Returns the value, the bound, a supergradient, the per-block results of
# UC.solve_block! with the block cost of the solution, and whether the method
# must stop (then trace.status says why: "target", "time_limit" or
# "unbounded").
function evaluate!(
    trace::Trace,
    oracle,
    π,
    stop::Stop;
    gap = 1e-9,
    timed = true,
)
    start = time()
    blocks = Vector{Any}(undef, length(oracle.blocks))
    dp = get(DP_BLOCKS, oracle, nothing)
    # the blocks of the dynamic program, on all the threads; the others, and
    # the fallbacks, by MIP
    mip = trues(length(oracle.blocks))
    if dp !== nothing
        Threads.@threads :greedy for b in eachindex(oracle.blocks)
            dp[b] === nothing && continue
            Threads.atomic_add!(DP_CALLS, 1)
            r = solve_dp(dp[b], π)
            r === nothing && (Threads.atomic_add!(DP_FALLBACKS, 1); continue)
            blocks[b], mip[b] = r, false
        end
    end
    function solve(b)
        model = oracle.blocks[b].model
        set_attribute(model, MOI.RelativeGapTolerance(), gap)
        r = UC.solve_block!(oracle, b, π)
        cost = isfinite(r.value) ? value(oracle.blocks[b].cost) : NaN
        blocks[b] = (; r..., cost)
        return
    end
    K = length(GUROBI_ENVS)
    @sync for k in 1:K
        Threads.@spawn foreach(solve, filter(b -> mip[b] && env_slot(b, K) == k, eachindex(mip)))
    end
    timed || (trace.start += time() - start)
    L = sum(r.value for r in blocks)
    bound = sum(r.bound for r in blocks)
    grad = -Vector(sum(r.a for r in blocks))
    if bound > maximum(trace.bound; init = -Inf)
        trace.best_π = copy(π)
    end
    push!(trace.time, time() - trace.start)
    push!(trace.value, L)
    push!(trace.bound, bound)
    push!(trace.upper, trace.U)
    trace.status =
        !isfinite(L) ? "unbounded" :
        bound >= stop.target ? "target" :
        trace.time[end] >= stop.time_limit ? "time_limit" : "running"
    return (; value = L, bound, grad, blocks, stop = trace.status != "running")
end

# Price box of the bundle and smoothing methods: π0 ± radius·‖π0‖∞, within
# π ≥ lower. With radius = 1 it contains the optimal prices found by dw.jl on
# all benchmark instances.
function price_box(oracle; radius = 1.0)
    R = radius * maximum(abs, oracle.π0)
    R == 0 && (R = radius)
    return max.(oracle.lower, oracle.π0 .- R), oracle.π0 .+ R
end

# Optimizers of the oracle blocks (MIPs, solved up to the relative gap 1e-9 by
# default; with Gurobi, without presolve, which takes most of the time on
# these small MIPs: the oracle is 1.4 to 4.8 times faster on or-lib, ca and
# ferc, with the same values), of the column generation masters (LPs, with
# duals) and of the bundle and smoothing models (QPs).
mip_optimizer(env = 1) = optimizer_with_attributes(
    () -> Gurobi.Optimizer(GUROBI_ENVS[env]),
    "OutputFlag" => 0,
    "MIPGap" => 1e-9,
    "Presolve" => 0,
    "Threads" => 1,
)
lp_optimizer(env = 1) =
    optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENVS[env]), "OutputFlag" => 0, "Threads" => 1)
qp_optimizer(env = 1) = lp_optimizer(env)

# Moves block b of the oracle (built with mip_optimizer()) to its Gurobi
# environment
function use_gurobi_envs!(oracle)
    K = length(GUROBI_ENVS)
    for (b, blk) in enumerate(oracle.blocks)
        set_optimizer(blk.model, mip_optimizer(env_slot(b, K)))
        set_silent(blk.model)
    end
    return oracle
end

# Projection of the subgradient methods onto π ≥ lower
project(oracle, π) = max.(π, oracle.lower)

include("dp.jl")
