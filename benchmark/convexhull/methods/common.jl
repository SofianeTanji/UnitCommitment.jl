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

using UnitCommitment, Clarabel, HiGHS, JuMP, LinearAlgebra, SparseArrays
const UC = UnitCommitment

# Solver of the oracle MIPs and of the models of the methods, from the
# environment variable SOLVER: "highs" (the default; Clarabel for the QPs) or
# "gurobi"
const SOLVER = get(ENV, "SOLVER", "highs")
SOLVER in ("highs", "gurobi") || error("SOLVER=$SOLVER: not highs or gurobi")
SOLVER == "gurobi" && @eval using Gurobi

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
    Threads.@threads :greedy for b in eachindex(oracle.blocks)
        model = oracle.blocks[b].model
        set_attribute(model, MOI.RelativeGapTolerance(), gap)
        r = UC.solve_block!(oracle, b, π)
        cost = isfinite(r.value) ? value(oracle.blocks[b].cost) : NaN
        blocks[b] = (; r..., cost)
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
# default), of the column generation masters (LPs, with duals) and of the
# bundle and smoothing models (QPs). With HiGHS, the QPs are solved by
# Clarabel, as the HiGHS QP solver can stall on them.
mip_optimizer() =
    SOLVER == "gurobi" ?
    optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "MIPGap" => 1e-9, "Threads" => 1) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "mip_rel_gap" => 1e-9,
        "threads" => 1,
    )
lp_optimizer() =
    SOLVER == "gurobi" ?
    optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => 1) :
    optimizer_with_attributes(
        HiGHS.Optimizer,
        "output_flag" => false,
        "threads" => 1,
    )
qp_optimizer() =
    SOLVER == "gurobi" ?
    optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "Threads" => 1) :
    optimizer_with_attributes(Clarabel.Optimizer, "verbose" => false)

# Projection of the subgradient methods onto π ≥ lower
project(oracle, π) = max.(π, oracle.lower)
