# Dynamic programming oracle for the thermal unit blocks, with the MIP as
# fallback (environment variable ORACLE: "dp", the default, or "mip").
#
# At fixed prices, the block of a unit is a single-unit commitment problem.
# Without the ramping constraints (MorLatRam2013.Ramping), the periods are
# only linked by the status of the unit: on or off, and for how long. The
# dynamic program runs on the states (on, k) and (off, k), where k is the
# number of periods in the current status, capped at max(UT, 2) when on and at
# max(DT, longest startup delay) when off. It enforces the minimum up and down
# times, the initial conditions, must-run and fixed commitments, the startup
# category of each startup (the cheapest allowed by the downtime) and the
# shutdown costs. A fixed commitment is a forced schedule, whose status
# variables are constants of the block (column 0 below). Given the status at
# t and whether t is a startup or the last period before a shutdown, the production and reserves of t solve a small LP:
# the startup and shutdown limits (MorLatRam2013.StartupShutdownLimits, and
# the segment limits of KnuOstWat2018.PwlCosts) cap the total and each segment,
# and the segments and the best spinning reserve fill the cap greedily, by
# decreasing profit. This is exact on the block without ramping: its value is
# a lower bound on the block value.
#
# The solution is then checked against every constraint of the block model
# (compiled once as sparse matrices). If it is feasible, it is optimal, with
# value = bound. Otherwise (a ramp is violated), the block is solved by the
# dynamic program with ramping of dp_ramp.jl, checked in the same way, and
# by the MIP if that fails too.
#
# With several scenarios, the commitment (and its costs) is common and each
# scenario has its own production and reserves, given the commitment: the
# cost of a period, or of an on-interval with ramping, is the sum over the
# scenarios (whose probabilities are in the costs). With an investment
# variable, the block is solved with the investment (the DP above) and
# without it (the unit off), and the best one is kept.
#
# Units the dynamic program does not cover (other reserve types, startup
# costs decreasing with the delay, variables it does not know) always use the
# MIP.

const ORACLE = get(ENV, "ORACLE", "dp")
const DP_BLOCKS = IdDict{Any,Vector{Any}}()
const DP_CALLS = Threads.Atomic{Int}(0)
const DP_RAMP = Threads.Atomic{Int}(0)
const DP_FALLBACKS = Threads.Atomic{Int}(0)

# Production of the unit in one scenario: the unit data of the scenario and
# its columns
struct ScenarioDP
    unit::UC.ThermalUnit
    segprod::Matrix{Int}
    prod_above::Vector{Int}
    spin::Vector{Vector{Int}}
    nonspin::Vector{Vector{Int}}
    q::Vector{Int}
    # flexiramp (WanHob2016): empty without it
    mfg::Vector{Int}                   # max feasible generation, t = 1..T
    up::Vector{Vector{Int}}            # up-flexiramp of each reserve, t < T
    dw::Vector{Vector{Int}}            # down-flexiramp
end

struct UnitDP
    unit::UC.ThermalUnit               # commitment data (first scenario)
    Kon::Int
    Koff::Int
    A::SparseMatrixCSC{Float64,Int}
    lo::Vector{Float64}
    hi::Vector{Float64}
    xlo::Vector{Float64}
    xhi::Vector{Float64}
    c::Vector{Float64}
    offset::Float64
    G::SparseMatrixCSC{Float64,Int}
    g0::Vector{Float64}
    is_on::Vector{Int}
    switch_on::Vector{Int}
    switch_off::Vector{Int}
    startup::Matrix{Int}
    invest::Int                        # investment column (0: none)
    scenarios::Vector{ScenarioDP}
    flex::Bool                         # flexiramp unit
    mlr::Bool                          # MorLatRam2013.Ramping constraints
end

# Dynamic program of block b, for the unit units[i] = (scenario name, unit
# in that scenario); nothing if it is not covered
function UnitDP(oracle, b, units, ramping = UC.MorLatRam2013.Ramping())
    blk = oracle.blocks[b]
    m = blk.model
    g = units[1][2]
    T = length(g.must_run)
    S = length(g.startup_categories)
    all(all(r -> r.type in (:spinning, :non_spinning), u.reserves) for (_, u) in units) || return nothing
    issorted([c.cost for c in g.startup_categories]) || return nothing
    Set(k[1] for k in keys(m[:prod_above])) == Set(first.(units)) || return nothing
    @objective(m, Min, blk.cost)
    d = lp_matrix_data(m)
    col = Dict(v => j for (j, v) in enumerate(d.variables))
    n = length(d.variables)
    covered = falses(n)
    function column(v)
        v isa VariableRef || return 0
        covered[col[v]] = true
        return col[v]
    end
    is_on = [column(m[:is_on][g.name, t]) for t in 1:T]
    switch_on = [column(m[:switch_on][g.name, t]) for t in 1:T]
    switch_off = [column(m[:switch_off][g.name, t]) for t in 1:T]
    startup = [column(m[:startup][g.name, t, s]) for t in 1:T, s in 1:S]
    invest = g.invest > 0 ? column(m[:invest][g.name]) : 0
    g.invest > 0 && invest == 0 && return nothing
    scenarios = ScenarioDP[]
    flex = haskey(object_dictionary(m), :mfg) && !isempty(m[:mfg])
    if flex
        # the closed forms assume min power ≤ SU ≤ min power + RU + max
        # power (same for SD, RD: no lower bound on the production at a
        # startup or before a shutdown), and NoRamping or MorLatRam2013
        ramping isa Union{UC.NoRamping,UC.MorLatRam2013.Ramping} || return nothing
        for (_, u) in units
            lo, hi = u.min_power[1], minimum(u.max_power)
            lo <= u.startup_limit <= lo + u.ramp_up_limit + hi || return nothing
            lo <= u.shutdown_limit <= lo + u.ramp_down_limit + hi || return nothing
        end
    end
    mlr = !flex || ramping isa UC.MorLatRam2013.Ramping
    for (sc, u) in units
        K = length(u.cost_segments)
        segprod = [column(m[:segprod][sc, g.name, t, k]) for t in 1:T, k in 1:K]
        prod_above = [column(m[:prod_above][sc, g.name, t]) for t in 1:T]
        spin = [[column(m[:reserve][sc, r.name, g.name, t]) for r in u.reserves if r.type == :spinning] for t in 1:T]
        nonspin = [[column(m[:reserve][sc, r.name, g.name, t]) for r in u.reserves if r.type == :non_spinning] for t in 1:T]
        q = [column(m[:qg_thermal][sc, g.name, t]) for t in 1:T]
        any(==(0), [vec(segprod); prod_above; q]) && return nothing
        mfg, up, dw = Int[], Vector{Int}[], Vector{Int}[]
        if flex
            mfg = [column(m[:mfg][sc, g.name, t]) for t in 1:T]
            up = [[column(v) for (k, v) in m[:upflexiramp] if k[1] == sc && k[4] == t] for t in 1:T-1]
            dw = [[column(v) for (k, v) in m[:dwflexiramp] if k[1] == sc && k[4] == t] for t in 1:T-1]
        end
        push!(scenarios, ScenarioDP(u, segprod, prod_above, spin, nonspin, q, mfg, up, dw))
    end
    fixed = UC._is_commitment_fixed(g)
    fixed == all(==(0), [is_on; switch_on; switch_off; vec(startup)]) || return nothing
    xlo, xhi = copy(d.x_lower), copy(d.x_upper)
    xlo[d.binaries] .= max.(xlo[d.binaries], 0.0)
    xhi[d.binaries] .= min.(xhi[d.binaries], 1.0)
    # every other variable must be fixed
    all(j -> covered[j] || xlo[j] == xhi[j], 1:n) || return nothing
    I, J, V = Int[], Int[], Float64[]
    g0 = zeros(length(oracle.keys))
    for (i, gi) in zip(blk.rows, blk.g)
        g0[i] += gi.constant
        for (v, a) in gi.terms
            push!(I, i)
            push!(J, col[v])
            push!(V, a)
        end
    end
    G = sparse(I, J, V, length(oracle.keys), n)
    # the reactive power must be free of cost and price
    all(j -> d.c[j] == 0 && isempty(nzrange(G, j)), Iterators.flatten(S.q for S in scenarios)) ||
        return nothing
    Kon = max(g.min_uptime, 2)
    Koff = max(g.min_downtime, S > 0 ? g.startup_categories[end].delay : 0, 2)
    return UnitDP(g, Kon, Koff, d.A, d.b_lower, d.b_upper, xlo, xhi, d.c, d.c_offset,
        G, g0, is_on, switch_on, switch_off, startup, invest, scenarios, flex, mlr)
end

# Builds the dynamic programs of the thermal unit blocks of the oracle, used by
# evaluate! from then on
function enable_dp!(oracle, instance)
    units = Dict(g.name => g for g in instance.scenarios[1][:thermal])
    by_scenario = [Dict(g.name => g for g in sc[:thermal]) for sc in instance.scenarios]
    ramping = first(e for e in instance.extensions if e isa UC.ThermalExt).ramping
    DP_BLOCKS[oracle] = Any[
        haskey(units, blk.name) ?
        UnitDP(oracle, b, [(sc.name, u[blk.name]) for (sc, u) in zip(instance.scenarios, by_scenario)], ramping) :
        nothing for (b, blk) in enumerate(oracle.blocks)
    ]
    return count(!isnothing, DP_BLOCKS[oracle])
end

# Cost of the production and reserves of period t (on), under the reduced
# costs d, if t is a startup (su) and/or the last period before a shutdown
# (sd), summed over the scenarios; Inf if infeasible. With x, writes the
# solution. buf holds three vectors of length K + 1 (profits, bounds and
# order of the items), for the largest number K of segments.
function production(D::UnitDP, d, t, su::Bool, sd::Bool, buf, x = nothing)
    total = 0.0
    for S in D.scenarios
        total += production(S, d, t, su, sd, buf, x)
    end
    return total
end

function production(D::ScenarioDP, d, t, su::Bool, sd::Bool, buf, x = nothing)
    K = size(D.segprod, 2)
    profit, bound, order = view(buf[1], 1:K+1), view(buf[2], 1:K+1), view(buf[3], 1:K+1)
    cap = limits!(bound, D, t, su, sd)
    cap < 0 && return Inf
    dpa = d[D.prod_above[t]]
    # items: segments k (profit, bound), then the best spinning reserve
    for k in 1:K
        profit[k] = -(d[D.segprod[t, k]] + dpa)
    end
    spin = D.spin[t]
    r = 0
    for j in spin
        (r == 0 || d[j] < d[r]) && (r = j)
    end
    profit[K+1] = r == 0 ? -Inf : -d[r]
    bound[K+1] = Inf
    total, left = 0.0, cap
    for i in sortperm!(order, profit; rev = true)
        (profit[i] > 0 && left > 0) || break
        y = min(bound[i], left)
        total -= profit[i] * y
        left -= y
        if x !== nothing
            if i <= K
                x[D.segprod[t, i]] = y
                x[D.prod_above[t]] += y
            else
                x[r] = y
            end
        end
    end
    return total
end

# Cap on prod_above + spinning reserve at period t (on), if t is a startup
# (su) and/or the last period before a shutdown (sd), by the startup and
# shutdown limits; the bounds of the segments in u[1:K]. -1 if infeasible.
function limits!(u, D::ScenarioDP, t, su::Bool, sd::Bool)
    g = D.unit
    Pmax, Pmin = g.max_power[t], g.min_power[t]
    SU, SD = g.startup_limit, g.shutdown_limit
    base = Pmax - Pmin
    a, b = (Pmax - SU) * su, (Pmax - SD) * sd
    cap = g.min_uptime > 1 ? min(base, base - a - b) : min(base, base - a, base - b)
    cap < -1e-9 && return -1.0
    P0 = Pmin
    for k in 1:size(D.segprod, 2)
        mw = g.cost_segments[k].mw[t]
        P1 = P0 + mw
        Cv = P1 <= SU ? 0.0 : P0 < SU ? P1 - SU : mw
        Cw = P1 <= SD ? 0.0 : P0 < SD ? P1 - SD : mw
        v = g.min_uptime > 1 ? mw - Cv * su - Cw * sd : min(mw - Cv * su, mw - Cw * sd)
        v < -1e-9 && return -1.0
        u[k] = max(min(v, mw), 0.0)
        P0 = P1
    end
    return max(cap, 0.0)
end

segment_buffers(D::UnitDP) =
    (K = maximum(size(S.segprod, 2) for S in D.scenarios); (zeros(K + 1), zeros(K + 1), zeros(Int, K + 1)))

# The startup and shutdown limits forbid a shutdown at t = 1 if the initial
# power is above the shutdown limit (in some scenario)
no_shutdown_at_1(D::UnitDP) =
    any(something(S.unit.initial_power, 0.0) > S.unit.shutdown_limit for S in D.scenarios)

# Profit of the non-spinning reserves of period t (off)
off_cost(D::UnitDP, d, t) = sum(
    min(0.0, d[j]) * S.unit.non_spinning_capacity for S in D.scenarios for j in S.nonspin[t];
    init = 0.0,
)

# Reduced cost of column j, and its value in x, where column 0 is a constant
cost(d, j) = j == 0 ? 0.0 : d[j]
set!(x, j, v) = j == 0 || (x[j] = v)

# Startup category at t after k periods off: the cheapest allowed
function category(D::UnitDP, d, t, k)
    cats = D.unit.startup_categories
    S = length(cats)
    best = S
    for s in 1:(S-1)
        cats[s].delay <= k <= cats[s+1].delay - 1 || continue
        cost(d, D.startup[t, s]) < cost(d, D.startup[t, best]) && (best = s)
    end
    return best
end

can_be_on(g, t) = g.commitment_status[t] !== false
can_be_off(g, t) = !g.must_run[t] && g.commitment_status[t] !== true

# Solves the block at π by the DP without ramping, then, if its solution
# violates a block constraint, by the DP with ramping (dp_ramp.jl); nothing if
# that one fails too (then the MIP is used)
function solve_dp(D::UnitDP, π)
    d = D.c .- D.G' * π
    constant = D.offset - dot(π, D.g0)
    # a flexiramp unit always has ramping constraints (WanHob2016)
    x = D.flex ? nothing : noramp_x(D, d)
    if x === nothing || !feasible(D, x)
        Threads.atomic_add!(DP_RAMP, 1)
        x = ramp_x(D, d)
        x === nothing || feasible(D, x) || (x = nothing)
    end
    if D.invest != 0
        # without the investment: the unit is off
        off = status_x(D, d, falses(length(D.is_on)))
        off[D.invest] = 0.0
        feasible(D, off) && (x === nothing || dot(d, off) < dot(d, x)) && (x = off)
    end
    x === nothing && return nothing
    value = dot(d, x) + constant
    a = sparse(D.G * x .+ D.g0)
    # x (in the column order of lp_matrix_data of the block model) is read by CRG
    return (; status = MOI.OPTIMAL, value, bound = value, a, cost = dot(D.c, x) + D.offset, x)
end

function feasible(D::UnitDP, x)
    Ax = D.A * x
    tol(v) = 1e-7 * (1 + abs(v))
    all(i -> D.lo[i] - tol(D.lo[i]) <= Ax[i] <= D.hi[i] + tol(D.hi[i]), eachindex(Ax)) || return false
    return all(j -> D.xlo[j] - tol(D.xlo[j]) <= x[j] <= D.xhi[j] + tol(D.xhi[j]), eachindex(x))
end

# Solution with the status on[t] (for the status variables, the startup
# categories, the reactive power and the non-spinning reserves; production and
# spinning reserves at 0)
function status_x(D::UnitDP, d, on)
    g = D.unit
    T = length(D.is_on)
    x = copy(D.xlo)
    for j in [D.is_on; D.switch_on; D.switch_off; vec(D.startup)]
        set!(x, j, 0.0)
    end
    D.invest != 0 && (x[D.invest] = 1.0)
    for S in D.scenarios
        foreach(j -> x[j] = 0.0, [vec(S.segprod); S.prod_above; S.q])
        foreach(v -> x[v] = 0.0, Iterators.flatten(S.spin))
        foreach(v -> x[v] = 0.0, Iterators.flatten(S.nonspin))
        foreach(j -> x[j] = 0.0, S.mfg)
        foreach(v -> x[v] = 0.0, Iterators.flatten(S.up))
        foreach(v -> x[v] = 0.0, Iterators.flatten(S.dw))
    end
    prev_on = g.initial_status > 0
    downtime = prev_on ? 0 : min(-g.initial_status, D.Koff)
    for t in 1:T
        if on[t]
            set!(x, D.is_on[t], 1.0)
            if !prev_on
                set!(x, D.switch_on[t], 1.0)
                set!(x, D.startup[t, category(D, d, t, downtime)], 1.0)
            end
            for S in D.scenarios
                x[S.q[t]] = clamp(0.0, S.unit.qmin, S.unit.qmax)
            end
            downtime = 0
        else
            prev_on && set!(x, D.switch_off[t], 1.0)
            for S in D.scenarios, j in S.nonspin[t]
                d[j] < 0 && (x[j] = S.unit.non_spinning_capacity)
            end
            downtime = min(downtime + 1, D.Koff)
        end
        prev_on = on[t]
    end
    return x
end

# Solution of the DP without ramping, under the reduced costs d; nothing if
# the block is infeasible without ramping
function noramp_x(D::UnitDP, d)
    g = D.unit
    T = length(D.is_on)
    Kon, Koff = D.Kon, D.Koff
    nS = Kon + Koff                    # states: on k = 1..Kon, off k = Kon+1..
    buf = segment_buffers(D)
    P = [production(D, d, t, su, sd, buf) for t in 1:T, su in (false, true), sd in (false, true)]
    off = [off_cost(D, d, t) for t in 1:T]
    init_on = g.initial_status > 0
    s0 = init_on ? min(g.initial_status, Kon) : Kon + min(-g.initial_status, Koff)
    no_off_at_1 = no_shutdown_at_1(D)
    # cur[s]: best cost of periods 1..t with state s at t, without the
    # production of t (which depends on a shutdown at t + 1). On k ≥ 2 and
    # off k ≥ 2 come from k - 1, except the capped states Kon and Kon + Koff,
    # which may also come from themselves; on 1 (startup) and off 1
    # (shutdown) come from the state recorded in from_off and from_on.
    cur, nxt = fill(Inf, nS), fill(Inf, nS)
    cur[s0] = 0.0
    from_off, from_on = zeros(Int, T), zeros(Int, T)
    on_capped, off_capped = falses(T), falses(T)
    for t in 1:T
        fill!(nxt, Inf)
        on_ok, off_ok = can_be_on(g, t), can_be_off(g, t)
        # production of t - 1 of on state k, if t is on (sd = 1) or off (sd = 2)
        pend(k, sd) = t == 1 ? 0.0 : P[t-1, 1+(k == 1), sd]
        if on_ok
            c_on = cost(d, D.is_on[t])
            for k in 2:Kon
                v = cur[k-1] + pend(k - 1, 1)
                if k == Kon
                    w = cur[Kon] + pend(Kon, 1)
                    w < v && ((v, on_capped[t]) = (w, true))
                end
                nxt[k] = v + c_on
            end
            c_start = c_on + cost(d, D.switch_on[t])
            for k in g.min_downtime:Koff
                k >= 1 || continue
                v = cur[Kon+k]
                isfinite(v) || continue
                v += c_start + cost(d, D.startup[t, category(D, d, t, k)])
                v < nxt[1] && ((nxt[1], from_off[t]) = (v, Kon + k))
            end
        end
        if off_ok
            for k in 2:Koff
                v = cur[Kon+k-1]
                if k == Koff
                    w = cur[Kon+Koff]
                    w < v && ((v, off_capped[t]) = (w, true))
                end
                nxt[Kon+k] = v + off[t]
            end
            if !(t == 1 && no_off_at_1)
                c_off = cost(d, D.switch_off[t]) + off[t]
                for k in max(g.min_uptime, 1):Kon
                    v = cur[k] + pend(k, 2) + c_off
                    v < nxt[Kon+1] && ((nxt[Kon+1], from_on[t]) = (v, k))
                end
            end
        end
        cur, nxt = nxt, cur
    end
    s, best = 0, Inf
    for i in 1:nS
        v = i <= Kon ? cur[i] + P[T, 1+(i == 1), 1] : cur[i]
        v < best && ((s, best) = (i, v))
    end
    isfinite(best) || return nothing
    # backtrack: the state of each period, then the solution
    states = zeros(Int, T)
    for t in T:-1:1
        states[t] = s
        s = s == 1 ? from_off[t] : s == Kon + 1 ? from_on[t] :
            s == Kon ? (on_capped[t] ? Kon : Kon - 1) :
            s == Kon + Koff ? (off_capped[t] ? s : s - 1) : s - 1
    end
    on = BitVector([states[t] <= Kon for t in 1:T])
    x = status_x(D, d, on)
    for t in 1:T
        on[t] || continue
        su = t == 1 ? !init_on : !on[t-1]
        production(D, d, t, su, t < T && !on[t+1], buf, x)
    end
    return x
end

include("dp_ramp.jl")
