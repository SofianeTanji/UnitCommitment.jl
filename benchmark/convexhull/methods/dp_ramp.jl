# Dynamic program with ramping for a thermal unit block (included by dp.jl):
# the algorithm of Frangioni and Gentile (2006), in the forward form with
# dominance pruning of Wuijts, van den Akker and van den Broek (2020, RRF+),
# with the spinning reserve eliminated in closed form.
#
# An on-interval starting at h is followed by its value function V_t(p), the
# best cost of the periods h..t with prod_above[t] = p: convex piecewise
# linear. With the reserve price μ_t ≥ 0 of the best spinning reserve, the
# optimal reserve is r_t = min(C_t - p_t, RU + p_{t-1} - p_t), where C_t caps
# prod_above + reserve, so that -μ_t r_t = μ_t (p_t - C_t) + μ_t (C_t - RU -
# p_{t-1})⁺. The update V_{t-1} → V_t is then:
#
#   W(q) = V_{t-1}(q) + μ_t (C_t - RU - q)⁺
#   G(p) = min {W(q) : p - RU ≤ q ≤ p + RD}   (split W at its minimizer q*,
#          shift the left part by -RD and the right part by +RU)
#   V_t(p) = G(p) + f_t(p) + μ_t (p - C_t) on [0, ub_t]
#
# where f_t is the cost of the segments (filled by increasing reduced cost)
# and ub_t = min(C_t, Σ_k u_k), and RD before a shutdown. The interval starts
# from p = 0 (startup, with its ramp), from the initial power (on at t = 1),
# or free (startup at t = 1: no ramping constraint at t = 1). The startup
# and shutdown limits change C_t and u at the first and last period. Each
# interval also gives Z_t, its best cost if t is the last period before a
# shutdown. The off states are those of the DP without ramping. Among the
# intervals older than UT, those whose V_t is above another one's (on a
# smaller domain) are dropped.
#
# The solution is rebuilt by replaying the chosen intervals: p_k minimizes
# V_k, and p_{t-1} is the projection of q* onto the window of p_t.

struct PWL
    x::Vector{Float64}
    y::Vector{Float64}
end

const NOPWL = PWL(Float64[], Float64[])
const PWL_TOL = 1e-9

# Value inside the domain (clamped at its ends)
function evalpwl(x, y, p)
    p <= x[1] && return y[1]
    p >= x[end] && return y[end]
    i = searchsortedlast(x, p)
    return y[i] + (y[i+1] - y[i]) * (p - x[i]) / (x[i+1] - x[i])
end

# Data of period t (on) for a startup (su) and/or a shutdown after t (sd)
struct RampPeriod
    C::Float64                         # cap on prod_above + reserve
    ub::Float64                        # bound on prod_above
    sx::Vector{Float64}                # cost of the segments: breakpoints,
    sy::Vector{Float64}                # values
    order::Vector{Int}                 # segments by increasing cost
    u::Vector{Float64}                 # segment bounds
    μ::Float64                         # reserve price (≥ 0)
    r::Int                             # reserve column (0: no reserve)
    μh::Float64                        # μ if the reserve is in the ramp-up
    λu::Float64                        # up-flexiramp price of t - 1 (≥ 0)
    au::Float64                        # its hinge: λu (q - au)⁺
end

# (e: reduced costs of the segments, order: their increasing order)
function insert_break!(x, y, a)
    i = searchsortedlast(x, a)
    (1 <= i < length(x) && x[i] + PWL_TOL < a < x[i+1] - PWL_TOL) || return
    v = y[i] + (y[i+1] - y[i]) * (a - x[i]) / (x[i+1] - x[i])
    insert!(x, i + 1, a)
    insert!(y, i + 1, v)
    return
end

function ramp_period(D::ScenarioDP, d, t, su::Bool, sd::Bool, e, order, mlr::Bool)
    K = size(D.segprod, 2)
    u = zeros(K)
    C = limits!(u, D, t, su, sd)
    C < 0 && return nothing
    sx, sy = sizehint!([0.0], K + 1), sizehint!([0.0], K + 1)
    for k in order
        u[k] > 0 || continue
        push!(sx, sx[end] + u[k])
        push!(sy, sy[end] + e[k] * u[k])
    end
    r = 0
    for j in D.spin[t]
        (r == 0 || d[j] < d[r]) && (r = j)
    end
    μ = r == 0 ? 0.0 : max(0.0, -d[r])
    μ == 0 && (r = 0)
    g = D.unit
    ub = min(C, sx[end], sd && mlr ? g.ramp_down_limit : Inf)
    λu, au = 0.0, Inf
    if !isempty(D.mfg)
        # Flexiramp (WanHob2016), in closed form given the status: the
        # down-flexiramp of t is min(RD, p_t) (g_t before a shutdown), the
        # up-flexiramp of t - 1 is min(RU, maxp_{t-1}, cw_t - p_{t-1}), where
        # cw_t caps prod_above[t]; before a startup they are constants.
        T = length(D.prod_above)
        minp = g.min_power[t]
        cw = min(g.max_power[t], su ? g.startup_limit : Inf, sd ? g.shutdown_limit : Inf) - minp
        cw < -1e-9 && return nothing
        # a startup at t ≥ 2 needs SD ≤ maxp_{t-1}: (24) at t - 1
        su && t > 1 && g.shutdown_limit > g.max_power[t-1] + 1e-9 && return nothing
        ub = min(ub, max(cw, 0.0))
        # Prices of the upper bounds of the up- and down-flexiramp of period
        # s: a variable with a positive cost is at its lower bound, which is
        # minus the upper bound of the other direction.
        pu(s) = sum(max(0.0, -d[j]) for j in D.up[s]; init = 0.0) + sum(max(0.0, d[j]) for j in D.dw[s]; init = 0.0)
        pd(s) = sum(max(0.0, -d[j]) for j in D.dw[s]; init = 0.0) + sum(max(0.0, d[j]) for j in D.up[s]; init = 0.0)
        if t < T
            λd = pd(t)
            if sd
                # up-flexiramp min(0, RU - SU + maxp_t), down-flexiramp g_t
                sy .-= λd .* (sx .+ minp) .+ pu(t) * min(0.0, g.ramp_up_limit - g.startup_limit + g.max_power[t])
            else
                RD = g.ramp_down_limit
                insert_break!(sx, sy, RD)
                sy .-= λd .* min.(sx, RD)
            end
        end
        if t > 1
            λ = pu(t - 1)
            if su
                sy .+= pd(t - 1) * minp -
                       λ * min(g.startup_limit, g.max_power[t-1], g.max_power[t], sd ? g.shutdown_limit : Inf)
            else
                RUm = min(g.ramp_up_limit, g.max_power[t-1])
                λu, au = λ, cw - RUm
                sy .-= λ * RUm
            end
        end
    end
    return RampPeriod(C, ub, sx, sy, order, u, μ, r, mlr ? μ : 0.0, λu, au)
end

# Buffers of ramp_step, reused across its calls
struct RampScratch
    wx::Vector{Float64}
    wy::Vector{Float64}
    gx::Vector{Float64}
    gy::Vector{Float64}
end

RampScratch() = RampScratch(Float64[], Float64[], Float64[], Float64[])

function push_increasing!(x, y, a, v)
    if isempty(x) || a > x[end] + PWL_TOL
        push!(x, a)
        push!(y, v)
    end
    return
end

# V_t from V_{t-1} and the data P of period t, with the constant c; also q*
# and the domain of W, for the backtracking
function ramp_step(V::PWL, P, c, RU, RD, S::RampScratch)
    (P === nothing || isempty(V.x)) && return NOPWL, NaN, NaN, NaN
    # W(q) = V(q) + μ (a - q)⁺ + λ (q - h)⁺, with breakpoints at a and h
    # (no variable captured by hinges may be reassigned: it would be boxed)
    wx, wy = empty!(S.wx), empty!(S.wy)
    a, μ = P.C - RU, P.μh
    h, λ = P.au, P.λu
    hinges(q) = μ * max(0.0, a - q) + λ * max(0.0, q - h)
    c1, c2 = μ > 0 ? a : Inf, λ > 0 ? h : Inf
    c1 > c2 && ((c1, c2) = (c2, c1))
    for i in eachindex(V.x)
        xi = V.x[i]
        if i > 1
            for c in (c1, c2)
                (V.x[i-1] < c < xi && c > wx[end]) || continue
                push!(wx, c)
                push!(wy, V.y[i-1] + (V.y[i] - V.y[i-1]) * (c - V.x[i-1]) / (xi - V.x[i-1]) + hinges(c))
            end
        end
        push!(wx, xi)
        push!(wy, V.y[i] + hinges(xi))
    end
    m = argmin(wy)
    q, wlo, whi = wx[m], wx[1], wx[end]
    # G: the part left of q* shifted by -RD, the part right of it by +RU
    gx, gy = empty!(S.gx), empty!(S.gy)
    for i in 1:m
        push_increasing!(gx, gy, wx[i] - RD, wy[i])
    end
    for i in m:length(wx)
        push_increasing!(gx, gy, wx[i] + RU, wy[i])
    end
    # V_t = G + segments + μ (p - C) + c on [0, ub]
    lo, hi = max(0.0, gx[1]), min(P.ub, gx[end])
    lo > hi + PWL_TOL && return NOPWL, q, wlo, whi
    hi = max(hi, lo)
    x = sizehint!([lo], length(gx) + length(P.sx) + 2)
    i, j = 1, 1
    while i <= length(gx) || j <= length(P.sx)
        b = if j > length(P.sx) || (i <= length(gx) && gx[i] <= P.sx[j])
            (i += 1; gx[i-1])
        else
            (j += 1; P.sx[j-1])
        end
        lo + PWL_TOL < b < hi - PWL_TOL && b > x[end] + PWL_TOL && push!(x, b)
    end
    hi > x[end] + PWL_TOL && push!(x, hi)
    y = [evalpwl(gx, gy, b) + evalpwl(P.sx, P.sy, b) + P.μ * (b - P.C) + c for b in x]
    return PWL(x, y), q, wlo, whi
end

# f(p) + cost of the segments + μ (p - C) + c, on the domain of f
function add_period(f::PWL, P::RampPeriod, c)
    lo, hi = f.x[1], f.x[end]
    x = copy(f.x)
    for a in P.sx
        lo + PWL_TOL < a < hi - PWL_TOL && push!(x, a)
    end
    sort!(x)
    y = [evalpwl(f.x, f.y, a) + evalpwl(P.sx, P.sy, a) + P.μ * (a - P.C) + c for a in x]
    return PWL(x, y)
end

# V_1 of a startup at t = 1 (no ramping constraint at t = 1)
function free_start(P, c)
    (P === nothing || P.ub < -PWL_TOL) && return NOPWL
    f = P.ub > PWL_TOL ? PWL([0.0, P.ub], [0.0, 0.0]) : PWL([0.0], [0.0])
    return add_period(f, P, c)
end

# An on-interval: its first period (0: on initially) and its value function
# in each scenario (the constant costs are in the first one)
mutable struct RampInterval
    h::Int
    V::Vector{PWL}
end

minvalue(f::PWL) = isempty(f.y) ? Inf : minimum(f.y)

# Is f ≥ g on the domain of f, which is inside the domain of g?
function dominated(f::PWL, g::PWL)
    (isempty(f.x) || isempty(g.x)) && return isempty(f.x)
    (f.x[1] >= g.x[1] - PWL_TOL && f.x[end] <= g.x[end] + PWL_TOL) || return false
    tol(v) = 1e-9 * (1 + abs(v))
    for a in f.x
        evalpwl(f.x, f.y, a) >= evalpwl(g.x, g.y, a) - tol(evalpwl(g.x, g.y, a)) || return false
    end
    for a in g.x
        f.x[1] <= a <= f.x[end] || continue
        evalpwl(f.x, f.y, a) >= evalpwl(g.x, g.y, a) - tol(evalpwl(g.x, g.y, a)) || return false
    end
    return true
end

# Ramping data of one scenario: the ramp limits, the initial prod_above, and
# the period data P[t, 1 + su, 1 + sd]
struct RampScenario
    RU::Float64
    RD::Float64
    p0::Float64
    P::Array{Union{Nothing,RampPeriod},3}
    mlr::Bool
end

function RampScenario(S::ScenarioDP, d, T, mlr)
    u = S.unit
    K = size(S.segprod, 2)
    P = Array{Union{Nothing,RampPeriod}}(undef, T, 2, 2)
    for t in 1:T
        e = [d[S.segprod[t, k]] + d[S.prod_above[t]] for k in 1:K]
        order = sortperm(e)
        for su in (false, true), sd in (false, true)
            P[t, 1+su, 1+sd] = ramp_period(S, d, t, su, sd, e, order, mlr)
        end
    end
    p0 = u.initial_status > 0 ? u.initial_power - u.min_power[1] : 0.0
    return RampScenario(u.ramp_up_limit, u.ramp_down_limit, p0, P, mlr)
end

# One step of every scenario at period t (su: startup at t, sd: shutdown
# after t), with the constant c in the first one; nothing if one is empty
function ramp_steps(V::Vector{PWL}, R, t, su, sd, c, buf)
    W = Vector{PWL}(undef, length(R))
    for (s, Rs) in enumerate(R)
        W[s] = first(ramp_step(V[s], Rs.P[t, 1+su, 1+sd], s == 1 ? c : 0.0, Rs.RU, Rs.RD, buf))
        isempty(W[s].x) && return nothing
    end
    return W
end

# First step of an interval starting at t: from p = 0 (startup at t ≥ 2) or
# free (startup at t = 1)
function ramp_starts(R, t, sd, c, buf)
    W = Vector{PWL}(undef, length(R))
    for (s, Rs) in enumerate(R)
        Pt, cs = Rs.P[t, 2, 1+sd], s == 1 ? c : 0.0
        W[s] = t == 1 || !Rs.mlr ? free_start(Pt, cs) : first(ramp_step(PWL([0.0], [0.0]), Pt, cs, Rs.RU, Rs.RD, buf))
        isempty(W[s].x) && return nothing
    end
    return W
end

total_min(V) = V === nothing ? Inf : sum(minvalue, V)

# Solution of the DP with ramping, under the reduced costs d; nothing if the
# block is infeasible
function ramp_x(D::UnitDP, d)
    g = D.unit
    T = length(D.is_on)
    Koff = D.Koff
    UT, DT = g.min_uptime, g.min_downtime
    for S in D.scenarios
        # the max feasible generation must be free of cost
        all(j -> d[j] == 0, S.mfg) || return nothing
    end
    R = [RampScenario(S, d, T, D.mlr) for S in D.scenarios]
    off = [off_cost(D, d, t) for t in 1:T]
    buf = RampScratch()
    init_on = g.initial_status > 0
    age(h, t) = h == 0 ? t + g.initial_status : t - h + 1    # periods on at t
    cur = fill(Inf, Koff)
    init_on || (cur[min(-g.initial_status, Koff)] = 0.0)
    live = init_on ? [RampInterval(0, [PWL([Rs.p0], [0.0]) for Rs in R])] : RampInterval[]
    # best shutdown after t: its cost and interval
    close_value, close_from = Inf, -1
    if init_on && g.initial_status >= UT && !no_shutdown_at_1(D) &&
       (!D.mlr || all(Rs.p0 <= Rs.RD + PWL_TOL for Rs in R))
        close_value, close_from = 0.0, 0                     # off at t = 1
    end
    start_from = zeros(Int, T)          # off state before the startup at t
    from_on = fill(-1, T)               # interval ended by the shutdown at t
    off_capped = falses(T)
    for t in 1:T
        on_ok, off_ok = can_be_on(g, t), can_be_off(g, t)
        nxt = fill(Inf, Koff)
        if off_ok
            if isfinite(close_value)
                nxt[1] = close_value + cost(d, D.switch_off[t]) + off[t]
                from_on[t] = close_from
            end
            for k in 2:Koff
                v = cur[k-1]
                if k == Koff && cur[Koff] < v
                    v, off_capped[t] = cur[Koff], true
                end
                nxt[k] = v + off[t]
            end
        end
        close_value, close_from = Inf, -1
        next = RampInterval[]
        if on_ok
            c_on = cost(d, D.is_on[t])
            can_close = t < T && can_be_off(g, t + 1)
            for I in live
                if can_close && age(I.h, t) >= UT
                    Z = total_min(ramp_steps(I.V, R, t, false, true, c_on, buf))
                    Z < close_value && ((close_value, close_from) = (Z, I.h))
                end
                V = ramp_steps(I.V, R, t, false, false, c_on, buf)
                V === nothing || push!(next, RampInterval(I.h, V))
            end
            # startup at t
            best, from = Inf, 0
            if t == 1
                k0 = min(-g.initial_status, Koff)
                if !init_on && -g.initial_status >= DT
                    best, from = cost(d, D.startup[1, category(D, d, 1, k0)]), k0
                end
            else
                for k in max(DT, 1):Koff
                    isfinite(cur[k]) || continue
                    v = cur[k] + cost(d, D.startup[t, category(D, d, t, k)])
                    v < best && ((best, from) = (v, k))
                end
            end
            if isfinite(best)
                start_from[t] = from
                c = best + c_on + cost(d, D.switch_on[t])
                if can_close && age(t, t) >= UT
                    Z = total_min(ramp_starts(R, t, true, c, buf))
                    Z < close_value && ((close_value, close_from) = (Z, t))
                end
                V = ramp_starts(R, t, false, c, buf)
                V === nothing || push!(next, RampInterval(t, V))
            end
        end
        # dominance among the intervals older than UT, in every scenario
        old = [i for i in eachindex(next) if age(next[i].h, t) >= UT]
        drop = falses(length(next))
        for i in old, j in old
            i != j && !drop[j] && !drop[i] &&
                all(dominated(next[i].V[s], next[j].V[s]) for s in eachindex(R)) && (drop[i] = true)
        end
        live = next[.!drop]
        cur = nxt
    end
    # best final state
    best, s = findmin(cur)
    h_end = -1
    for I in live
        v = total_min(I.V)
        v < best && ((best, h_end) = (v, I.h))
    end
    isfinite(best) || return nothing
    # backtrack: the on-intervals (h, k, shutdown after k)
    on = falses(T)
    intervals = Tuple{Int,Int,Bool}[]
    t, k_off = T, s
    if h_end >= 0
        push!(intervals, (h_end, T, false))
        t = h_end - 1
        k_off = h_end <= 1 ? 0 : start_from[h_end]
    end
    while t >= 1 && k_off > 0
        if k_off == 1
            h = from_on[t]
            push!(intervals, (h, t - 1, true))
            k_off = h <= 1 ? 0 : start_from[h]
            t = h - 1
        else
            k_off = k_off == Koff && off_capped[t] ? Koff : k_off - 1
            t -= 1
        end
    end
    for (h, k, _) in intervals
        on[max(h, 1):k] .= true
    end
    x = status_x(D, d, on)
    for (h, k, closed) in intervals, (S, Rs) in zip(D.scenarios, R)
        k >= max(h, 1) || continue
        replay!(x, S, Rs, h, k, closed, buf) || return nothing
    end
    for S in D.scenarios
        flex_fill!(x, S, on, d)
    end
    return x
end

# Flexiramp variables of scenario S, given the status and the production in x
function flex_fill!(x, S::ScenarioDP, on, d)
    isempty(S.mfg) && return
    g = S.unit
    T = length(on)
    minp = g.min_power[1]
    SU, SD, RU, RD = g.startup_limit, g.shutdown_limit, g.ramp_up_limit, g.ramp_down_limit
    ison(t) = t == 0 ? g.initial_status > 0 : on[t]
    gen(t) = t == 0 ? g.initial_power : on[t] ? x[S.prod_above[t]] + minp : 0.0
    closing(t) = t < T && on[t] && !on[t+1]
    gbar(t) = !on[t] ? 0.0 : min(g.max_power[t], ison(t - 1) ? gen(t - 1) + RU : SU, closing(t) ? SD : Inf)
    for t in 1:T
        x[S.mfg[t]] = gbar(t)
    end
    for t in 1:T-1
        ur, dr = if on[t] && on[t+1]
            min(RU, g.max_power[t], gbar(t + 1) - gen(t)), min(RD, gen(t) - minp)
        elseif on[t]
            min(0.0, RU - SU + g.max_power[t]), gen(t)
        elseif on[t+1]
            min(SU, g.max_power[t], gbar(t + 1)), -minp
        else
            0.0, 0.0
        end
        foreach(j -> x[j] = d[j] <= 0 ? ur : -dr, S.up[t])
        foreach(j -> x[j] = d[j] <= 0 ? dr : -ur, S.dw[t])
    end
    return
end

# Production and reserves of scenario S over the on-interval h..k (shutdown
# after k if closed), written to x
function replay!(x, S::ScenarioDP, Rs::RampScenario, h, k, closed, buf)
    RU, RD = Rs.RU, Rs.RD
    first_t = max(h, 1)
    n = k - first_t + 1
    q, lo, hi = fill(NaN, n), fill(NaN, n), fill(NaN, n)
    data = [Rs.P[t, 1+(t == h), 1+(closed && t == k)] for t in first_t:k]
    any(isnothing, data) && return false
    free = h >= 1 && (!Rs.mlr || (h == 1 && S.unit.initial_status <= 0))
    q_before = h == 0 ? Rs.p0 : 0.0
    V = free ? free_start(data[1], 0.0) : first(ramp_step(PWL([q_before], [0.0]), data[1], 0.0, RU, RD, buf))
    for i in 2:n
        isempty(V.x) && return false
        V, q[i], lo[i], hi[i] = ramp_step(V, data[i], 0.0, RU, RD, buf)
    end
    isempty(V.x) && return false
    p = zeros(n)
    p[n] = V.x[argmin(V.y)]
    for i in n:-1:2
        p[i-1] = clamp(q[i], max(lo[i], p[i] - RU), min(hi[i], p[i] + RD))
    end
    for i in 1:n
        t, Pt = first_t + i - 1, data[i]
        left = p[i]
        for kk in Pt.order
            s = min(Pt.u[kk], max(left, 0.0))
            x[S.segprod[t, kk]] = s
            left -= s
        end
        x[S.prod_above[t]] = p[i]
        if Pt.r != 0
            prev = i == 1 ? q_before : p[i-1]
            r = (i == 1 && free) || !Rs.mlr ? Pt.C - p[i] : min(Pt.C - p[i], RU + prev - p[i])
            x[Pt.r] = max(r, 0.0)
        end
    end
    return true
end
