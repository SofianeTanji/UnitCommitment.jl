# Column-and-row generation (CRG) on the interval extended formulation of the
# thermal units (Knueven, Ostrowski and Watson; Sadykov and Vanderbeck, 2013;
# Paquet, 2021, chapter 6).
#
# A schedule of a unit is a sequence of on-intervals [a, b]. The extended
# formulation of a unit is a flow of value 1 from a source to a sink through
# interval nodes. An arc joins [a, b] to [c, d] if the unit can be off from
# b + 1 to c - 1 (minimum down time, must-run, initial status), and costs the
# startup at c. Each node carries its own copy x of the unit's continuous
# variables, in its dispatch polytope scaled by the node flow γ: the polytope
# of the unit's block model with the commitment fixed to "on exactly during
# [a, b]". The projection of this formulation is the convex hull of the
# unit's schedules. With all the intervals, the master LP (these formulations,
# the blocks without integer variables, and the coupling rows) therefore has
# the value max L, and its coupling row duals are convex hull prices.
#
# CRG generates the intervals. At the duals λ of the restricted master, the
# blocks are solved (L(λ) is a lower bound on max L), and the intervals of the
# unit schedules found enter the master, with their rows and their arcs to the
# other intervals of the unit. The master value is an upper bound, and the
# method stops when it is at most the best lower bound.
#
# The master is a HiGHS model, built in batches through the C API, whose
# simplex basis is kept between rounds. The dispatch polytopes are sliced from
# one sparse matrix per unit (lp_matrix_data of the block model), without the
# variables that the commitment forces to zero and the rows that the bounds of
# the other variables imply.

const HInt = HiGHS.HighsInt

# Master LP, with the columns and rows of the current batch, whose entries are
# kept as (row, column, value) until the batch is sent to HiGHS
mutable struct CRGMaster
    h::Ptr{Cvoid}
    ncol::Int
    nrow::Int
    cost::Vector{Float64}
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    row_lo::Vector{Float64}
    row_hi::Vector{Float64}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
end

function CRGMaster()
    h = HiGHS.Highs_create()
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", false)
    HiGHS.Highs_setIntOptionValue(h, "threads", 1)
    return CRGMaster(h, 0, 0, [], [], [], [], [], [], [], [])
end

function new_col!(M::CRGMaster, cost, lo, hi)
    push!(M.cost, cost)
    push!(M.col_lo, lo)
    push!(M.col_hi, hi)
    return M.ncol + length(M.cost)
end

function new_row!(M::CRGMaster, lo, hi)
    push!(M.row_lo, lo)
    push!(M.row_hi, hi)
    return M.nrow + length(M.row_lo)
end

function entry!(M::CRGMaster, i, j, v)
    v == 0 && return
    push!(M.I, i)
    push!(M.J, j)
    push!(M.V, v)
    return
end

# Sends the batch: the new columns with their entries in the existing rows,
# then the new rows with their entries in all columns
function flush!(M::CRGMaster)
    nc, nr = length(M.cost), length(M.row_lo)
    old = M.I .<= M.nrow
    any(old .& (M.J .<= M.ncol)) && error("entry in an existing row and column")
    if nc > 0
        A = sparse(M.I[old], M.J[old] .- M.ncol, M.V[old], max(M.nrow, 1), nc)
        status = HiGHS.Highs_addCols(M.h, nc, M.cost, M.col_lo, M.col_hi, nnz(A),
            HInt.(A.colptr[1:end-1] .- 1), HInt.(A.rowval .- 1), A.nzval)
        status >= 0 || error("Highs_addCols: $status")
    end
    M.ncol += nc
    if nr > 0
        A = sparse(M.J[.!old], M.I[.!old] .- M.nrow, M.V[.!old], M.ncol, nr)
        status = HiGHS.Highs_addRows(M.h, nr, M.row_lo, M.row_hi, nnz(A),
            HInt.(A.colptr[1:end-1] .- 1), HInt.(A.rowval .- 1), A.nzval)
        status >= 0 || error("Highs_addRows: $status")
    end
    M.nrow += nr
    foreach(empty!, (M.cost, M.col_lo, M.col_hi, M.row_lo, M.row_hi, M.I, M.J, M.V))
    return
end

# Solves the master; returns its value and its row duals. The new columns of a
# batch are at zero in the last basis, which keeps it primal feasible (the
# rows of a node are homogeneous in its flow), so the masters after the first
# are solved by the primal simplex method from the last basis.
function solve!(M::CRGMaster)
    HiGHS.Highs_run(M.h)
    status = HiGHS.Highs_getModelStatus(M.h)
    status == HiGHS.kHighsModelStatusOptimal || error("master: HiGHS model status $status")
    HiGHS.Highs_setIntOptionValue(M.h, "simplex_strategy", 4)
    col_value, col_dual = zeros(M.ncol), zeros(M.ncol)
    row_value, row_dual = zeros(M.nrow), zeros(M.nrow)
    HiGHS.Highs_getSolution(M.h, col_value, col_dual, row_value, row_dual)
    return HiGHS.Highs_getObjectiveValue(M.h), row_dual
end

# A block model as sparse matrices: lo ≤ A x ≤ hi, xlo ≤ x ≤ xhi, cost c'x +
# offset, and its share G x + g0 of the coupling rows (indexed like
# oracle.keys)
struct BlockMatrix
    vars::Vector{VariableRef}
    A::SparseMatrixCSC{Float64,Int}
    At::SparseMatrixCSC{Float64,Int}
    lo::Vector{Float64}
    hi::Vector{Float64}
    xlo::Vector{Float64}
    xhi::Vector{Float64}
    c::Vector{Float64}
    offset::Float64
    G::SparseMatrixCSC{Float64,Int}
    g0::Vector{Float64}
    integer::BitVector
end

function BlockMatrix(oracle, b)
    blk = oracle.blocks[b]
    @objective(blk.model, Min, blk.cost)
    d = lp_matrix_data(blk.model)
    column = Dict(v => j for (j, v) in enumerate(d.variables))
    I, J, V = Int[], Int[], Float64[]
    g0 = zeros(length(oracle.keys))
    for (i, gi) in zip(blk.rows, blk.g)
        g0[i] += gi.constant
        for (v, a) in gi.terms
            push!(I, i)
            push!(J, column[v])
            push!(V, a)
        end
    end
    n = length(d.variables)
    integer = falses(n)
    integer[d.integers] .= true
    integer[d.binaries] .= true
    xlo, xhi = copy(d.x_lower), copy(d.x_upper)
    xlo[d.binaries] .= max.(xlo[d.binaries], 0.0)
    xhi[d.binaries] .= min.(xhi[d.binaries], 1.0)
    return BlockMatrix(d.variables, d.A, sparse(d.A'), d.b_lower, d.b_upper, xlo, xhi,
        d.c, d.c_offset, sparse(I, J, V, length(oracle.keys), n), g0, integer)
end

# A thermal unit in the master: its block matrix, the columns of its binaries
# by period, the rows that involve its startup categories, whether it can be
# off at period 1, and its source row and interval nodes (node flow column, in
# and out rows)
struct CRGUnit
    b::Int
    unit::Any
    B::BlockMatrix
    is_on::Vector{Int}
    switch_on::Vector{Int}
    switch_off::Vector{Int}
    startup::Matrix{Int}
    startup_rows::BitVector
    off_at_1::Bool
    source::Int
    nodes::Dict{Tuple{Int,Int},NTuple{3,Int}}
end

function CRGUnit(oracle, b, B::BlockMatrix, unit, M::CRGMaster)
    m = oracle.blocks[b].model
    column = Dict(v => j for (j, v) in enumerate(B.vars))
    T = length(unit.must_run)
    S = length(unit.startup_categories)
    is_on = [column[m[:is_on][unit.name, t]] for t in 1:T]
    switch_on = [column[m[:switch_on][unit.name, t]] for t in 1:T]
    switch_off = [column[m[:switch_off][unit.name, t]] for t in 1:T]
    startup = [column[m[:startup][unit.name, t, s]] for t in 1:T, s in 1:S]
    startup_rows = falses(size(B.A, 1))
    for j in startup, k in nzrange(B.A, j)
        startup_rows[B.A.rowval[k]] = true
    end
    # The rows that no node holds (all their variables forced to zero) bind
    # the off periods; those of the initial conditions (e.g. the ramp down from
    # the initial power) can forbid being off at period 1
    v = zeros(length(B.vars))
    unit.initial_status > 0 && (v[switch_off[1]] = 1)
    shift = B.A * v
    zero = forced_zero(B, B.lo .- shift, B.hi .- shift, startup_rows)
    off_at_1 = all(axes(B.At, 2)) do r
        startup_rows[r] || any(k -> !zero[B.At.rowval[k]], nzrange(B.At, r)) ||
            B.lo[r] - 1e-6 <= shift[r] <= B.hi[r] + 1e-6
    end
    source = new_row!(M, 1.0, 1.0)
    return CRGUnit(b, unit, B, is_on, switch_on, switch_off, startup, startup_rows,
        off_at_1, source, Dict())
end

# Cost of the arc from interval `from` (nothing: the source) to interval `to`
# (nothing: the sink), or nothing if the unit cannot be off in between. The
# shutdown at b + 1 is in the cost of the node [a, b].
function arc_cost(U::CRGUnit, from, to)
    g = U.unit
    T = length(U.is_on)
    on0 = g.initial_status > 0
    c = to === nothing ? T + 1 : to[1]
    cost = 0.0
    if from === nothing
        on0 && c == 1 && return 0.0
        U.off_at_1 || return nothing
        if on0
            # shutdown at period 1, allowed once the initial minimum up time is over
            g.initial_status >= g.min_uptime || return nothing
            k = 1
            cost += U.B.c[U.switch_off[1]]
        else
            k = g.initial_status + 1
        end
    else
        from[2] == T && return to === nothing ? 0.0 : nothing
        k = from[2] + 1
    end
    for t in max(k, 1):(c-1)
        (g.must_run[t] || g.commitment_status[t] == true) && return nothing
    end
    to === nothing && return cost
    c - k >= g.min_downtime || return nothing
    # cheapest startup category allowed by the time since the shutdown at k
    S = length(g.startup_categories)
    delay = [cat.delay for cat in g.startup_categories]
    startup = U.B.c[U.startup[c, S]]
    for s in 1:(S-1)
        if c - delay[s+1] + 1 <= k <= c - delay[s]
            startup = min(startup, U.B.c[U.startup[c, s]])
        end
    end
    return cost + startup
end

# Variables that the fixed commitment forces to zero: fixed at zero, or in a
# row Σ a_j x_j ≤ 0 (or ≥ 0) with x ≥ 0 and all a_j > 0 (or < 0)
function forced_zero(B::BlockMatrix, lo, hi, skip)
    zero = B.integer .| ((B.xlo .== 0) .& (B.xhi .== 0))
    changed = true
    while changed
        changed = false
        for r in axes(B.At, 2)
            skip[r] && continue
            ks = [k for k in nzrange(B.At, r) if !zero[B.At.rowval[k]]]
            isempty(ks) && continue
            js, as = B.At.rowval[ks], B.At.nzval[ks]
            all(B.xlo[js] .>= 0) || continue
            if (abs(hi[r]) <= 1e-9 && all(as .> 0)) || (abs(lo[r]) <= 1e-9 && all(as .< 0))
                zero[js] .= true
                changed = true
            end
        end
    end
    return zero
end

# Adds the node of interval [a, b] of unit U, and its arcs to the source, the
# sink and the other nodes
function add_node!(M::CRGMaster, U::CRGUnit, a, b, coupling)
    B = U.B
    T = length(U.is_on)
    v = zeros(length(B.vars))
    v[U.is_on[a:b]] .= 1
    (a > 1 || U.unit.initial_status <= 0) && (v[U.switch_on[a]] = 1)
    b < T && (v[U.switch_off[b+1]] = 1)
    shift = B.A * v
    lo, hi = B.lo .- shift, B.hi .- shift
    zero = forced_zero(B, lo, hi, U.startup_rows)
    # The node polytope is a cone in (x, γ): its slice at γ = 1 is described by
    # bounds l ≤ x ≤ u (the variable bounds and the rows with one variable), and
    # by the rows with more variables that these bounds do not imply
    l, u = copy(B.xlo), copy(B.xhi)
    rows = Tuple{Int,Vector{Int}}[]
    for r in axes(B.At, 2)
        U.startup_rows[r] && continue
        ks = [k for k in nzrange(B.At, r) if !zero[B.At.rowval[k]] && B.At.nzval[k] != 0]
        if length(ks) == 1
            j, α = B.At.rowval[ks[1]], B.At.nzval[ks[1]]
            lj, uj = α > 0 ? (lo[r] / α, hi[r] / α) : (hi[r] / α, lo[r] / α)
            l[j], u[j] = max(l[j], lj), min(u[j], uj)
        elseif length(ks) > 1
            push!(rows, (r, ks))
        end
    end
    kept = Tuple{Int,Vector{Int},Bool,Bool}[]
    used = falses(length(B.vars))
    for (r, ks) in rows
        js, as = B.At.rowval[ks], B.At.nzval[ks]
        low = sum(α > 0 ? α * l[j] : α * u[j] for (j, α) in zip(js, as))
        up = sum(α > 0 ? α * u[j] : α * l[j] for (j, α) in zip(js, as))
        need_lo, need_hi = isfinite(lo[r]) && low < lo[r], isfinite(hi[r]) && up > hi[r]
        (need_lo || need_hi) || continue
        push!(kept, (r, ks, need_lo, need_hi))
        used[js] .= true
    end
    # the variables in no kept row, no coupling row and with no cost are left
    # out (any value within their bounds will do)
    γ = new_col!(M, dot(B.c, v), 0.0, Inf)
    x = Dict{Int,Int}()
    for j in findall(.!zero)
        used[j] || B.c[j] != 0 || !isempty(nzrange(B.G, j)) || continue
        x[j] = new_col!(M, B.c[j], l[j] >= 0 ? 0.0 : -Inf, u[j] <= 0 ? 0.0 : Inf)
        if l[j] == u[j]
            l[j] == 0 || row!(M, [x[j], γ], [1.0, -l[j]], 0.0, 0.0)
        else
            isfinite(l[j]) && l[j] != 0 && row!(M, [x[j], γ], [1.0, -l[j]], 0.0, Inf)
            isfinite(u[j]) && u[j] != 0 && row!(M, [x[j], γ], [1.0, -u[j]], -Inf, 0.0)
        end
    end
    for (r, ks, need_lo, need_hi) in kept
        cols = [x[B.At.rowval[k]] for k in ks]
        vals = B.At.nzval[ks]
        if need_lo && need_hi && lo[r] == hi[r]
            row!(M, [cols; γ], [vals; -lo[r]], 0.0, 0.0)
        else
            need_lo && row!(M, [cols; γ], [vals; -lo[r]], 0.0, Inf)
            need_hi && row!(M, [cols; γ], [vals; -hi[r]], -Inf, 0.0)
        end
    end
    # coupling rows: G x + (G v) γ (the constant shares g0 are in the row
    # bounds)
    Gv = B.G * v
    for i in findall(!iszero, Gv)
        entry!(M, coupling[i], γ, Gv[i])
    end
    for (j, col) in x, k in nzrange(B.G, j)
        entry!(M, coupling[B.G.rowval[k]], col, B.G.nzval[k])
    end
    # flow conservation: γ = inflow = outflow
    in_row, out_row = new_row!(M, 0.0, 0.0), new_row!(M, 0.0, 0.0)
    entry!(M, in_row, γ, 1.0)
    entry!(M, out_row, γ, 1.0)
    U.nodes[(a, b)] = (γ, in_row, out_row)
    for (from, to) in [(nothing, (a, b)); ((a, b), nothing);
        [((a, b), n) for n in keys(U.nodes) if n != (a, b)];
        [(n, (a, b)) for n in keys(U.nodes) if n != (a, b)]]
        add_arc!(M, U, from, to)
    end
    return
end

function row!(M::CRGMaster, cols, vals, lo, hi)
    r = new_row!(M, lo, hi)
    foreach((j, v) -> entry!(M, r, j, v), cols, vals)
    return r
end

function add_arc!(M::CRGMaster, U::CRGUnit, from, to)
    cost = arc_cost(U, from, to)
    cost === nothing && return false
    f = new_col!(M, cost, 0.0, Inf)
    from === nothing ? entry!(M, U.source, f, 1.0) : entry!(M, U.nodes[from][3], f, -1.0)
    to === nothing || entry!(M, U.nodes[to][2], f, -1.0)
    return true
end

# On-intervals of the schedule of unit U in its last block solution
function intervals(U::CRGUnit)
    on = [value(U.B.vars[j]) > 0.5 for j in U.is_on]
    starts = [t for t in eachindex(on) if on[t] && (t == 1 || !on[t-1])]
    return [(a, something(findnext(!, on, a), length(on) + 1) - 1) for a in starts]
end

# Adds the intervals of the last block solutions that are not in the master;
# returns how many were added
function add_intervals!(M::CRGMaster, units, coupling)
    added = 0
    for U in units
        path = intervals(U)
        for (from, to) in zip([nothing; path], [path; nothing])
            arc_cost(U, from, to) === nothing &&
                error("unit $(U.unit.name): schedule rejected between $from and $to")
        end
        for (a, b) in path
            haskey(U.nodes, (a, b)) && continue
            add_node!(M, U, a, b, coupling)
            added += 1
        end
    end
    flush!(M)
    return added
end

# Adds a block without integer variables (the system block, units with a
# fixed commitment) to the master
function add_lp_block!(M::CRGMaster, B::BlockMatrix, coupling)
    any(B.integer) && error("block with integer variables")
    cols = [new_col!(M, B.c[j], B.xlo[j], B.xhi[j]) for j in eachindex(B.vars)]
    for r in axes(B.At, 2)
        ks = nzrange(B.At, r)
        row!(M, cols[B.At.rowval[ks]], B.At.nzval[ks], B.lo[r], B.hi[r])
    end
    for (i, j, a) in zip(findnz(B.G)...)
        entry!(M, coupling[i], cols[j], a)
    end
    return
end

function crg(oracle, π0, stop::Stop; instance)
    trace = Trace(π0)
    length(instance.scenarios) == 1 || error("CRG supports one scenario")
    thermal = instance.scenarios[1][:thermal_by_name]
    M = CRGMaster()
    n = length(oracle.keys)

    # Coupling rows, shifted by the constant shares of the blocks; the slacks
    # keep the first masters feasible
    lp_blocks, units = BlockMatrix[], CRGUnit[]
    for (b, blk) in enumerate(oracle.blocks)
        B = BlockMatrix(oracle, b)
        if any(B.integer)
            haskey(thermal, blk.name) || error("block $(blk.name): not a thermal unit")
            push!(units, CRGUnit(oracle, b, B, thermal[blk.name], M))
        else
            push!(lp_blocks, B)
        end
    end
    constant = sum(B.g0 for B in [lp_blocks; [U.B for U in units]]; init = zeros(n))
    coupling = [
        new_row!(M, -constant[i], oracle.lower[i] == 0 ? Inf : -constant[i]) for i in 1:n
    ]
    flush!(M)
    penalty = 1e3 * (1 + maximum(abs, π0))
    for i in 1:n
        entry!(M, coupling[i], new_col!(M, penalty, 0.0, Inf), 1.0)
        oracle.lower[i] == 0 || entry!(M, coupling[i], new_col!(M, penalty, 0.0, Inf), -1.0)
    end
    foreach(B -> add_lp_block!(M, B, coupling), lp_blocks)
    for U in units
        add_arc!(M, U, nothing, nothing)
    end
    offset = sum(B.offset for B in lp_blocks; init = 0.0) + sum(U.B.offset for U in units; init = 0.0)
    HiGHS.Highs_changeObjectiveOffset(M.h, offset)
    flush!(M)

    r = evaluate!(trace, oracle, π0, stop)
    r.stop && return trace
    add_intervals!(M, units, coupling)
    try
        while true
            value, dual = solve!(M)
            trace.U = min(trace.U, value)
            λ = max.(dual[coupling], oracle.lower)
            r = evaluate!(trace, oracle, λ, stop)
            r.stop && break
            if value <= maximum(trace.bound) + 1e-9 * abs(value)
                trace.status = "optimal"
                break
            end
            if add_intervals!(M, units, coupling) == 0
                trace.status = "no new interval"
                break
            end
        end
    finally
        HiGHS.Highs_destroy(M.h)
    end
    return trace
end
