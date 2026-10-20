# UnitCommitment.jl: Optimization Package for Security-Constrained Unit Commitment
# Copyright (C) 2020-2026, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.

using JuMP
using LinearAlgebra
using SparseArrays

# Coupling rows of the Lagrangian dual. In the requirement rows, the shortfall
# variable belongs to the system block, not to the units.
const _CH_ROWS = (
    :eq_net_injection,
    :eq_min_reserve,
    :eq_min_flexiramp_up,
    :eq_min_flexiramp_down,
)
const _CH_SHORTFALL = Dict(
    :eq_min_reserve => :reserve_shortfall,
    :eq_min_flexiramp_up => :upflexiramp_shortfall,
    :eq_min_flexiramp_down => :dwflexiramp_shortfall,
)

"""
    ConvexHullBlock

One block of the Lagrangian decomposition used by [`ConvexHullOracle`](@ref):
a thermal unit, a storage unit, or the system block (buses, network, reserve
shortfalls and all other components).

# Fields
- `name::String`: unit name, or `"system"`.
- `model::JuMP.Model`: the block model, without the coupling rows.
- `cost::AffExpr`: the block cost.
- `rows::Vector{Int}`, `g::Vector{AffExpr}`: the block's share `g[i]` of
  `f(x) - rhs` for coupling row `rows[i]`. The shares of all blocks add up to
  the coupling rows `f(x) - rhs` of the full model.
- `x_lp::Dict{VariableRef,Float64}`: value of each block variable in the
  optimal solution of the LP relaxation of the full model.
"""
struct ConvexHullBlock
    name::String
    model::JuMP.Model
    cost::AffExpr
    rows::Vector{Int}
    g::Vector{AffExpr}
    x_lp::Dict{VariableRef,Float64}
end

"""
    ConvexHullOracle(instance; optimizer)

Lagrangian dual of the unit commitment problem in which the coupling rows
(energy balance, reserve and flexiramp requirements) are dualized:

    L(π) = Σ_b min { cost_b(x_b) - π' g_b(x_b) : x_b feasible for block b }.

Convex hull prices maximize `L` over `π ≥ lower`. Prices follow the JuMP sign
convention for duals, so that the LP relaxation duals `π0` are feasible and
the energy price of an `eq_net_injection` row is `-π`, as in
[`ConventionalLMP`](@ref).

Each thermal unit (with its flexiramp variables) and each storage unit is a
block; everything else forms the `"system"` block. `optimizer` is set on all
block models and on the LP relaxation; it can be changed on a block with
`set_optimizer(oracle.blocks[b].model, ...)`.

Transmission constraints must be built in the model: use
`ShiftFactorsTransmissionExt(lazy = false)`. AC transmission is not supported.

# Fields
- `keys::Vector{Tuple{Symbol,Any}}`: coupling row of each price, e.g.
  `(:eq_net_injection, ("s1", "b1", 1))`.
- `lower::Vector{Float64}`: `-Inf` for equality rows, `0` for requirements.
- `π0::Vector{Float64}`: coupling row duals in the LP relaxation.
- `blocks::Vector{ConvexHullBlock}`.
"""
struct ConvexHullOracle
    keys::Vector{Tuple{Symbol,Any}}
    lower::Vector{Float64}
    π0::Vector{Float64}
    blocks::Vector{ConvexHullBlock}
end

function ConvexHullOracle(instance::UnitCommitmentInstance; optimizer)
    if any(
        e -> e isa ShiftFactorsTransmissionExt && e.lazy,
        instance.extensions,
    )
        error(
            "ConvexHullOracle requires ShiftFactorsTransmissionExt(lazy = false)",
        )
    end
    if any(e -> e isa ACTransmissionExt, instance.extensions)
        error("ConvexHullOracle does not support AC transmission")
    end

    # LP relaxation of the full model: initial prices and reference point
    lp = build_model(instance).inner
    relax_integrality(lp)
    set_optimizer(lp, optimizer)
    set_silent(lp)
    JuMP.optimize!(lp)
    if !is_solved_and_feasible(lp; dual = true)
        error("LP relaxation: $(termination_status(lp))")
    end
    keys = Tuple{Symbol,Any}[]
    lower = Float64[]
    π0 = Float64[]
    for name in _CH_ROWS
        haskey(object_dictionary(lp), name) || continue
        for (k, c) in lp[name]
            push!(keys, (name, k))
            push!(lower, constraint_object(c).set isa MOI.EqualTo ? -Inf : 0.0)
            push!(π0, dual(c))
        end
    end
    row = Dict(k => i for (i, k) in enumerate(keys))
    lp_value = Dict(k => value(v) for (v, k) in _ch_variable_keys(lp))

    # Blocks are built from one private copy of the instance, in which the
    # unit lists are set to the units of the block
    I = deepcopy(instance)
    lists = [
        (
            thermal = sc[:thermal],
            storage = sc[:storage],
            reserves = [
                r => r.thermal_units for
                r in [sc[:reserves]; get(sc, :flexiramp_reserves, [])]
            ],
        ) for sc in I.scenarios
    ]
    thermal_exts = [
        e for e in instance.extensions if
        e isa ThermalExt || e isa WanHob2016.FlexirampExt
    ]
    storage_exts = [e for e in instance.extensions if e isa StorageExt]
    blocks = ConvexHullBlock[]
    for g in instance.scenarios[1][:thermal]
        _ch_select!(I, lists; thermal = (g.name,))
        push!(blocks, _ch_unit_block(g.name, I, thermal_exts, row, lp_value))
    end
    for s in instance.scenarios[1][:storage]
        _ch_select!(I, lists; storage = (s.name,))
        push!(blocks, _ch_unit_block(s.name, I, storage_exts, row, lp_value))
    end
    push!(blocks, _ch_system_block(_ch_select!(I, lists), row, lp_value))
    for blk in blocks
        set_optimizer(blk.model, optimizer)
        set_silent(blk.model)
    end
    return ConvexHullOracle(keys, lower, π0, blocks)
end

"""
    solve_block!(oracle, b, π; extra = nothing)

Minimize `cost_b(x) - π' g_b(x) + extra` over the feasible set of block `b`,
where `extra` is an optional expression in the block variables (e.g. a
smoothing term). The block model keeps its solution, which can be queried
with `value`.

Returns `(; status, value, bound, a)`: the termination status, the objective
value, a lower bound on the optimal value (the solver's dual bound when the
block has integer variables), and the share `a = g_b(x)` of the solution as a
sparse vector indexed like `oracle.keys`. If the block is unbounded, `π` is
outside the domain of the dual: `value` and `bound` are `-Inf` and `a` is zero.
"""
function solve_block!(
    oracle::ConvexHullOracle,
    b::Int,
    π::AbstractVector{<:Real};
    extra = nothing,
)
    blk = oracle.blocks[b]
    obj = copy(blk.cost)
    for (i, gi) in zip(blk.rows, blk.g)
        add_to_expression!(obj, -π[i], gi)
    end
    if extra === nothing
        @objective(blk.model, Min, obj)
    else
        @objective(blk.model, Min, obj + extra)
    end
    JuMP.optimize!(blk.model)
    status = termination_status(blk.model)
    if status in (DUAL_INFEASIBLE, INFEASIBLE_OR_UNBOUNDED)
        return (; status, value = -Inf, bound = -Inf, a = spzeros(length(π)))
    end
    has_values(blk.model) || error("block $(blk.name): $status")
    obj_value = objective_value(blk.model)
    has_int =
        num_constraints(blk.model, VariableRef, MOI.ZeroOne) +
        num_constraints(blk.model, VariableRef, MOI.Integer) > 0
    bound =
        has_int ? objective_bound(blk.model) :
        status == OPTIMAL ? obj_value : -Inf
    a = sparsevec(blk.rows, value.(blk.g), length(oracle.keys))
    return (; status, value = obj_value, bound, a)
end

"""
    evaluate!(oracle, π)

Evaluate the Lagrangian dual at `π` by solving every block with
[`solve_block!`](@ref). Returns `(; value, bound, grad, blocks)`: the sum of
the block values, the sum of the block bounds (`bound ≤ L(π) ≤ value`), the
supergradient `-Σ_b a_b` and the result of each block.
"""
function evaluate!(oracle::ConvexHullOracle, π::AbstractVector{<:Real})
    blocks = [solve_block!(oracle, b, π) for b in eachindex(oracle.blocks)]
    return (;
        value = sum(r.value for r in blocks),
        bound = sum(r.bound for r in blocks),
        grad = -Vector(sum(r.a for r in blocks)),
        blocks,
    )
end

# Set the unit lists of `I` to the given thermal and storage units, taken from
# the original lists
function _ch_select!(I, lists; thermal = (), storage = ())
    for (sc, l) in zip(I.scenarios, lists)
        sc[:thermal] = filter(g -> g.name in thermal, l.thermal)
        sc[:thermal_by_name] = Dict(g.name => g for g in sc[:thermal])
        for (r, units) in l.reserves
            r.thermal_units = filter(g -> g.name in thermal, units)
        end
        sc[:storage] = filter(s -> s.name in storage, l.storage)
    end
    return I
end

function _ch_unit_block(name, I, exts, row, lp_value)
    m = Model()
    m[:obj] = AffExpr()
    bus_keys = [
        (sc.name, b.name, t) for sc in I.scenarios for b in sc[:bus] for
        t in 1:I.time
    ]
    m[:net_injection] = OrderedDict(k => AffExpr() for k in bus_keys)
    m[:net_reactive_injection] = OrderedDict(k => AffExpr() for k in bus_keys)
    for ext in exts
        build_model(m, I, ext)
    end
    rows, g = Int[], AffExpr[]
    # eq_net_injection: ni - net_injection - curtail == load
    for (k, e) in m[:net_injection]
        iszero(e) && continue
        push!(rows, row[(:eq_net_injection, k)])
        push!(g, -e)
    end
    for (name, shortfall) in _CH_SHORTFALL
        haskey(object_dictionary(m), name) || continue
        for (k, c) in m[name]
            f = copy(constraint_object(c).func)
            s = m[shortfall][k]
            if s isa VariableRef
                delete!(f.terms, s)
                fix(s, 0.0; force = true)
            end
            delete(m, c)
            iszero(f) && continue
            push!(rows, row[(name, k)])
            push!(g, f)
        end
    end
    return ConvexHullBlock(name, m, m[:obj], rows, g, _ch_x_lp(m, lp_value))
end

function _ch_system_block(I, row, lp_value)
    m = build_model(I).inner
    rows, g = Int[], AffExpr[]
    for name in _CH_ROWS
        haskey(object_dictionary(m), name) || continue
        for (k, c) in m[name]
            co = constraint_object(c)
            push!(rows, row[(name, k)])
            push!(g, co.func - MOI.constant(co.set))
            delete(m, c)
        end
    end
    return ConvexHullBlock("system", m, m[:obj], rows, g, _ch_x_lp(m, lp_value))
end

# Block variables are matched to the full model by container name and key
function _ch_x_lp(m, lp_value)
    x_lp = Dict(
        v => is_fixed(v) ? fix_value(v) : lp_value[k] for
        (v, k) in _ch_variable_keys(m)
    )
    length(x_lp) == num_variables(m) || error("unnamed variables in block")
    return x_lp
end

function _ch_variable_keys(m)
    return [
        (v, (sym, k)) for
        (sym, obj) in object_dictionary(m) if obj isa AbstractDict for
        (k, v) in obj if v isa VariableRef
    ]
end

function _after_optimize!(
    instance::UnitCommitmentInstance,
    model::UnitCommitmentModel,
    ext::ConvexHullLMP,
)::Nothing
    oracle = ConvexHullOracle(instance; optimizer = ext.optimizer)
    π, _ = _ch_subgradient(oracle, oracle.π0, ext.iterations, ext.step)
    model.data[:lmp] = OrderedDict(
        k => -π[i] for (i, (name, k)) in enumerate(oracle.keys) if
        name == :eq_net_injection
    )
    return _update_solution(instance, model, ConventionalLMP())
end

# Projected subgradient method with step lengths step / k, from π. Stops if π
# leaves the domain of the dual. Returns the prices with the best dual value,
# and that value.
function _ch_subgradient(
    oracle::ConvexHullOracle,
    π::AbstractVector{<:Real},
    iterations::Int,
    step::Real,
)
    best_π, best_value = π, -Inf
    for k in 1:iterations
        r = evaluate!(oracle, π)
        isfinite(r.value) || break
        if r.value > best_value
            best_π, best_value = π, r.value
        end
        norm_g = norm(r.grad)
        norm_g == 0 && break
        π = max.(π .+ step / k .* r.grad ./ norm_g, oracle.lower)
    end
    return best_π, best_value
end
