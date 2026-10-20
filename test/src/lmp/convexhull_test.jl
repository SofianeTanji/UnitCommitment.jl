# UnitCommitment.jl: Optimization Package for Security-Constrained Unit Commitment
# Copyright (C) 2020-2026, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.

using UnitCommitment, HiGHS, JuMP, SCIP

function _ch_optimizer()
    return optimizer_with_attributes(
        HiGHS.Optimizer,
        "log_to_console" => false,
        "mip_rel_gap" => 1e-9,
    )
end

# Optimal value of the full model, with the coupling rows dualized at π if
# given, and with integrality relaxed if `relax`
function _ch_full_value(instance, oracle; π = nothing, relax = false)
    m = UnitCommitment.build_model(instance).inner
    if π !== nothing
        penalty = AffExpr()
        for (i, (name, k)) in enumerate(oracle.keys)
            co = constraint_object(m[name][k])
            add_to_expression!(penalty, -π[i], co.func - MOI.constant(co.set))
            delete(m, m[name][k])
        end
        @objective(m, Min, m[:obj] + penalty)
    end
    relax && relax_integrality(m)
    set_optimizer(m, _ch_optimizer())
    optimize!(m)
    return objective_value(m)
end

@testfunction lmp_convexhull_test begin
    sf = UnitCommitment.ShiftFactorsTransmissionExt(lazy = false)
    cases = [
        # two scenarios, reserves
        [fixture("case14/base.json"), fixture("case14/congested.json")] =>
            [sf],
        # flexiramp
        fixture("case14/flex.json") => [
            UnitCommitment.ThermalExt(ramping = UnitCommitment.NoRamping()),
            UnitCommitment.WanHob2016.FlexirampExt(),
            sf,
        ],
        # storage
        fixture("case14-storage.json.gz") => [sf],
    ]
    for (path, extensions) in cases
        instance = UnitCommitment.read(path; extensions)
        oracle = UnitCommitment.ConvexHullOracle(
            instance;
            optimizer = _ch_optimizer(),
        )
        lp_value = _ch_full_value(instance, oracle; relax = true)

        # Blocks add up to the full model with the coupling rows dualized
        r = UnitCommitment.evaluate!(oracle, oracle.π0)
        @test r.value ≈ _ch_full_value(instance, oracle; π = oracle.π0) rtol =
            1e-6
        @test r.bound <= r.value + 1e-6 * abs(r.value)
        @test r.value >= lp_value - 1e-6 * abs(lp_value)

        # Supergradient inequality
        π1 = max.(oracle.π0 .+ 1.0, oracle.lower)
        r1 = UnitCommitment.evaluate!(oracle, π1)
        @test r1.value <=
              r.value + r.grad' * (π1 - oracle.π0) + 1e-6 * abs(r.value)

        # Smoothed block (MIQP) solved by SCIP
        blk = oracle.blocks[1]
        x = [v for v in all_variables(blk.model) if !is_fixed(v)]
        extra = sum(0.5 * (v - blk.x_lp[v])^2 for v in x)
        set_optimizer(blk.model, SCIP.Optimizer)
        set_silent(blk.model)
        s = UnitCommitment.solve_block!(oracle, 1, oracle.π0; extra)
        @test s.status == OPTIMAL
        @test s.value >= r.blocks[1].value - 1e-6 * abs(r.blocks[1].value)

        # LP relaxation: x_lp is optimal and π0 maximizes the relaxed dual
        @test sum(value(v -> b.x_lp[v], b.cost) for b in oracle.blocks) ≈
              lp_value rtol = 1e-6
        for b in oracle.blocks
            relax_integrality(b.model)
            set_optimizer(b.model, _ch_optimizer())
        end
        @test UnitCommitment.evaluate!(oracle, oracle.π0).value ≈ lp_value rtol =
            1e-6
    end

    # Unbounded system block: reserve price above the shortfall penalty
    instance =
        UnitCommitment.read(fixture("case14/base.json"); extensions = [sf])
    oracle =
        UnitCommitment.ConvexHullOracle(instance; optimizer = _ch_optimizer())
    π = copy(oracle.π0)
    π[findfirst(k -> k[1] == :eq_min_reserve, oracle.keys)] = 1e6
    @test UnitCommitment.evaluate!(oracle, π).value == -Inf

    # Lazy transmission constraints are not supported
    instance = UnitCommitment.read(fixture("case14/base.json"))
    @test_throws ErrorException UnitCommitment.ConvexHullOracle(
        instance;
        optimizer = _ch_optimizer(),
    )
end

@testfunction lmp_convexhull_lmp_test begin
    sf = UnitCommitment.ShiftFactorsTransmissionExt(lazy = false)
    ext = UnitCommitment.ConvexHullLMP(
        optimizer = _ch_optimizer(),
        iterations = 5,
    )
    instance =
        UnitCommitment.read(fixture("case14/base.json"); extensions = [sf, ext])
    model = UnitCommitment.build_model(instance; optimizer = _ch_optimizer())
    optimize!(model)
    lmp = solution(model)["LMP: Total (\$/MWh)"]

    # The LMPs are the energy prices found by the subgradient method
    oracle =
        UnitCommitment.ConvexHullOracle(instance; optimizer = _ch_optimizer())
    π, _ = UnitCommitment._ch_subgradient(oracle, oracle.π0, 5, ext.step)
    for (i, (name, (_, b, t))) in enumerate(oracle.keys)
        name == :eq_net_injection || continue
        @test lmp[b][t] ≈ -π[i]
    end

    # The method recovers the dual value lost by perturbing the prices
    L0 = UnitCommitment.evaluate!(oracle, oracle.π0).value
    π1 = max.(oracle.π0 .+ 20 .* (-1) .^ eachindex(oracle.π0), oracle.lower)
    L1 = UnitCommitment.evaluate!(oracle, π1).value
    _, best = UnitCommitment._ch_subgradient(oracle, π1, 50, 100.0)
    @test best >= L1 + 0.9 * (L0 - L1)
end
