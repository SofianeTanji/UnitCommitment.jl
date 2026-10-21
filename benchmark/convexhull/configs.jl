# The benchmarked configurations, their tuned parameter, and the instances.
#
# Every configuration has at most one tuned parameter; the others are fixed
# below. The bundle methods are run with all combinations of their switches:
# cuts (1: aggregated, 4: clustered, 0: one model per block), preconditioning
# and inexact oracle (inexact = 0.1, max_gap = 1e-2).
#
# Tuned parameters are relative to the instance: the value v of the grid is
# passed to the method as v · scale, where scale is P = ‖π0‖∞ for prices
# (steps and distances), F = |L(π0)| for objective values, and P² / F for the
# proximal and gradient steps (π moves by about P when the step times the
# supergradient is about F / P). Grids: :log, 11 values 10^-5, ..., 10^5;
# :lin, 11 equally spaced values over `range`. The fine grid adds the points
# at a quarter and at half a coarse step on each side of the best coarse value.

struct Config
    label::String
    method::String
    fixed::Dict{Symbol,Any}
    tuned::Union{Nothing,Symbol}
    grid::Symbol                       # :log or :lin
    range::Tuple{Float64,Float64}      # log10 bounds for :log
    scale::Symbol                      # :P, :F, :P2F or :one
end

Config(label, method, fixed; tuned = nothing, grid = :log, range = (-5.0, 5.0), scale = :one) =
    Config(label, method, Dict{Symbol,Any}(pairs(fixed)), tuned, grid, range, scale)

function bundle_configs()
    configs = Config[]
    variants = [
        ("blm", "blm", (;)),
        ("bplm", "bplm", (;)),
        ("bpm", "bpm", (; test = :classic)),
        ("moreau-pp", "bpm", (; test = :ms)),
        ("moreau-fgm", "bpm", (; test = :ms, accelerate = true)),
    ]
    for (base, method, options) in variants, cuts in (1, 4, 0),
        precondition in (false, true), inexact in (false, true)

        label =
            base * "-" * Dict(1 => "agg", 4 => "k4", 0 => "block")[cuts] *
            (precondition ? "-pc" : "") * (inexact ? "-inexact" : "")
        fixed = Dict{Symbol,Any}(pairs(options))
        fixed[:cuts] = cuts
        fixed[:precondition] = precondition
        fixed[:inexact] = inexact ? 0.1 : 0.0
        push!(
            configs,
            method == "bpm" ?
            Config(label, method, fixed; tuned = :t, scale = :P2F) :
            Config(label, method, fixed; tuned = :α, grid = :lin, range = (0.1, 1.0)),
        )
    end
    return configs
end

const CONFIGS = [
    Config("subgradient", "subgradient", (;); tuned = :step, scale = :P),
    Config("subgradient_last", "subgradient_last", (;); tuned = :R, scale = :P),
    Config("polyak_estimated", "polyak_estimated", (;); tuned = :α, scale = :F),
    Config("d_adaptation", "d_adaptation", (;); tuned = :d0, scale = :P),
    Config("dowg", "dowg", (;); tuned = :d0, scale = :P),
    Config("fgm_ch", "fgm_ch", (; σ = 1e-8); tuned = :η, scale = :P2F),
    bundle_configs()...,
    Config("cg-single", "cg", (; multicut = false)),
    Config("cg-multicut", "cg", (; multicut = true)),
    Config("cg-stabilized", "cg", (; multicut = true); tuned = :α, grid = :lin, range = (0.1, 0.9)),
]

const CONFIG = Dict(c.label => c for c in CONFIGS)

# Datasets: or-lib, pglib-uc/ca, pglib-uc/ferc and matpower/case14
dataset(name) = startswith(name, "or-lib") ? "or-lib" : join(split(name, "/")[1:2], "/")

# One tuning instance per dataset: the instance with the median relative gap
# (L* - L(π0)) / |L*| in the benchmark set
const TUNING = [
    "or-lib/200_0_7_w",
    "pglib-uc/ca/2014-12-01_reserves_3",
    "pglib-uc/ferc/2015-11-02_lw",
    "matpower/case14/2017-09-04",
]

# The benchmark set: the instances solved by dw.jl on which L(π0) is not
# already within `tol` of L*
function benchmark_instances(; tol = 1e-6)
    names = String[]
    for name in readlines("instances.txt")
        file = joinpath("results", "$name.json")
        isfile(file) || continue
        r = JSON.parsefile(file)
        r["status"] == "optimal" || continue
        L_star, L0 = r["lower_bound"], r["dual_value_at_lp_duals"]
        L_star - L0 > tol * abs(L_star) && push!(names, name)
    end
    return names
end

coarse_grid(c::Config) = round.(
    c.grid == :log ? 10.0 .^ range(c.range..., 11) : collect(range(c.range..., 11));
    sigdigits = 6,
)

function fine_grid(c::Config, best)
    lo, hi = c.range
    h = (hi - lo) / 10
    x = c.grid == :log ? log10(best) : best
    points = [clamp(x + d * h, lo, hi) for d in (-0.5, -0.25, 0.25, 0.5)]
    points = unique(filter(p -> !isapprox(p, x), points))
    return round.(c.grid == :log ? 10.0 .^ points : points; sigdigits = 6)
end
