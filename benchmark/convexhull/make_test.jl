# Held-out test instances: randomized variants of the pglib-uc/ca and
# pglib-uc/ferc instances of instances.txt, by the randomization of
# UnitCommitment.XavQiuAhm2021 (costs ×U(0.95, 1.05) per unit, load shares,
# load profile), except for the peak load: ρ times the peak of the original
# instance, with ρ ~ U(0.925, 1.075), instead of U(0.555, 0.645) times the
# capacity (the peak is 0.34 to 0.78 times the capacity on ferc).
#
# The variant `s` of the `i`-th base instance (in instances.txt order) uses
# MersenneTwister(100i + s) and is written to
# instances/pglib-uc/<ca|ferc>-rand/<base>_s<s>.json.gz. Each file is checked
# against UnitCommitment.randomize! with the same seed. The names are written
# to instances_test.txt, after its lines of other instances (the Belgian ones
# of make_belgian.jl).
#
# Usage: julia --project=. make_test.jl [<variants per instance>] (default 2)

using UnitCommitment, JSON, Random
const UC = UnitCommitment
const R = UC.XavQiuAhm2021
const D = R.Distributions
const Z = UC.CodecZlib

scale(x::Number, α) = x * α
scale(x::AbstractVector, α) = [scale(y, α) for y in x]

lastpoint(c) = c isa AbstractVector ? c[end] : c
capacity(g) = (c = lastpoint(g["Production cost curve (MW)"]); c isa AbstractVector ? maximum(c) : c)

# Randomization of XavQiuAhm2021, on the JSON of a single-scenario instance,
# drawing in the same order as UC.randomize!
function randomize_json!(json, rng, method::R.Randomization)
    for g in values(json["Generators"])
        α = rand(rng, method.cost)
        g["Production cost curve (\$)"] = scale(g["Production cost curve (\$)"], α)
        haskey(g, "Startup costs (\$)") && (g["Startup costs (\$)"] = scale(g["Startup costs (\$)"], α))
    end
    buses = collect(values(json["Buses"]))
    T = json["Parameters"]["Time (h)"]
    load(b) = (l = b["Load (MW)"]; l isa AbstractVector ? Float64.(l) : fill(Float64(l), T))
    for b in buses
        b["Load (MW)"] = load(b)
    end
    α = rand(rng, method.load_share, length(buses))
    for t in 1:T
        total = sum(b["Load (MW)"][t] for b in buses)
        den = sum(b["Load (MW)"][t] / total * α[i] for (i, b) in enumerate(buses))
        for (i, b) in enumerate(buses)
            b["Load (MW)"][t] *= α[i] / den
        end
    end
    system = [1.0]
    for t in 2:T
        k = (t - 1) % length(method.load_profile_mu) + 1
        push!(system, system[t-1] * rand(rng, D.Normal(method.load_profile_mu[k], method.load_profile_sigma[k])))
    end
    C = sum(capacity, values(json["Generators"]))
    system = system ./ maximum(system) .* (rand(rng, method.peak_load) * C)
    previous = [sum(b["Load (MW)"][t] for b in buses) for t in 1:T]
    for b in buses, t in 1:T
        b["Load (MW)"][t] *= system[t] / previous[t]
    end
    return json
end

function check(file, base, seed, method)
    expected = UC.read(base)
    UC.randomize!(expected, method; rng = MersenneTwister(seed))
    got = UC.read(file)
    a, b = expected.scenarios[1], got.scenarios[1]
    close(x, y) = isapprox(x, y; rtol = 1e-12, atol = 1e-9)
    all(close(x.load, y.load) for (x, y) in zip(a[:bus], b[:bus])) || error("$file: loads differ")
    for (x, y) in zip(a[:thermal], b[:thermal])
        x.name == y.name || error("$file: unit order differs")
        close(x.min_power_cost, y.min_power_cost) || error("$file: $(x.name) costs differ")
        all(close(s.cost, r.cost) for (s, r) in zip(x.cost_segments, y.cost_segments)) ||
            error("$file: $(x.name) segment costs differ")
        all(close(s.cost, r.cost) for (s, r) in zip(x.startup_categories, y.startup_categories)) ||
            error("$file: $(x.name) startup costs differ")
    end
end

function main(variants)
    bases = filter(n -> occursin(r"^pglib-uc/(ca|ferc)/", n), readlines("instances.txt"))
    names = String[]
    for (i, name) in enumerate(bases), s in 1:variants
        base = joinpath("instances", "$name.json.gz")
        json = UC._parse_json_file(base)
        peak = maximum(sum(b.load for b in UC.read(base).scenarios[1][:bus]))
        C = sum(capacity, values(json["Generators"]))
        method = R.Randomization(peak_load = D.Uniform(0.925 * peak / C, 1.075 * peak / C))
        seed = 100i + s
        randomize_json!(json, MersenneTwister(seed), method)
        dataset, day = split(name, "/")[2:3]
        test = "pglib-uc/$dataset-rand/$(day)_s$s"
        file = joinpath("instances", "$test.json.gz")
        mkpath(dirname(file))
        open(file, "w") do raw
            io = Z.GzipCompressorStream(raw)
            JSON.print(io, json)
            close(io)
        end
        check(file, base, seed, method)
        push!(names, test)
        println(test)
    end
    others = isfile("instances_test.txt") ?
        filter(n -> !isempty(n) && !occursin("-rand/", n), readlines("instances_test.txt")) : String[]
    write("instances_test.txt", join([others; names], "\n") * "\n")
    return
end

main(isempty(ARGS) ? 2 : parse(Int, ARGS[1]))
