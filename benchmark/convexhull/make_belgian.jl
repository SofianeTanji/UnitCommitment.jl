# Held-out test instances: the 8 Belgian instances of ConvexHullPricing.jl
# (data/belgian/belgian-<day>.json, one bus, 68 units, 96 periods), converted
# to the UnitCommitment.jl format as instances/belgian/<day>.json.gz.
#
# The data give, per unit, the minimum and maximum power, one marginal cost, a
# no-load cost, one startup cost, the ramp limits and the minimum up and down
# times. The cost curve is (Pmin, no-load + c·Pmin), (Pmax, no-load + c·Pmax);
# every other parameter (power balance penalty, startup and shutdown limits)
# takes the UnitCommitment.jl default. The initial conditions are computed by
# UnitCommitment.generate_initial_conditions!. The names are appended to
# instances_test.txt (run after make_test.jl, which rewrites it).
#
# Usage: julia --project=. make_belgian.jl <data/belgian directory>

using UnitCommitment, Gurobi, JuMP, JSON
const UC = UnitCommitment
const Z = UC.CodecZlib

function convert(source)
    data = JSON.parsefile(source)
    T = data["time_periods"]
    generators = Dict{String,Any}()
    for (name, g) in data["thermal_generators"]
        length(g["piecewise_production"]) == 1 && length(g["startup"]) == 1 ||
            error("$source: $name has several cost segments or startup categories")
        pmin, pmax = g["power_output_minimum"], g["power_output_maximum"]
        c, noload = g["piecewise_production"][1]["cost"], g["no_load_consumption"]
        generators[name] = Dict(
            "Bus" => "b1",
            "Production cost curve (MW)" => [pmin, pmax],
            "Production cost curve (\$)" => [noload + c * pmin, noload + c * pmax],
            "Startup costs (\$)" => [g["startup"][1]["cost"]],
            "Startup delays (h)" => [1],
            "Ramp up limit (MW)" => g["ramp_up_limit"],
            "Ramp down limit (MW)" => g["ramp_down_limit"],
            "Minimum uptime (h)" => g["time_up_minimum"],
            "Minimum downtime (h)" => g["time_down_minimum"],
        )
    end
    return Dict(
        "Parameters" => Dict("Version" => "0.3", "Time (h)" => T),
        "Buses" => Dict("b1" => Dict("Load (MW)" => Float64.(data["demand"]))),
        "Generators" => generators,
    )
end

function write_gz(file, json)
    mkpath(dirname(file))
    open(file, "w") do raw
        io = Z.GzipCompressorStream(raw)
        JSON.print(io, json)
        close(io)
    end
end

function main(dir)
    optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)
    names = String[]
    for source in sort(filter(f -> endswith(f, ".json"), readdir(dir; join = true)))
        day = replace(basename(source), r"^belgian-|\.json$" => "")
        name = "belgian/$day"
        file = joinpath("instances", "$name.json.gz")
        json = convert(source)
        write_gz(file, json)
        instance = UC.read(file)
        UC.generate_initial_conditions!(instance, optimizer)
        for g in instance.scenarios[1][:thermal]
            json["Generators"][g.name]["Initial status (h)"] = g.initial_status
            json["Generators"][g.name]["Initial power (MW)"] = g.initial_power
        end
        write_gz(file, json)
        check = UC.read(file).scenarios[1]
        all(g.initial_status !== nothing && g.initial_power !== nothing for g in check[:thermal]) ||
            error("$file: missing initial conditions")
        push!(names, name)
        println(name)
    end
    previous = isfile("instances_test.txt") ? readlines("instances_test.txt") : String[]
    write("instances_test.txt", join(unique([previous; names]), "\n") * "\n")
    return
end

main(ARGS[1])
