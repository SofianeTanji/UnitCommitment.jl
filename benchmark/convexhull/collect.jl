# Gathers results/<name>.json for every instance of instances.txt into
# results.csv (status "missing" when a job left no result, e.g. out of memory).
#
# Usage: julia --project=. collect.jl

using JSON

columns = [
    "status",
    "units",
    "prices",
    "lp_value",
    "dual_value_at_lp_duals",
    "lower_bound",
    "upper_bound",
    "gap",
    "iterations",
    "pricings",
    "artificials",
    "threads",
    "build_time",
    "total_time",
    "error",
]
open("results.csv", "w") do io
    println(io, join(["name"; columns], ","))
    for name in readlines("instances.txt")
        file = joinpath("results", "$name.json")
        r = isfile(file) ? JSON.parsefile(file) : Dict("status" => "missing")
        row = [name; [repr(get(r, c, "")) for c in columns]]
        println(io, join(row, ","))
    end
end
