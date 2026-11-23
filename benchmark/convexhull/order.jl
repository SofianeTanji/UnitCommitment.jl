# Sorts a list of instances for submit.sh by decreasing estimated time, the
# longest first (which shortens the makespan when the number of running jobs
# is capped): the total time of an earlier dw.jl result of the instance, or of
# its base instance for a randomized variant (pglib-uc/<dataset>-rand/<day>_s<k>
# for pglib-uc/<dataset>/<day>); the instances without one come first.
#
# Usage: julia --project=. order.jl <list> <results dir>, e.g.
#
#   julia --project=. order.jl instances_dev.txt results_highs
#
# rewrites the list in place.

using JSON

function estimate(name, dir)
    base = replace(name, r"^pglib-uc/(\w+)-rand/(.*)_s\d+$" => s"pglib-uc/\1/\2")
    for n in unique([name, base])
        file = joinpath(dir, "$n.json")
        isfile(file) && return Float64(get(JSON.parsefile(file), "total_time", Inf))
    end
    return Inf
end

list, dir = ARGS
names = filter(!isempty, readlines(list))
t = [estimate(n, dir) for n in names]
order = sortperm(t; rev = true)
write(list, join(names[order], "\n") * "\n")
println("$list: $(length(names)) instances, $(count(isinf, t)) without an estimate, ",
    "estimated $(round(sum(filter(isfinite, t)) / 3600; digits = 1)) job-hours")
