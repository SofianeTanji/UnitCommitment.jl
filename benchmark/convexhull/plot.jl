# Figures of the evaluation, from the tables of analyze.jl, with one hue per
# family of methods (bundle, subgradient, parameter-free, primal):
#
#   curves.pdf    mean gap over time on each dataset, for the best variant of
#                 each bundle method (by geometric mean time over all
#                 instances), the subgradient and parameter-free methods,
#                 column generation and column-and-row generation
#   solved.pdf    share of the instances of each dataset on which the gap
#                 reaches the target 1e-6, over time, for the same methods
#   ranking.pdf   capped geometric mean of the time to the target of every
#                 configuration, on each dataset and over all instances
#   switches.pdf  speedup of each switch of the bundle methods, over the 5
#                 bundle methods, on each dataset and over all instances
#
# Usage: julia --project=plots plot.jl [<analysis dir>], by default analysis;
# writes <analysis dir>/figures.

using Plots, DelimitedFiles

const TARGET = 1e-6
const DATASETS = ["or-lib", "pglib-uc/ca", "pglib-uc/ferc"]
const DATASET_COLORS = Dict("or-lib" => "#4F7AA0", "pglib-uc/ca" => "#D08B54", "pglib-uc/ferc" => "#5A8F74", "all" => "#222222")

const STYLES = Dict(
    "bpm" => (color = "#8F5D34", linestyle = :solid),
    "moreau-pp" => (color = "#8F5D34", linestyle = :dash),
    "bplm" => (color = "#C97A3D", linestyle = :solid),
    "blm" => (color = "#E4A874", linestyle = :dash),
    "subgradient" => (color = "#9AB4CB", linestyle = :dot),
    "subgradient_last" => (color = "#315D86", linestyle = :dot),
    "polyak_estimated" => (color = "#5E82A6", linestyle = :solid),
    "d_adaptation" => (color = "#6F5A8D", linestyle = :solid),
    "dowg" => (color = "#A88FB9", linestyle = :dash),
    "cg-stabilized" => (color = "#9B4F5D", linestyle = :solid),
    "cg-multicut" => (color = "#9B4F5D", linestyle = :dash),
    "cg-single" => (color = "#C98A95", linestyle = :dot),
    "crg" => (color = "#6B2E3A", linestyle = :dashdot),
)

# Rows of a CSV file of analyze.jl, as dictionaries of strings
function read_csv(file)
    data, header = readdlm(file, ',', String; header = true)
    return [Dict(zip(vec(header), data[i, :])) for i in axes(data, 1)]
end

number(s) = isempty(s) ? nothing : parse(Float64, s)

bundle_base(label) = first(split(label, r"-(agg|k4|block)"))

# The plotted configurations: the fastest variant of each bundle method, then
# the others of STYLES
function selection(summary)
    overall = sort(filter(r -> r["dataset"] == "all", summary); by = r -> number(r["geomean_time"]))
    labels = String[]
    for base in ("bpm", "moreau-pp", "bplm", "blm")
        i = findfirst(r -> bundle_base(r["label"]) == base && r["label"] != base, overall)
        i === nothing || push!(labels, overall[i]["label"])
    end
    available = Set(r["label"] for r in overall)
    for l in ["subgradient", "subgradient_last", "polyak_estimated", "d_adaptation", "dowg",
        "cg-stabilized", "cg-multicut", "cg-single", "crg"]
        l in available && push!(labels, l)
    end
    return labels
end

style(label) = STYLES[haskey(STYLES, label) ? label : bundle_base(label)]

const FONTS = (guidefontsize = 9, tickfontsize = 8, legendfontsize = 7, titlefontsize = 10)
const TIME_TICKS = ([0.1, 1, 10, 100, 300], ["0.1", "1", "10", "100", "300"])

function curves(rows, labels, out)
    panels = map(enumerate(DATASETS)) do (k, d)
        p = plot(; xscale = :log10, yscale = :log10, title = d, xlabel = "time (s)",
            ylabel = k == 1 ? "mean gap" : "", legend = k == length(DATASETS) ? :outerright : false,
            ylims = (1e-8, 1e-2), xlims = (0.1, 300), xticks = TIME_TICKS, FONTS...)
        for l in labels
            r = filter(r -> r["dataset"] == d && r["label"] == l, rows)
            plot!(p, number.(getindex.(r, "time")), max.(number.(getindex.(r, "mean_gap")), 1e-8);
                label = l, linewidth = 2, style(l)...)
        end
        hline!(p, [TARGET]; color = :black, linestyle = :dashdot, linewidth = 1, label = "target")
        p
    end
    savefig(plot(panels...; layout = grid(1, 3, widths = [0.27, 0.27, 0.46]), size = (1300, 380),
        left_margin = 4Plots.mm, bottom_margin = 5Plots.mm), joinpath(out, "curves.pdf"))
end

function solved(runs, times, labels, out)
    panels = map(enumerate(DATASETS)) do (k, d)
        p = plot(; xscale = :log10, title = d, xlabel = "time (s)", ylabel = k == 1 ? "instances solved (%)" : "",
            legend = k == length(DATASETS) ? :outerright : false, ylims = (0, 100), xlims = (0.1, 300),
            xticks = TIME_TICKS, FONTS...)
        for l in labels
            t = [number(r["time_to_target"]) for r in runs if r["dataset"] == d && r["label"] == l]
            share = [100 * count(x -> x !== nothing && x <= s, t) / length(t) for s in times]
            plot!(p, times, share; label = l, linewidth = 2, seriestype = :steppost, style(l)...)
        end
        p
    end
    savefig(plot(panels...; layout = grid(1, 3, widths = [0.27, 0.27, 0.46]), size = (1300, 380),
        left_margin = 7Plots.mm, bottom_margin = 5Plots.mm), joinpath(out, "solved.pdf"))
end

function ranking(summary, out)
    overall = sort(filter(r -> r["dataset"] == "all", summary); by = r -> -number(r["geomean_time"]))
    labels = getindex.(overall, "label")
    p = plot(; xscale = :log10, xlabel = "geometric mean of the time to the target (s), 300 s when not reached",
        xticks = ([10, 20, 50, 100, 300], ["10", "20", "50", "100", "300"]),
        yticks = (eachindex(labels), labels), ylims = (0.5, length(labels) + 0.5), legend = :outertop,
        legend_columns = 4, size = (800, 14 * length(labels) + 140), FONTS..., ytickfontsize = 7,
        left_margin = 6Plots.mm, bottom_margin = 4Plots.mm)
    for d in [DATASETS; "all"]
        x = [number(only(filter(r -> r["dataset"] == d && r["label"] == l, summary))["geomean_time"]) for l in labels]
        scatter!(p, x, eachindex(labels); label = d, color = DATASET_COLORS[d], markersize = d == "all" ? 5 : 4,
            markershape = d == "all" ? :diamond : :circle, markerstrokewidth = 0)
    end
    savefig(p, joinpath(out, "ranking.pdf"))
end

function switches(rows, out)
    rows = filter(r -> r["base"] == "all", rows)
    names = unique(getindex.(rows, "switch"))
    p = plot(; xscale = :log10, xlabel = "speedup (geometric mean over the pairs of configurations)",
        xticks = ([0.8, 1, 1.5, 2, 3, 5, 8], ["0.8", "1", "1.5", "2", "3", "5", "8"]), xlims = (0.8, 8),
        yticks = (eachindex(names), names), ylims = (0.5, length(names) + 0.5), legend = :topright,
        size = (700, 300), bottom_margin = 5Plots.mm, FONTS...)
    vline!(p, [1.0]; color = :black, alpha = 0.3, label = "")
    for d in [DATASETS; "all"]
        x = [number(only(filter(r -> r["dataset"] == d && r["switch"] == s, rows))["speedup"]) for s in names]
        scatter!(p, x, eachindex(names); label = d, color = DATASET_COLORS[d], markersize = d == "all" ? 6 : 5,
            markershape = d == "all" ? :diamond : :circle, markerstrokewidth = 0)
    end
    savefig(p, joinpath(out, "switches.pdf"))
end

function main(dir)
    out = joinpath(dir, "figures")
    mkpath(out)
    summary = read_csv(joinpath(dir, "summary.csv"))
    labels = selection(summary)
    rows = read_csv(joinpath(dir, "curves.csv"))
    times = sort(unique(number.(getindex.(rows, "time"))))
    curves(rows, labels, out)
    solved(read_csv(joinpath(dir, "runs.csv")), times, labels, out)
    ranking(summary, out)
    switches(read_csv(joinpath(dir, "switches.csv")), out)
    println("$out: curves.pdf, solved.pdf, ranking.pdf, switches.pdf")
end

main(get(ARGS, 1, "analysis"))
