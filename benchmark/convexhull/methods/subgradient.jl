# Subgradient methods. All take projected steps along the supergradient g_k of
# L at π_k and differ in the step size.

# Normalized, diminishing steps (the method of UnitCommitment.ConvexHullLMP):
#
#     π_{k+1} = max(π_k + step / k · g_k / ‖g_k‖, lower)
function subgradient(oracle, π0, stop::Stop; step)
    trace = Trace(π0)
    π = π0
    for k in Iterators.countfrom(1)
        r = evaluate!(trace, oracle, π, stop)
        r.stop && break
        norm_g = norm(r.grad)
        if norm_g == 0
            trace.status = "optimal"
            break
        end
        π = project(oracle, π .+ step / k .* r.grad ./ norm_g)
    end
    return trace
end

# Last-iterate optimal steps for a fixed number of iterations N:
#
#     π_{k+1} = max(π_k + R (N + 1 - k) / (N + 1)^{3/2} · g_k / ‖g_k‖, lower)
function subgradient_last(oracle, π0, stop::Stop; N, R)
    trace = Trace(π0)
    π = π0
    for k in 1:N
        r = evaluate!(trace, oracle, π, stop)
        r.stop && break
        norm_g = norm(r.grad)
        if norm_g == 0
            trace.status = "optimal"
            break
        end
        step = R * (N + 1 - k) / (N + 1)^1.5
        π = project(oracle, π .+ step .* r.grad ./ norm_g)
        k == N && (trace.status = "iterations")
    end
    return trace
end

# Polyak steps with the estimated optimal value L_best + α / k:
#
#     π_{k+1} = max(π_k + 2 (L_best + α / k - L(π_k)) / ‖g_k‖² · g_k, lower)
function polyak_estimated(oracle, π0, stop::Stop; α)
    trace = Trace(π0)
    π = π0
    for k in Iterators.countfrom(1)
        r = evaluate!(trace, oracle, π, stop)
        r.stop && break
        norm_g2 = sum(abs2, r.grad)
        if norm_g2 == 0
            trace.status = "optimal"
            break
        end
        step = (maximum(trace.value) + α / k - r.value) / norm_g2
        π = project(oracle, π .+ 2 * step .* r.grad)
    end
    return trace
end

# D-Adaptation (Defazio and Mishchenko, 2023), from the distance estimate d0:
#
#     s_k = s_{k-1} + d_k g_k,   γ_{k+1} = 1 / sqrt(Σ_{j≤k} ‖g_j‖²),
#     d_{k+1} = max(d_k, (γ_{k+1} ‖s_k‖² - Σ_{j≤k} γ_j d_j² ‖g_j‖²) / (2 ‖s_k‖)),
#     π_{k+1} = max(π_1 + γ_{k+1} s_k, lower),
#
# with γ_1 = 1 / ‖g_1‖.
function d_adaptation(oracle, π0, stop::Stop; d0)
    trace = Trace(π0)
    π = π0
    s = zeros(length(π0))
    d, γ, sum_g2, sum_γdg = Float64(d0), NaN, 0.0, 0.0
    for k in Iterators.countfrom(1)
        r = evaluate!(trace, oracle, π, stop)
        r.stop && break
        norm_g2 = sum(abs2, r.grad)
        if norm_g2 == 0
            trace.status = "optimal"
            break
        end
        k == 1 && (γ = 1 / sqrt(norm_g2))
        s .+= d .* r.grad
        sum_γdg += γ * d^2 * norm_g2
        sum_g2 += norm_g2
        γ = 1 / sqrt(sum_g2)
        norm_s = norm(s)
        d = max(d, (γ * norm_s^2 - sum_γdg) / (2 * norm_s))
        π = project(oracle, π0 .+ γ .* s)
    end
    return trace
end

# DoWG (Khaled, Mishchenko and Jin, 2023), from the distance estimate d0:
#
#     r_k = max(r_{k-1}, ‖π_k - π_1‖),   v_k = v_{k-1} + r_k² ‖g_k‖²,
#     π_{k+1} = max(π_k + r_k² / sqrt(v_k) · g_k, lower),
#
# with r_0 = d0 and v_0 = 0.
function dowg(oracle, π0, stop::Stop; d0)
    trace = Trace(π0)
    π = π0
    radius, v = Float64(d0), 0.0
    for k in Iterators.countfrom(1)
        r = evaluate!(trace, oracle, π, stop)
        r.stop && break
        norm_g2 = sum(abs2, r.grad)
        if norm_g2 == 0
            trace.status = "optimal"
            break
        end
        radius = max(radius, norm(π .- π0))
        v += radius^2 * norm_g2
        π = project(oracle, π .+ radius^2 / sqrt(v) .* r.grad)
    end
    return trace
end
