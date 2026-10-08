@enum StepSizePolicy POLYAK GEOMETRIC HARMONIC

struct LagrangianResult
    best_ub::Float64
    best_lb::Float64
    gap::Float64
    time::Float64
    x::Matrix{Float64}
    v::Vector{Float64}
    λ::Vector{Float64}
    lb_history::Vector{Float64}
    ub_history::Vector{Float64}
    iterations::Int
end

function initialize_multipliers(inst::Instance)
    λ = zeros(inst.N)
    for i in 1:inst.N
        λ[i] = minimum(combined_cost(inst, i, j) for j in 1:inst.M)
    end
    return λ
end

function lagrangian_relaxation(inst::Instance;
                               policy::StepSizePolicy=POLYAK,
                               max_iter::Int=500,
                               tol::Float64=0.01,
                               α_init::Float64=2.0,
                               stall_limit::Int=30,
                               t0_geo::Float64=5.0,
                               ρ_geo::Float64=0.995,
                               a_harm::Float64=10.0,
                               b_harm::Float64=5.0,
                               verbose::Bool=false)
    N = inst.N
    λ = initialize_multipliers(inst)
    best_lb = -Inf
    best_ub = Inf
    best_x = zeros(inst.N, inst.M)
    best_v = zeros(inst.M)
    α_k = α_init
    stall_count = 0
    prev_best_lb = -Inf

    lb_history = Float64[]
    ub_history = Float64[]

    t_start = time()

    for k in 1:max_iter
        x_lr, v_lr, lb_k = solve_all_subproblems(inst, λ)

        if lb_k > best_lb
            best_lb = lb_k
        end

        if best_lb <= prev_best_lb + abs(prev_best_lb) * 1e-4
            stall_count += 1
        else
            stall_count = 0
        end
        prev_best_lb = best_lb

        if stall_count >= stall_limit && policy == POLYAK
            α_k = max(α_k * 0.5, 0.005)
            stall_count = 0
        end

        feas = recover_feasibility(inst, x_lr, v_lr, λ)
        if feas.feasible && feas.objective < best_ub
            best_ub = feas.objective
            best_x = copy(feas.x)
            best_v = copy(feas.v)
        end

        push!(lb_history, best_lb)
        push!(ub_history, best_ub < Inf ? best_ub : lb_k)

        if best_ub < Inf && best_lb > -Inf
            gap = (best_ub - best_lb) / best_ub
            if gap < tol
                if verbose
                    println("Converged at iteration $k, gap = $(round(gap*100, digits=2))%")
                end
                elapsed = time() - t_start
                return LagrangianResult(best_ub, best_lb,
                    gap * 100, elapsed, best_x, best_v, λ,
                    lb_history, ub_history, k)
            end
        end

        g = zeros(N)
        for i in 1:N
            g[i] = 1.0 - sum(x_lr[i, j] for j in 1:inst.M)
        end

        g_norm_sq = sum(g .^ 2)
        if g_norm_sq < 1e-12
            if verbose
                println("Zero subgradient at iteration $k")
            end
            elapsed = time() - t_start
            gap_val = best_ub < Inf ? (best_ub - best_lb) / best_ub * 100 : 100.0
            return LagrangianResult(best_ub, best_lb,
                gap_val, elapsed, best_x, best_v, λ,
                lb_history, ub_history, k)
        end

        if policy == POLYAK
            target = best_ub < Inf ? best_ub : lb_k * 1.05
            t_k = α_k * (target - lb_k) / g_norm_sq
        elseif policy == GEOMETRIC
            t_k = t0_geo * ρ_geo^k
        else
            t_k = a_harm / (b_harm + k)
        end

        t_k = max(t_k, 1e-8)

        for i in 1:N
            λ[i] = λ[i] + t_k * g[i]
        end

        if verbose && k % 50 == 0
            gap_str = best_ub < Inf ? "$(round((best_ub - best_lb)/best_ub*100, digits=2))%" : "N/A"
            println("Iter $k: LB=$(round(best_lb, digits=1)), UB=$(round(best_ub, digits=1)), gap=$gap_str")
        end
    end

    elapsed = time() - t_start
    gap_val = best_ub < Inf && best_ub > 0 ? (best_ub - best_lb) / best_ub * 100 : 100.0
    return LagrangianResult(best_ub, best_lb,
        gap_val, elapsed, best_x, best_v, λ,
        lb_history, ub_history, max_iter)
end
