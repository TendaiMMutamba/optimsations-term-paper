using JuMP, HiGHS

struct SubproblemResult
    x::Vector{Float64}
    v::Float64
    objective::Float64
end

mutable struct BBState
    best_val::Float64
    best_sol::Vector{Bool}
    node_count::Int
    node_limit::Int
end

function knapsack_bb!(state::BBState, profits::Vector{Float64},
                      weights::Matrix{Float64}, capacities::Vector{Float64},
                      idx::Int, n::Int, ndim::Int,
                      current_val::Float64, current_weight::Vector{Float64},
                      current_sol::Vector{Bool})
    state.node_count += 1
    if state.node_count > state.node_limit
        return
    end

    if current_val < state.best_val
        state.best_val = current_val
        state.best_sol .= current_sol
    end

    if idx > n
        return
    end

    ub = current_val
    for k in idx:n
        if profits[k] >= 0
            continue
        end
        feasible = true
        for d in 1:ndim
            if current_weight[d] + weights[k, d] > capacities[d]
                feasible = false
                break
            end
        end
        if feasible
            ub += profits[k]
        end
    end
    if ub >= state.best_val
        return
    end

    if profits[idx] < 0
        feasible = true
        for d in 1:ndim
            if current_weight[d] + weights[idx, d] > capacities[d]
                feasible = false
                break
            end
        end
        if feasible
            current_sol[idx] = true
            for d in 1:ndim
                current_weight[d] += weights[idx, d]
            end
            knapsack_bb!(state, profits, weights, capacities,
                         idx + 1, n, ndim,
                         current_val + profits[idx], current_weight, current_sol)
            current_sol[idx] = false
            for d in 1:ndim
                current_weight[d] -= weights[idx, d]
            end
        end
    end

    knapsack_bb!(state, profits, weights, capacities,
                 idx + 1, n, ndim,
                 current_val, current_weight, current_sol)
end

function solve_knapsack_exact(profits::Vector{Float64},
                              weights::Matrix{Float64},
                              capacities::Vector{Float64})
    n = length(profits)
    ndim = length(capacities)

    order = sortperm(profits)
    sorted_profits = profits[order]
    sorted_weights = weights[order, :]

    state = BBState(0.0, zeros(Bool, n), 0, 50_000)
    current_sol = zeros(Bool, n)
    current_weight = zeros(ndim)

    knapsack_bb!(state, sorted_profits, sorted_weights, capacities,
                 1, n, ndim, 0.0, current_weight, current_sol)

    if state.node_count >= state.node_limit
        return solve_knapsack_jump(profits, weights, capacities)
    end

    final_sol = zeros(Bool, n)
    for k in 1:n
        final_sol[order[k]] = state.best_sol[k]
    end

    return state.best_val, final_sol
end

function solve_knapsack_jump(profits::Vector{Float64},
                             weights::Matrix{Float64},
                             capacities::Vector{Float64})
    n = length(profits)
    ndim = length(capacities)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, y[1:n], Bin)
    @objective(model, Min, sum(profits[k] * y[k] for k in 1:n))
    for d in 1:ndim
        @constraint(model, sum(weights[k, d] * y[k] for k in 1:n) <= capacities[d])
    end
    optimize!(model)

    sol = zeros(Bool, n)
    val = 0.0
    if termination_status(model) == OPTIMAL
        val = objective_value(model)
        for k in 1:n
            sol[k] = value(y[k]) > 0.5
        end
    end
    return val, sol
end

function solve_server_subproblem(inst::Instance, j::Int, λ::Vector{Float64})
    N = inst.N
    F_j = activation_cost(inst, j)

    profits = Vector{Float64}(undef, N)
    for i in 1:N
        profits[i] = combined_cost(inst, i, j) - λ[i]
    end

    profitable_idx = findall(p -> p < 0, profits)

    if isempty(profitable_idx)
        return SubproblemResult(zeros(N), 0.0, 0.0)
    end

    n_prof = length(profitable_idx)
    sub_profits = profits[profitable_idx]
    sub_weights = zeros(n_prof, 3)
    for (k, i) in enumerate(profitable_idx)
        sub_weights[k, 1] = inst.d_cpu[i]
        sub_weights[k, 2] = inst.d_mem[i]
        sub_weights[k, 3] = inst.d_sto[i]
    end
    caps = [inst.C_cpu[j], inst.C_mem[j], inst.C_sto[j]]

    knapsack_val, sol = solve_knapsack_exact(sub_profits, sub_weights, caps)

    if F_j + knapsack_val < 0
        x_j = zeros(N)
        for (k, i) in enumerate(profitable_idx)
            if sol[k]
                x_j[i] = 1.0
            end
        end
        return SubproblemResult(x_j, 1.0, F_j + knapsack_val)
    else
        return SubproblemResult(zeros(N), 0.0, 0.0)
    end
end

function solve_all_subproblems(inst::Instance, λ::Vector{Float64})
    N, M = inst.N, inst.M
    x = zeros(N, M)
    v = zeros(M)
    total_obj = sum(λ)

    for j in 1:M
        result = solve_server_subproblem(inst, j, λ)
        x[:, j] = result.x
        v[j] = result.v
        total_obj += result.objective
    end

    return x, v, total_obj
end
