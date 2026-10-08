struct FeasibleResult
    x::Matrix{Float64}
    v::Vector{Float64}
    objective::Float64
    feasible::Bool
end

function can_fit(inst::Instance, j::Int, i::Int, current_cpu::Vector{Float64},
                 current_mem::Vector{Float64}, current_sto::Vector{Float64})
    return (current_cpu[j] + inst.d_cpu[i] <= inst.C_cpu[j] &&
            current_mem[j] + inst.d_mem[i] <= inst.C_mem[j] &&
            current_sto[j] + inst.d_sto[i] <= inst.C_sto[j])
end

function recover_feasibility(inst::Instance, x_lr::Matrix{Float64},
                             v_lr::Vector{Float64}, λ::Vector{Float64})
    N, M = inst.N, inst.M
    x = zeros(N, M)
    v = zeros(M)

    current_cpu = zeros(M)
    current_mem = zeros(M)
    current_sto = zeros(M)

    order = sortperm(λ, rev=true)

    for i in order
        assigned = false

        lr_servers = [j for j in 1:M if x_lr[i, j] > 0.5]
        sort!(lr_servers, by=j -> combined_cost(inst, i, j))

        for j in lr_servers
            if can_fit(inst, j, i, current_cpu, current_mem, current_sto)
                x[i, j] = 1.0
                v[j] = 1.0
                current_cpu[j] += inst.d_cpu[i]
                current_mem[j] += inst.d_mem[i]
                current_sto[j] += inst.d_sto[i]
                assigned = true
                break
            end
        end

        if !assigned
            active = [j for j in 1:M if v[j] > 0.5]
            sort!(active, by=j -> combined_cost(inst, i, j))
            for j in active
                if can_fit(inst, j, i, current_cpu, current_mem, current_sto)
                    x[i, j] = 1.0
                    current_cpu[j] += inst.d_cpu[i]
                    current_mem[j] += inst.d_mem[i]
                    current_sto[j] += inst.d_sto[i]
                    assigned = true
                    break
                end
            end
        end

        if !assigned
            inactive = [j for j in 1:M if v[j] < 0.5]
            sort!(inactive, by=j -> activation_cost(inst, j) + combined_cost(inst, i, j))
            for j in inactive
                if can_fit(inst, j, i, current_cpu, current_mem, current_sto)
                    x[i, j] = 1.0
                    v[j] = 1.0
                    current_cpu[j] += inst.d_cpu[i]
                    current_mem[j] += inst.d_mem[i]
                    current_sto[j] += inst.d_sto[i]
                    assigned = true
                    break
                end
            end
        end

        if !assigned
            return FeasibleResult(x, v, Inf, false)
        end
    end

    for j in 1:M
        if sum(x[:, j]) < 0.5
            v[j] = 0.0
        end
    end

    obj = evaluate_objective(inst, x, v)
    return FeasibleResult(x, v, obj, true)
end
