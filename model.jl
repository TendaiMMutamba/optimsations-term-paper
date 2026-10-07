using JuMP, HiGHS

struct MIPResult
    objective::Float64
    bound::Float64
    gap::Float64
    time::Float64
    x::Matrix{Float64}
    v::Vector{Float64}
    status::String
end

function solve_direct_mip(inst::Instance; time_limit::Float64=600.0, verbose::Bool=false)
    N, M = inst.N, inst.M

    model = Model(HiGHS.Optimizer)
    if !verbose
        set_silent(model)
    end
    set_time_limit_sec(model, time_limit)

    @variable(model, x[1:N, 1:M], Bin)
    @variable(model, v[1:M], Bin)

    z_cost = sum(inst.f[j] * v[j] for j in 1:M) +
             sum(inst.c[i, j] * x[i, j] for i in 1:N, j in 1:M)
    z_time = sum(inst.t[i, j] * x[i, j] for i in 1:N, j in 1:M)
    z_energy = sum(inst.P_idle[j] * v[j] +
                   (inst.P_peak[j] - inst.P_idle[j]) / inst.C_cpu[j] *
                   sum(inst.d_cpu[i] * x[i, j] for i in 1:N)
                   for j in 1:M)

    @objective(model, Min, inst.α * z_cost + inst.β * z_time + inst.γ * z_energy)

    @constraint(model, assign[i=1:N], sum(x[i, j] for j in 1:M) == 1)
    @constraint(model, cap_cpu[j=1:M], sum(inst.d_cpu[i] * x[i, j] for i in 1:N) <= inst.C_cpu[j] * v[j])
    @constraint(model, cap_mem[j=1:M], sum(inst.d_mem[i] * x[i, j] for i in 1:N) <= inst.C_mem[j] * v[j])
    @constraint(model, cap_sto[j=1:M], sum(inst.d_sto[i] * x[i, j] for i in 1:N) <= inst.C_sto[j] * v[j])
    @constraint(model, link[i=1:N, j=1:M], x[i, j] <= v[j])
    @constraint(model, budget,
        sum(inst.f[j] * v[j] for j in 1:M) +
        sum(inst.c[i, j] * x[i, j] for i in 1:N, j in 1:M) <= inst.B)

    t_start = time()
    optimize!(model)
    elapsed = time() - t_start

    status = string(termination_status(model))
    obj = has_values(model) ? objective_value(model) : Inf
    bnd = has_values(model) ? objective_bound(model) : -Inf
    gap_val = obj > 0 ? (obj - bnd) / obj * 100 : 0.0

    x_val = has_values(model) ? value.(x) : zeros(N, M)
    v_val = has_values(model) ? value.(v) : zeros(M)

    return MIPResult(obj, bnd, gap_val, elapsed, x_val, v_val, status)
end

function solve_lp_relaxation(inst::Instance; verbose::Bool=false)
    N, M = inst.N, inst.M

    model = Model(HiGHS.Optimizer)
    if !verbose
        set_silent(model)
    end

    @variable(model, 0 <= x[1:N, 1:M] <= 1)
    @variable(model, 0 <= v[1:M] <= 1)

    z_cost = sum(inst.f[j] * v[j] for j in 1:M) +
             sum(inst.c[i, j] * x[i, j] for i in 1:N, j in 1:M)
    z_time = sum(inst.t[i, j] * x[i, j] for i in 1:N, j in 1:M)
    z_energy = sum(inst.P_idle[j] * v[j] +
                   (inst.P_peak[j] - inst.P_idle[j]) / inst.C_cpu[j] *
                   sum(inst.d_cpu[i] * x[i, j] for i in 1:N)
                   for j in 1:M)

    @objective(model, Min, inst.α * z_cost + inst.β * z_time + inst.γ * z_energy)

    @constraint(model, assign[i=1:N], sum(x[i, j] for j in 1:M) == 1)
    @constraint(model, cap_cpu[j=1:M], sum(inst.d_cpu[i] * x[i, j] for i in 1:N) <= inst.C_cpu[j] * v[j])
    @constraint(model, cap_mem[j=1:M], sum(inst.d_mem[i] * x[i, j] for i in 1:N) <= inst.C_mem[j] * v[j])
    @constraint(model, cap_sto[j=1:M], sum(inst.d_sto[i] * x[i, j] for i in 1:N) <= inst.C_sto[j] * v[j])
    @constraint(model, link[i=1:N, j=1:M], x[i, j] <= v[j])
    @constraint(model, budget,
        sum(inst.f[j] * v[j] for j in 1:M) +
        sum(inst.c[i, j] * x[i, j] for i in 1:N, j in 1:M) <= inst.B)

    optimize!(model)

    return objective_value(model)
end
