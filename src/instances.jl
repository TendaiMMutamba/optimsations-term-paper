using Random

struct Instance
    N::Int
    M::Int
    d_cpu::Vector{Float64}
    d_mem::Vector{Float64}
    d_sto::Vector{Float64}
    C_cpu::Vector{Float64}
    C_mem::Vector{Float64}
    C_sto::Vector{Float64}
    f::Vector{Float64}
    c::Matrix{Float64}
    t::Matrix{Float64}
    P_idle::Vector{Float64}
    P_peak::Vector{Float64}
    α::Float64
    β::Float64
    γ::Float64
    B::Float64
end

function generate_instance(N::Int, M::Int;
                           seed::Int=42,
                           α::Float64=1.0, β::Float64=0.5, γ::Float64=0.3,
                           size_class::Symbol=:medium)
    rng = MersenneTwister(seed)

    if size_class == :small
        cpu_range = (1, 8)
        mem_range = (1, 16)
        sto_range = (10, 100)
        cap_cpu_range = (16, 64)
        cap_mem_range = (32, 128)
        cap_sto_range = (200, 1000)
        f_range = (50, 200)
        idle_range = (50, 150)
        peak_range = (200, 500)
    elseif size_class == :large
        cpu_range = (1, 16)
        mem_range = (2, 64)
        sto_range = (20, 500)
        cap_cpu_range = (32, 128)
        cap_mem_range = (64, 512)
        cap_sto_range = (500, 4000)
        f_range = (100, 500)
        idle_range = (100, 300)
        peak_range = (300, 800)
    else
        cpu_range = (1, 16)
        mem_range = (2, 64)
        sto_range = (20, 500)
        cap_cpu_range = (32, 128)
        cap_mem_range = (64, 512)
        cap_sto_range = (500, 4000)
        f_range = (100, 500)
        idle_range = (100, 300)
        peak_range = (300, 800)
    end

    urand(lo, hi) = lo + rand(rng) * (hi - lo)

    d_cpu = [urand(cpu_range...) for _ in 1:N]
    d_mem = [urand(mem_range...) for _ in 1:N]
    d_sto = [urand(sto_range...) for _ in 1:N]

    C_cpu = [urand(cap_cpu_range...) for _ in 1:M]
    C_mem = [urand(cap_mem_range...) for _ in 1:M]
    C_sto = [urand(cap_sto_range...) for _ in 1:M]

    f_cost = [urand(f_range...) for _ in 1:M]
    P_idle = [urand(idle_range...) for _ in 1:M]
    P_peak = [urand(peak_range...) for _ in 1:M]

    p_cpu = [urand(0.5, 3.0) for _ in 1:M]
    p_mem = [urand(0.5, 3.0) for _ in 1:M]
    p_sto = [urand(0.5, 3.0) for _ in 1:M]

    c = zeros(N, M)
    for i in 1:N, j in 1:M
        c[i, j] = p_cpu[j] * d_cpu[i] + p_mem[j] * d_mem[i] + p_sto[j] * d_sto[i]
    end

    proc_time = zeros(N, M)
    for i in 1:N, j in 1:M
        proc_time[i, j] = urand(0.1, 2.0)
    end

    total_demand_cpu = sum(d_cpu)
    total_cap_cpu = sum(C_cpu)
    if total_cap_cpu < 1.3 * total_demand_cpu
        scale = 1.5 * total_demand_cpu / total_cap_cpu
        C_cpu .*= scale
        C_mem .*= scale
        C_sto .*= scale
    end

    total_cost_est = sum(f_cost) + sum(c) / M
    B = 2.0 * total_cost_est

    return Instance(N, M, d_cpu, d_mem, d_sto,
                    C_cpu, C_mem, C_sto,
                    f_cost, c, proc_time,
                    P_idle, P_peak,
                    α, β, γ, B)
end

function small_instance(; seed=42, α=1.0, β=0.5, γ=0.3)
    generate_instance(10, 3; seed, α, β, γ, size_class=:small)
end

function medium_instance(; seed=42, α=1.0, β=0.5, γ=0.3)
    generate_instance(50, 10; seed, α, β, γ, size_class=:medium)
end

function large_instance(; seed=42, α=1.0, β=0.5, γ=0.3)
    generate_instance(200, 30; seed, α, β, γ, size_class=:large)
end

function combined_cost(inst::Instance, i::Int, j::Int)
    G = inst.α * inst.c[i, j] +
        inst.β * inst.t[i, j] +
        inst.γ * (inst.P_peak[j] - inst.P_idle[j]) * inst.d_cpu[i] / inst.C_cpu[j]
    return G
end

function activation_cost(inst::Instance, j::Int)
    return inst.α * inst.f[j] + inst.γ * inst.P_idle[j]
end

function evaluate_objective(inst::Instance, x::Matrix{Float64}, v::Vector{Float64})
    z_cost = sum(inst.f[j] * v[j] for j in 1:inst.M) +
             sum(inst.c[i, j] * x[i, j] for i in 1:inst.N, j in 1:inst.M)

    z_time = sum(inst.t[i, j] * x[i, j] for i in 1:inst.N, j in 1:inst.M)

    z_energy = sum(inst.P_idle[j] * v[j] +
                   (inst.P_peak[j] - inst.P_idle[j]) / inst.C_cpu[j] *
                   sum(inst.d_cpu[i] * x[i, j] for i in 1:inst.N)
                   for j in 1:inst.M)

    return inst.α * z_cost + inst.β * z_time + inst.γ * z_energy
end
