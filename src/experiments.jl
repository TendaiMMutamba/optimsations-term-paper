using Printf
using Statistics

function run_experiment1(; num_runs=2, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 1: Lagrangian vs Direct MIP")
    verbose && println("="^70)

    configs = [
        (:small, 5, 3, :small),
        (:medium, 10, 5, :medium),
        (:large, 15, 5, :large),
    ]

    results = Dict()

    for (label, N, M, sc) in configs
        verbose && println("\n--- $label instances (N=$N, M=$M) ---")

        mip_objs = Float64[]
        mip_bounds = Float64[]
        mip_times = Float64[]
        lr_objs = Float64[]
        lr_bounds = Float64[]
        lr_times = Float64[]

        for run in 1:num_runs
            inst = generate_instance(N, M; seed=run*100, size_class=sc)

            mip = solve_direct_mip(inst; time_limit=30.0)
            push!(mip_objs, mip.objective)
            push!(mip_bounds, mip.bound)
            push!(mip_times, mip.time)

            lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=50)
            push!(lr_objs, lr.best_ub)
            push!(lr_bounds, lr.best_lb)
            push!(lr_times, lr.time)

            if verbose
                mip_gap = mip.objective > 0 ? (mip.objective - mip.bound) / mip.objective * 100 : 0.0
                lr_gap = lr.best_ub > 0 ? (lr.best_ub - lr.best_lb) / lr.best_ub * 100 : 0.0
                @printf("  Run %2d: MIP obj=%.1f gap=%.1f%% t=%.1fs | LR obj=%.1f gap=%.1f%% t=%.1fs\n",
                    run, mip.objective, mip_gap, mip.time, lr.best_ub, lr_gap, lr.time)
            end
        end

        results[label] = Dict(
            :mip_obj => mean(mip_objs), :mip_obj_std => std(mip_objs),
            :mip_bound => mean(mip_bounds), :mip_time => mean(mip_times),
            :lr_obj => mean(lr_objs), :lr_obj_std => std(lr_objs),
            :lr_bound => mean(lr_bounds), :lr_time => mean(lr_times),
        )
    end

    if verbose
        println("\n--- Summary Table (averages over $num_runs runs) ---")
        println("-"^85)
        @printf("%-8s %-12s %10s %10s %8s %8s\n", "Instance", "Method", "Objective", "LB", "Gap(%)", "Time(s)")
        println("-"^85)
        for (label, _, _, _) in configs
            r = results[label]
            mip_gap = (r[:mip_obj] - r[:mip_bound]) / r[:mip_obj] * 100
            lr_gap = (r[:lr_obj] - r[:lr_bound]) / r[:lr_obj] * 100
            @printf("%-8s %-12s %10.1f %10.1f %7.1f%% %8.1f\n",
                label, "Direct MIP", r[:mip_obj], r[:mip_bound], mip_gap, r[:mip_time])
            @printf("%-8s %-12s %10.1f %10.1f %7.1f%% %8.1f\n",
                label, "Lagrangian", r[:lr_obj], r[:lr_bound], lr_gap, r[:lr_time])
        end
        println("-"^85)
    end

    return results
end

function run_experiment2(; num_runs=2, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 2: Subgradient Step Size Comparison")
    verbose && println("="^70)

    policies = [
        (POLYAK, "Polyak"),
        (GEOMETRIC, "Geometric"),
        (HARMONIC, "Harmonic"),
    ]

    results = Dict()

    for (pol, name) in policies
        lbs = Float64[]
        gaps = Float64[]
        iters_to_90 = Int[]

        for run in 1:num_runs
            inst = generate_instance(20, 5; seed=run*100, size_class=:medium)
            lr = lagrangian_relaxation(inst; policy=pol, max_iter=50)
            push!(lbs, lr.best_lb)
            push!(gaps, lr.gap)

            target_90 = 0.9 * lr.best_lb
            iter90 = findfirst(x -> x >= target_90, lr.lb_history)
            push!(iters_to_90, iter90 === nothing ? lr.iterations : iter90)
        end

        results[name] = Dict(
            :lb => mean(lbs),
            :gap => mean(gaps),
            :iters_90 => round(Int, mean(iters_to_90)),
        )

        if verbose
            @printf("  %-12s  Final LB: %8.1f  Iters to 90%%: %4d  Gap: %.1f%%\n",
                name, mean(lbs), round(Int, mean(iters_to_90)), mean(gaps))
        end
    end

    return results
end

function run_experiment3(; num_runs=2, verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 3: Duality Gap Analysis")
    verbose && println("="^70)

    configs = [
        (:small, 10, 3, :small),
        (:medium, 20, 5, :medium),
        (:large, 30, 8, :large),
    ]

    results = Dict()

    for (label, N, M, sc) in configs
        lp_bounds = Float64[]
        lr_bounds = Float64[]
        mip_objs = Float64[]

        for run in 1:num_runs
            inst = generate_instance(N, M; seed=run*100, size_class=sc)

            lp = solve_lp_relaxation(inst)
            push!(lp_bounds, lp)

            lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=50)
            push!(lr_bounds, lr.best_lb)

            mip = solve_direct_mip(inst; time_limit=30.0)
            best_feas = min(mip.objective, lr.best_ub)
            push!(mip_objs, best_feas)
        end

        results[label] = Dict(
            :lp_bound => mean(lp_bounds),
            :lr_bound => mean(lr_bounds),
            :best_feas => mean(mip_objs),
        )

        if verbose
            r = results[label]
            lr_gap = (r[:best_feas] - r[:lr_bound]) / r[:best_feas] * 100
            lp_gap = (r[:best_feas] - r[:lp_bound]) / r[:best_feas] * 100
            @printf("  %-8s  LP bound: %8.1f  LR bound: %8.1f  Best feasible: %8.1f  LR gap: %.1f%%  LP gap: %.1f%%\n",
                label, r[:lp_bound], r[:lr_bound], r[:best_feas], lr_gap, lp_gap)
        end
    end

    return results
end

function run_experiment4(; verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 4: Multiplier Analysis")
    verbose && println("="^70)

    inst = generate_instance(20, 5; seed=42, size_class=:medium)
    lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=50)

    order = sortperm(lr.λ, rev=true)

    if verbose
        println("\n  Top-10 workloads by multiplier value:")
        @printf("  %4s %10s %8s %8s %8s\n", "ID", "λ", "CPU", "Mem", "Sto")
        println("  " * "-"^42)
        for k in 1:min(10, inst.N)
            i = order[k]
            @printf("  %4d %10.2f %8.1f %8.1f %8.1f\n",
                i, lr.λ[i], inst.d_cpu[i], inst.d_mem[i], inst.d_sto[i])
        end

        println("\n  Bottom-5 workloads:")
        println("  " * "-"^42)
        for k in max(1, inst.N-4):inst.N
            i = order[k]
            @printf("  %4d %10.2f %8.1f %8.1f %8.1f\n",
                i, lr.λ[i], inst.d_cpu[i], inst.d_mem[i], inst.d_sto[i])
        end
    end

    return lr.λ, order, inst
end

function run_experiment5(; verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 5: Scalability")
    verbose && println("="^70)

    configs = [
        (5, 2), (10, 3), (20, 5), (30, 8),
    ]

    results = []

    for (N, M) in configs
        sc = N <= 10 ? :small : (N <= 50 ? :medium : :large)
        inst = generate_instance(N, M; seed=42, size_class=sc)

        tl = min(30.0, max(5.0, Float64(N)))
        mip = solve_direct_mip(inst; time_limit=tl)
        lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=50)

        mip_gap = mip.objective > 0 ? (mip.objective - mip.bound) / mip.objective * 100 : 0.0
        lr_gap = lr.gap

        push!(results, (N=N, M=M,
            mip_time=mip.time, lr_time=lr.time,
            mip_gap=mip_gap, lr_gap=lr_gap))

        if verbose
            tl_str = mip.time >= tl - 1 ? "*" : ""
            @printf("  N=%3d M=%3d  MIP: %.1fs%s gap=%.1f%%  |  LR: %.1fs gap=%.1f%%\n",
                N, M, mip.time, tl_str, mip_gap, lr.time, lr_gap)
        end
    end

    return results
end

function run_experiment6(; verbose=true)
    verbose && println("\n" * "="^70)
    verbose && println("EXPERIMENT 6: Objective Weight Sensitivity")
    verbose && println("="^70)

    weight_configs = [
        ("Cost-only",    1.0, 0.0, 0.0),
        ("Time-only",    0.0, 1.0, 0.0),
        ("Energy-only",  0.0, 0.0, 1.0),
        ("Balanced",     0.33, 0.33, 0.34),
        ("Cost+Time",    0.5, 0.5, 0.0),
        ("Cost+Energy",  0.5, 0.0, 0.5),
    ]

    results = []

    if verbose
        @printf("  %-14s %5s %5s %5s %7s %10s %10s %10s\n",
            "Config", "α", "β", "γ", "Servers", "Cost", "Time", "Energy")
        println("  " * "-"^75)
    end

    for (name, α, β, γ) in weight_configs
        inst = generate_instance(20, 5; seed=42, α=α, β=β, γ=γ, size_class=:medium)
        lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=50)

        active_servers = sum(lr.v[j] > 0.5 for j in 1:inst.M)

        z_cost = sum(inst.f[j] * lr.v[j] for j in 1:inst.M) +
                 sum(inst.c[i, j] * lr.x[i, j] for i in 1:inst.N, j in 1:inst.M)
        z_time = sum(inst.t[i, j] * lr.x[i, j] for i in 1:inst.N, j in 1:inst.M)
        z_energy = sum(
            inst.P_idle[j] * lr.v[j] +
            (inst.P_peak[j] - inst.P_idle[j]) / inst.C_cpu[j] *
            sum(inst.d_cpu[i] * lr.x[i, j] for i in 1:inst.N)
            for j in 1:inst.M)

        push!(results, (name=name, α=α, β=β, γ=γ,
            servers=active_servers, cost=z_cost, time=z_time, energy=z_energy))

        if verbose
            @printf("  %-14s %5.2f %5.2f %5.2f %7d %10.1f %10.1f %10.1f\n",
                name, α, β, γ, active_servers, z_cost, z_time, z_energy)
        end
    end

    return results
end

function save_results(results, filename::String)
    open(filename, "w") do io
        println(io, "="^70)
        println(io, "  CLOUD RESOURCE ALLOCATION - LAGRANGIAN RELAXATION")
        println(io, "  Experimental Results")
        println(io, "  Generated: ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "="^70)

        # Experiment 1
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 1: Lagrangian vs Direct MIP")
        println(io, "="^70)
        println(io, "-"^85)
        @printf(io, "%-8s %-12s %10s %10s %8s %8s\n", "Instance", "Method", "Objective", "LB", "Gap(%)", "Time(s)")
        println(io, "-"^85)
        configs_labels = [:small, :medium, :large]
        for label in configs_labels
            if haskey(results.exp1, label)
                r = results.exp1[label]
                mip_gap = r[:mip_obj] > 0 ? (r[:mip_obj] - r[:mip_bound]) / r[:mip_obj] * 100 : 0.0
                lr_gap = r[:lr_obj] > 0 ? (r[:lr_obj] - r[:lr_bound]) / r[:lr_obj] * 100 : 0.0
                @printf(io, "%-8s %-12s %10.1f %10.1f %7.1f%% %8.1f\n",
                    label, "Direct MIP", r[:mip_obj], r[:mip_bound], mip_gap, r[:mip_time])
                @printf(io, "%-8s %-12s %10.1f %10.1f %7.1f%% %8.1f\n",
                    label, "Lagrangian", r[:lr_obj], r[:lr_bound], lr_gap, r[:lr_time])
            end
        end
        println(io, "-"^85)

        # Experiment 2
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 2: Subgradient Step Size Comparison")
        println(io, "="^70)
        for name in ["Polyak", "Geometric", "Harmonic"]
            if haskey(results.exp2, name)
                r = results.exp2[name]
                @printf(io, "  %-12s  Final LB: %8.1f  Iters to 90%%: %4d  Gap: %.1f%%\n",
                    name, r[:lb], r[:iters_90], r[:gap])
            end
        end

        # Experiment 3
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 3: Duality Gap Analysis")
        println(io, "="^70)
        for label in configs_labels
            if haskey(results.exp3, label)
                r = results.exp3[label]
                lr_gap = r[:best_feas] > 0 ? (r[:best_feas] - r[:lr_bound]) / r[:best_feas] * 100 : 0.0
                lp_gap = r[:best_feas] > 0 ? (r[:best_feas] - r[:lp_bound]) / r[:best_feas] * 100 : 0.0
                @printf(io, "  %-8s  LP bound: %8.1f  LR bound: %8.1f  Best feasible: %8.1f  LR gap: %.1f%%  LP gap: %.1f%%\n",
                    label, r[:lp_bound], r[:lr_bound], r[:best_feas], lr_gap, lp_gap)
            end
        end

        # Experiment 4
        λ, order, inst = results.exp4
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 4: Multiplier Analysis")
        println(io, "="^70)
        println(io, "\n  Top-10 workloads by multiplier value:")
        @printf(io, "  %4s %10s %8s %8s %8s\n", "ID", "λ", "CPU", "Mem", "Sto")
        println(io, "  " * "-"^42)
        for k in 1:min(10, inst.N)
            i = order[k]
            @printf(io, "  %4d %10.2f %8.1f %8.1f %8.1f\n",
                i, λ[i], inst.d_cpu[i], inst.d_mem[i], inst.d_sto[i])
        end
        println(io, "\n  Bottom-5 workloads:")
        println(io, "  " * "-"^42)
        for k in max(1, inst.N-4):inst.N
            i = order[k]
            @printf(io, "  %4d %10.2f %8.1f %8.1f %8.1f\n",
                i, λ[i], inst.d_cpu[i], inst.d_mem[i], inst.d_sto[i])
        end

        # Experiment 5
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 5: Scalability")
        println(io, "="^70)
        for r in results.exp5
            tl_str = ""
            @printf(io, "  N=%3d M=%3d  MIP: %.1fs%s gap=%.1f%%  |  LR: %.1fs gap=%.1f%%\n",
                r.N, r.M, r.mip_time, tl_str, r.mip_gap, r.lr_time, r.lr_gap)
        end

        # Experiment 6
        println(io, "\n" * "="^70)
        println(io, "EXPERIMENT 6: Objective Weight Sensitivity")
        println(io, "="^70)
        @printf(io, "  %-14s %5s %5s %5s %7s %10s %10s %10s\n",
            "Config", "α", "β", "γ", "Servers", "Cost", "Time", "Energy")
        println(io, "  " * "-"^75)
        for r in results.exp6
            @printf(io, "  %-14s %5.2f %5.2f %5.2f %7d %10.1f %10.1f %10.1f\n",
                r.name, r.α, r.β, r.γ, r.servers, r.cost, r.time, r.energy)
        end

        println(io, "\n" * "="^70)
        println(io, "  End of results.")
        println(io, "="^70)
    end
    println("Results saved to $filename")
end

function run_all_experiments(; num_runs=2)
    println("="^70)
    println("  CLOUD RESOURCE ALLOCATION - LAGRANGIAN RELAXATION")
    println("  Full Experimental Suite")
    println("="^70)

    println("\nWarming up JIT (first small solve)...")
    warmup = generate_instance(5, 2; seed=1, size_class=:small)
    solve_direct_mip(warmup; time_limit=10.0)
    lagrangian_relaxation(warmup; max_iter=10)
    println("JIT warmup complete.\n")

    r1 = run_experiment1(; num_runs, verbose=true)
    r2 = run_experiment2(; num_runs, verbose=true)
    r3 = run_experiment3(; num_runs, verbose=true)
    r4 = run_experiment4(; verbose=true)
    r5 = run_experiment5(; verbose=true)
    r6 = run_experiment6(; verbose=true)

    println("\n" * "="^70)
    println("  All experiments complete.")
    println("="^70)

    return (exp1=r1, exp2=r2, exp3=r3, exp4=r4, exp5=r5, exp6=r6)
end
