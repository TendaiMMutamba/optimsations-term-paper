include("src/instances.jl")
include("src/model.jl")
include("src/subproblems.jl")
include("src/heuristic.jl")
include("src/lagrangian.jl")

println("Quick verification on a small instance (N=10, M=3)")
println("="^50)

inst = small_instance(seed=42)

println("\n1. Direct MIP solver...")
mip = solve_direct_mip(inst; time_limit=60.0)
println("   Optimal value: $(round(mip.objective, digits=1))")
println("   Time: $(round(mip.time, digits=2))s")

println("\n2. LP relaxation bound...")
lp = solve_lp_relaxation(inst)
println("   LP bound: $(round(lp, digits=1))")

println("\n3. Lagrangian relaxation (Polyak, 300 iterations)...")
lr = lagrangian_relaxation(inst; policy=POLYAK, max_iter=300)
println("   Upper bound (feasible): $(round(lr.best_ub, digits=1))")
println("   Lower bound (dual):     $(round(lr.best_lb, digits=1))")
println("   Gap: $(round(lr.gap, digits=1))%")
println("   Time: $(round(lr.time, digits=2))s")

println("\n4. Verification:")
ok1 = lr.best_lb <= mip.objective + 0.01
println("   LB <= MIP optimal?  $ok1  (required: Lagrangian bound is a valid lower bound)")
println("   LR bound >= LP bound? $(lr.best_lb >= lp - 0.01)  (expected: LR with integer subproblems is at least as tight)")
println("   Result: $(ok1 ? "PASSED" : "FAILED")")

println("\n5. Multiplier interpretation (top-3 costliest workloads):")
order = sortperm(lr.λ, rev=true)
for k in 1:3
    i = order[k]
    println("   Workload $i: λ=$(round(lr.λ[i], digits=1)), CPU=$(round(inst.d_cpu[i], digits=1)), Mem=$(round(inst.d_mem[i], digits=1))GB")
end

println("\nQuick test complete. Run 'julia run_all.jl' for the full experiment suite.")
