using Dates

println("Loading source files...")
include("src/instances.jl")
include("src/model.jl")
include("src/subproblems.jl")
include("src/heuristic.jl")
include("src/lagrangian.jl")
include("src/experiments.jl")
include("src/plots.jl")
println("All modules loaded.\n")

results = run_all_experiments(num_runs=2)

println("\nSaving results to file...")
save_results(results, "results.txt")

println("\nGenerating figures...")
generate_all_plots(results; output_dir=".")

println("\nDone. Results saved to results.txt. Figures saved as fig1-fig4 PNG files.")
