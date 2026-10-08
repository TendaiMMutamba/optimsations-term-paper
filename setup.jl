using Pkg

println("Installing required packages...")
Pkg.add("JuMP")
Pkg.add("HiGHS")
Pkg.add("Plots")
println("All packages installed.")
