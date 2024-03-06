using OrdinaryDiffEq, Plots
using SteadyStateDiffEq
using NonlinearSolve
using SciMLNLSolve

# The general holling-tanner differential equation.
# This subtracts predator and fishing mortality from capacity-limited growth.
# B is the biomass for this trophic level.
# K is its carrying capacity, which may be a function of prey biomass.
# M is the mortality rate, potentially due to predation.
# F is the fishing mortality rate.
holling_tanner(B, r, K, M, F) = r * B * (1 - B / K) - M * B - F * B

# Mortality rate of prey due to predation.
# This plugs in as M above for prey species with biomass B.
# p is the maximum rate of predation.
# B is the biomass of the prey, and B_pred is the biomass of the predator.
# D is the biomass when the predation rate reaches half of its maximum.
mortality_from_predation(B, B_pred, p, D) = (p * B_pred) / (D + B)

# Carrying capacity of the predator — this plugs in as K above for predator species.
# e is the proportion of prey biomass that is converted into predator biomass.
carrying_capacity_of_predator(B_prey, e) = B_prey * e

# Equation for intermediate trophic levels that are both prey and predator.
function intermediate_trophic_level(B, r, B_prey, e, B_pred, p, D, F)
    K = carrying_capacity_of_predator(B_prey, e)
    M = mortality_from_predation(B, B_pred, p, D)
    holling_tanner(B, r, K, M, F)
end

# The top trophic level that has some constant natural mortality rate.
function apex_trophic_level(B, r, B_prey, e, M, F)
    K = carrying_capacity_of_predator(B_prey, e)
    return holling_tanner(B, r, K, M, F)
end

# The bottom trophic level that has some constant carrying capacity
function forage_trophic_level(B, r, K, B_pred, p, D, F)
    M = mortality_from_predation(B, B_pred, p, D)
    return holling_tanner(B, r, K, M, F)
end

# And now the differential equations fishery model.
function three_level_fishery_model(du, u, p, t)
    # Unpack the parameters.
    r1, K1, p12, D1, F1, r2, e12, p23, D2, F2, r3, e23, M3, F3 = p

    # Unpack the state.
    B1, B2, B3 = u

    # Calculate the derivatives.
    du[1] = forage_trophic_level(B1, r1, K1, B2, p12, D1, F1)
    du[2] = intermediate_trophic_level(B2, r2, B1, e12, B3, p23, D2, F2)
    du[3] = apex_trophic_level(B3, r3, B2, e23, M3, F3)
end

mutable struct ThreeLevelFisheryParameters
    r1::Float64
    K1::Float64
    p12::Float64
    D1::Float64
    F1::Float64
    r2::Float64
    e12::Float64
    p23::Float64
    D2::Float64
    F2::Float64
    r3::Float64
    e23::Float64
    M3::Float64
    F3::Float64
end

function ThreeLevelFisheryParameters(;r1::Float64, K1::Float64, p12::Float64, D1::Float64, F1::Float64,
                                      r2::Float64, e12::Float64, p23::Float64, D2::Float64, F2::Float64,
                                      r3::Float64, e23::Float64, M3::Float64, F3::Float64)
    return ThreeLevelFisheryParameters(r1, K1, p12, D1, F1, r2, e12, p23, D2, F2, r3, e23, M3, F3)
end

function create_three_level_problem(u0, tspan, pstruct)
    p = [
        pstruct.r1,
        pstruct.K1,
        pstruct.p12,
        pstruct.D1,
        pstruct.F1,
        pstruct.r2,
        pstruct.e12,
        pstruct.p23,
        pstruct.D2,
        pstruct.F2,
        pstruct.r3,
        pstruct.e23,
        pstruct.M3,
        pstruct.F3
    ]
    prob = ODEProblem(
        three_level_fishery_model,
        u0,
        tspan,
        p
    )
    return prob
end

# "Ground truth" stable parameters, taken from Table 1 of Zhou and Smith (2017).
function default_three_level_fishery_parameters(F1, F2, F3)
    K1 = 1000.0
    r1, r2, r3 = 2.0, 1.0, 0.25
    p12, p23 = 0.5, 0.5
    D1, D2 = 100.0, 10.0
    e12, e23 = 0.2, 0.2
    M3 = 0.01
    return ThreeLevelFisheryParameters(
        r1=r1,
        K1=K1,
        p12=p12,
        D1=D1,
        F1=F1,
        r2=r2,
        e12=e12,
        p23=p23,
        D2=D2,
        F2=F2,
        r3=r3,
        e23=e23,
        M3=M3,
        F3=F3
    )
end

function plot_default_three_level_fishery(B1, B2, B3, F1, F2, F3)
    pstruct = default_three_level_fishery_parameters(F1, F2, F3)
    plot_three_level_fishery(B1, B2, B3, pstruct)
end

function plot_three_level_fishery(B1, B2, B3, pstruct::ThreeLevelFisheryParameters)
    u0 = [B1, B2, B3]
    tspan = (0.0, 10.0)
    prob = create_three_level_problem(u0, tspan, pstruct)
    sol = solve(prob, Tsit5())
    plot(sol, title="Zhou and Smith (2017) Model", xaxis="Time", yaxis="Biomass", yscale=:log10)
end

function plot_three_level_fishery(pstruct::ThreeLevelFisheryParameters)
    return plot_three_level_fishery(DEFAULT_B1, DEFAULT_B2, DEFAULT_B3, pstruct)
end

function create_three_level_ss_problem(u0, pstruct)
    p = [
        pstruct.r1,
        pstruct.K1,
        pstruct.p12,
        pstruct.D1,
        pstruct.F1,
        pstruct.r2,
        pstruct.e12,
        pstruct.p23,
        pstruct.D2,
        pstruct.F2,
        pstruct.r3,
        pstruct.e23,
        pstruct.M3,
        pstruct.F3
    ]
    prob = ODEProblem(
        three_level_fishery_model,
        u0,
        (0.0, Inf),
        p
    )
    return NonlinearProblem(prob)
end

DEFAULT_B1 = 958.833
DEFAULT_B2 = 174.356
DEFAULT_B3 = 33.476

function simulate_ss_three_level_fishery(B1, B2, B3, pstruct)
    u0 = [B1, B2, B3]
    # FIXME WIP so we need compile the problem once and then reparameterize with pstruct actually.
    prob = create_three_level_ss_problem(u0, pstruct)
    sol = solve(prob, NLSolveJL())
    return sol
end

function simulate_ss_three_level_fishery(pstruct::ThreeLevelFisheryParameters)
    return simulate_ss_three_level_fishery(DEFAULT_B1, DEFAULT_B2, DEFAULT_B3, pstruct)
end

function simulate_ss_three_level_fishery(B1, B2, B3, F1, F2, F3)
    pstruct = default_three_level_fishery_parameters(F1, F2, F3)
    return simulate_ss_three_level_fishery(B1, B2, B3, F1, F2, F3, pstruct)
end

function simulate_ss_three_level_fishery(F1, F2, F3)
    # Default stable parameters. Taken by solving for steady state with no fishing.
    return simulate_ss_three_level_fishery(DEFAULT_B1, DEFAULT_B2, DEFAULT_B3, F1, F2, F3)
end

function sustained_yield(sssol, pstruct, price1, price2, price3)
    # The sustained yield is the steady state biomass times the fishing mortality rate.
    # This is not the necessarily the "sustainable" yield if fishing rates drive a species
    #  to extinction.
    return sum(sssol .* [
        pstruct.F1 * pstruct.r1 * price1,
        pstruct.F2 * pstruct.r2 * price2,
        pstruct.F3 * pstruct.r3 * price3
    ])
end

function sustained_yield(sssol, pstruct)
    return sustained_yield(sssol, pstruct, 1.0, 1.0, 1.0)
end

function disturbance_index(sssol_fished, sssol_unfished)
    # The disturbance index (courtesy of Bundy et al. 2005 and Zhou and Smith 2017) simply
    #  sums over the deviations in biomass ratios between trophic levels between the fished
    #  and unfished regimes.
    di = 0.0
    for i in 1:(length(sssol_fished) - 1)
        di += abs(sssol_fished[i+1] / sssol_fished[i] - sssol_unfished[i+1] / sssol_unfished[i])
    end
    return di
end

function update_parameter!(pstruct::ThreeLevelFisheryParameters, param_name::String, value)
    field = Symbol(param_name)
    if hasproperty(pstruct, field)
        setfield!(pstruct, field, value)
    else
        error("Parameter '$param_name' does not exist in ThreeLevelFisheryParameters")
    end
end


function plot_parameter_sensitivities(
    param_name::String,
    param_values::Array{Float64, 1},
    F1::Float64, F2::Float64, F3::Float64
)
    biomass = []
    disturbance = []
    totalyield = []
    fished_ss = []
    unfished_ss = []

    fished_pstruct = default_three_level_fishery_parameters(F1, F2, F3)
    unfished_pstruct = default_three_level_fishery_parameters(0.0, 0.0, 0.0)

    for value in param_values
        # Update the specified parameter
        update_parameter!(fished_pstruct, param_name, value)
        update_parameter!(unfished_pstruct, param_name, value)
        # Simulate out to steady states.
        fished_sssol = simulate_ss_three_level_fishery(fished_pstruct)
        unfished_sssol = simulate_ss_three_level_fishery(unfished_pstruct)
        # Sum over sssol to get total relative biomass. Larger is better.
        push!(biomass, sum(fished_sssol)/sum(unfished_sssol))
        # Compute the disturbance index.
        push!(disturbance, disturbance_index(fished_sssol, unfished_sssol))
        # Compute the total yield. This is not a function of unfished and therefore 0 yield.
        push!(totalyield, sum(sustained_yield(fished_sssol, fished_pstruct, 1.0, 30.0, 1000.0)))

        # Add the steady states to the list for later plotting. Copy.
        push!(fished_ss, copy(fished_sssol))
        push!(unfished_ss, copy(unfished_sssol))
    end

    fished_ss = permutedims(hcat(fished_ss...))
    unfished_ss = permutedims(hcat(unfished_ss...))

    pbiomass = plot(param_values, biomass, label="Relative Biomass", title="Parameter Sensitivities", xaxis=param_name)
    # Plot each species relative biomass on the same plot as above.
    for i in 1:3
        plot!(param_values, fished_ss[:, i] ./ unfished_ss[:, i], label="TL $i", xaxis=param_name)
    end
    pdi = plot(param_values, disturbance, label="Disturbance Index", xaxis=param_name)
    pyield = plot(param_values, totalyield, label="Total Yield", xaxis=param_name)

    # Plot all three with the same x axis, title with the parameter of interest. Make tall to account for stacked plots.
    plot(pbiomass, pdi, pyield, layout=(3, 1), title=param_name, size=(800, 500))

end

# plot_parameter_sensitivities("e12", collect(range(0.1, 0.3, 100)), 0.1 * 2.0, 0.1 * 1.0, 0.1 * 0.25)

# As an initial exploration, we want to find out when extinction occurs within plausible parameter ranges.
# K1 — 1000 fixed (this won't affect ratios for the same reason TL1 fishing doesn't, just time to extinction)
# r1 — (1.0, 2.0, 5.68); r2 — (0.5, 1.0, 2.0); r3 — (0.114, 0.25, 1.0)
#  - comment: after perusing FishBase (fishbase.se) a bit and looking at r in the "Estimates based on models" of common commercial fish, and multiplying
#             by 2 as specified by Zhou and Smith (2017), these ranges seem plausible.
RANGE_R1 = (1.0, 2.0, 5.68)
RANGE_R2 = (0.5, 1.0, 2.0)
RANGE_R3 = (0.114, 0.25, 1.0)
# p12 — (0.01, 0.5, 10.0); p23 — (0.01, 0.5, 10.0) logscale, fairly clear from their exposition, and I'm cutting off an order on each side.
RANGE_P12 = (0.01, 0.5, 10.0)
RANGE_P23 = (0.01, 0.5, 10.0)
# D1 — 100; D2 — 10 fixed (they don't give enough information to vary these)
RANGE_D1 = (50.0, 100.0, 200.0)
RANGE_D2 = (5.0, 10.0, 20.0)
# e12 — (0.02, 0.2, 0.24); e23 — (0.02, 0.2, 0.24) logscale (probably...not super clear from their exposition)
RANGE_E12 = (0.02, 0.2, 0.24)
RANGE_E23 = (0.02, 0.2, 0.24)
# M3 — 0.01 fixed (they just ballpark/handwave this)
# F1, F2, F3 — (0, r, 2r) respectively, where r is the maximum intrinsic growth rate of the species. In our
#  setting this is unknown, so take this to be the expected r. If f ever exceeds r you get extinction, with some additional
#  constraints arising from the other parameters as analyzed by Zhou and Smith (2017).

# Zhou and Smith (2017) report:
# 1. that models are less sensitive to K, r, D and M.
#  — comment: the relative values of r to fishing mortality is obviously important.
# 2. biologically, r ranges from 0.057 to 2.84 for the schafer model. they multiply by 2 to account for predation, so r — (0.114, 5.68), decreasing with TL
# 3. p rarely measured, but they say similar qualitative curves across 0.001 to 100.0 (on a log scale 0.5 is roughly in the middle), even though big changes
#     are induced to absolute biomass ratios.
# 4. they claim greatest sensitivity to energy transfer efficiency, which controls the carrying capacity of the predators (makes sense)
#     biologically, e ranges between 0.02 and 0.24.

# Brainstorm:
# So we don't know exactly what r is, and couldn't know what it is when setting fishing rates. As such, we let that remain uncertain.
# This will mean that an extinction averse fishing policy will set its fishing rates to be lower than say, the 10th percentile of likely
#  growth rates. That would put it into a regime where extinction has a 10% chance of occurring, though in reality it would need to be set lower
#  (at least for higher trophic levels) to account for the fact that their capacity will be reduced due to fishing for their food.
# This translates to rare r probably being the primary driver of catastrophic events. The other parameters will augment this to
#  greater rarity, requiring the fishing rates to be even lower. For internal analysis, we can quantify the degree to which other parameters
#  are affecting this by constraining to non-extincting r and seeing the percentage of prior cases in which extinction occurs.

# So this is a bit unrealistic, because in reality, we would have time to react if the system started behaving in unexpected ways (i.e. our estimates)
#  for R were way off. This would be akin to our online setting, where we just want to avoid extinction for the next year, after we which we recalibrate
#  our parameter estimates with the year's data, and then set our fishing rates for the next year.

# TODO WIP still wanting to do our pairwise grid scatter simulation here to sanity check this.
# 1. linspace across all unfixed parameters.
# 2. solve fished and unfished steady states across the whole grid.
# 3. sort into three categories: extinct in unfished (fished irrelevant), extinct in fished but not in unfished, and not extinct in either.
# 4. plot pairwise scatters for each pair of parameters and color by the three categories.

function rand_unif(a, b, n)
    return a .+ (b - a) .* rand(n)
end

function grid_scatter_extinction(F1, F2, F3, n)
    
    results = zeros(n) .- 1
    parameters = zeros(n, 9)
    parameters[:, 1] = rand_unif(RANGE_R1[1], RANGE_R1[3], n)
    parameters[:, 2] = rand_unif(RANGE_R2[1], RANGE_R2[3], n)
    parameters[:, 3] = rand_unif(RANGE_R3[1], RANGE_R3[3], n)
    parameters[:, 4] = rand_unif(RANGE_P12[1], RANGE_P12[3], n)
    parameters[:, 5] = rand_unif(RANGE_P23[1], RANGE_P23[3], n)
    parameters[:, 6] = rand_unif(RANGE_D1[1], RANGE_D1[3], n)
    parameters[:, 7] = rand_unif(RANGE_D2[1], RANGE_D2[3], n)
    parameters[:, 8] = rand_unif(RANGE_E12[1], RANGE_E12[3], n)
    parameters[:, 9] = rand_unif(RANGE_E23[1], RANGE_E23[3], n)

    pstruct = default_three_level_fishery_parameters(F1, F2, F3)

    for i in 1:n
        update_parameter!(pstruct, "r1", parameters[i, 1])
        update_parameter!(pstruct, "r2", parameters[i, 2])
        update_parameter!(pstruct, "r3", parameters[i, 3])
        update_parameter!(pstruct, "p12", parameters[i, 4])
        update_parameter!(pstruct, "p23", parameters[i, 5])
        update_parameter!(pstruct, "D1", parameters[i, 6])
        update_parameter!(pstruct, "D2", parameters[i, 7])
        update_parameter!(pstruct, "e12", parameters[i, 8])
        update_parameter!(pstruct, "e23", parameters[i, 9])

        update_parameter!(pstruct, "F1", 0.0)
        update_parameter!(pstruct, "F2", 0.0)
        update_parameter!(pstruct, "F3", 0.0)
        unfished_sssol = simulate_ss_three_level_fishery(pstruct)

        # If any unfished species are extinct, i.e. any less than 1e-6, then store a result of 1.
        if any(unfished_sssol .< 1e-6)
            results[i] = 1
            continue
        end

        update_parameter!(pstruct, "F1", F1)
        update_parameter!(pstruct, "F2", F2)
        update_parameter!(pstruct, "F3", F3)
        fished_sssol = simulate_ss_three_level_fishery(pstruct)

        # If any fished species are extinct, i.e. any less than 1e-6, then store a result of 2
        if any(fished_sssol .< 1e-6)
            results[i] = 2
        # If no species are extinct, store a result of 0.
        else
            results[i] = 0
        end
    end

    print("unique results", unique(results))
    # Percentage of each case.
    print("percentage of each case", [count(==(i), results) / length(results) for i in [-1, 0, 1, 2]])
    
    # Now plot the results in a 2d grid for each pair of parameters. Each pair gets a cell, and in that cell we can plot a simple scatter.
    # Just ignore the diagonals.
    plots = []
    for i in axes(parameters, 2)
        for j in axes(parameters, 2)
            if i != j
                scatterplot = scatter(
                    parameters[:, i], parameters[:, j],
                    zcolor=results, color=:viridis, markersize=1.0, markerstrokewidth=0, alpha=0.5,
                    legend=false, colorbar=false, xrotation=45
                )
                push!(plots, scatterplot)
            else
                # HACK suggested by chatgpt — seems like there should be a better way to do this.
                emptyplot = plot(legend=false, axis=false, grid=false, foreground_color_subplot=:white, background_color_subplot=:white)
                push!(plots, emptyplot)
            end
        end
    end

    # Plot all the scatterplots in a grid.
    plot(plots..., layout=(size(parameters, 2), size(parameters, 2)), size=(1200, 1200), fontsize=4)
end
