using OrdinaryDiffEq, Plots
using SteadyStateDiffEq

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

struct ThreeLevelFisheryParameters
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
    u0 = [B1, B2, B3]
    tspan = (0.0, 10.0)
    prob = create_three_level_problem(u0, tspan, pstruct)
    sol = solve(prob, Tsit5())
    # plot(sol, title="Default Zhou and Smith (2017)", xaxis="Time", yaxis="Biomass")
    # Use logscale instead.
    plot(sol, title="Default Zhou and Smith (2017)", xaxis="Time", yaxis="Biomass", yscale=:log10)
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
    prob = SteadyStateProblem(
        three_level_fishery_model,
        u0,
        p
    )
    return prob
end

function simulate_ss_three_level_fishery(B1, B2, B3, pstruct)
    u0 = [B1, B2, B3]
    # FIXME WIP so we need compile the problem once and then reparameterize with pstruct actually.
    prob = create_three_level_ss_problem(u0, pstruct)
    sol = solve(prob, SSRootfind())
    return sol
end

function simulate_ss_three_level_fishery(B1, B2, B3, F1, F2, F3)
    pstruct = default_three_level_fishery_parameters(F1, F2, F3)
    return simulate_ss_three_level_fishery(B1, B2, B3, F1, F2, F3, pstruct)
end

function simulate_ss_three_level_fishery(F1, F2, F3)
    # Default stable parameters. Taken by solving for steady state with no fishing.
    B1, B2, B3 = 958.833, 174.356, 33.476
    return simulate_ss_three_level_fishery(B1, B2, B3, F1, F2, F3)
end

function sustained_yield(sssol, pstruct)
    # The sustained yield is the steady state biomass times the fishing mortality rate.
    # This is not the necessarily the "sustainable" yield if fishing rates drive a species
    #  to extinction.
    return ssol * [pstruct.F1 * pstruct.r1, pstruct.F2 * pstruct.r2, pstruct.F3 * pstruct.r3]
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

