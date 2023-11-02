import Pkg

Pkg.add([
  "Catlab",
  "CombinatorialSpaces",
  "Decapodes",
  "MLStyle",
  "MultiScaleArrays",
  "LinearAlgebra",
  "OrdinaryDiffEq",
  "JLD2",
  "SparseArrays",
  "Statistics",
  "GLMakie",
  "GeometryBasics"])

# AlgebraicJulia Dependencies
using Catlab
using Catlab.Graphics
using CombinatorialSpaces
using Decapodes

# External Dependencies
using MLStyle
using MultiScaleArrays
using LinearAlgebra
using OrdinaryDiffEq
using JLD2
using SparseArrays
using Statistics
using GLMakie # Just for visualization
using GeometryBasics: Point2, Point3
Point2D = Point2{Float64};
Point3D = Point3{Float64};

@info("Packages Loaded")

halfar_eq2 = @decapode begin
  h::Form0
  Γ::Form1
  n::Constant

  ḣ == ∂ₜ(h)
  ḣ == ∘(⋆, d, ⋆)(Γ * d(h) * avg₀₁(mag(♯(d(h)))^(n-1)) * avg₀₁(h^(n+2)))
end
glens_law = @decapode begin
  Γ::Form1
  (A,ρ,g,n)::Constant

  Γ == (2/(n+2))*A*(ρ*g)^n
end

@info("Decapodes Defined")

ice_dynamics_composition_diagram = @relation () begin
  dynamics(Γ,n)
  stress(Γ,n)
end
ice_dynamics_cospan = oapply(ice_dynamics_composition_diagram,
  [Open(halfar_eq2, [:Γ,:n]),
  Open(glens_law, [:Γ,:n])])
ice_dynamics = apex(ice_dynamics_cospan)
ice_dynamics1D = expand_operators(ice_dynamics)
infer_types!(ice_dynamics1D, op1_inf_rules_1D, op2_inf_rules_1D)
resolve_overloads!(ice_dynamics1D, op1_res_rules_1D, op2_res_rules_1D)

s′ = EmbeddedDeltaSet1D{Bool, Point2D}()
add_vertices!(s′, 100, point=Point2D.(range(-2, 2, length=100), 0))
add_edges!(s′, 1:nv(s′)-1, 2:nv(s′))
orient!(s′)
s = EmbeddedDeltaDualComplex1D{Bool, Float64, Point2D}(s′)
subdivide_duals!(s, Circumcenter())

@info("Spaces Defined")

beta = 1/18
R₀ = 1
H₀ = 1

n = 3
g = 9.8101
ρ = 910
flwa = 1e-16
#A = fill(1e-16, ne(s))
# Note: The A to use likely differs.
A = fill(1e-16 * 5e7, ne(s))

Gamma = 2.0/(n+2) * flwa * (ρ * g)^n

t₀ = (1/(18*Gamma))*(7/4)^3 * ((R₀^4)/(H₀^7))

@info("Constants Defined")

# Written in radial coordinates:
function height_at_p(r,t)
  t₀ = (1 / (18 * Gamma))*((7/4)^3) * (R₀^4 / H₀^7)
  t = t + t₀
  r = abs(r)
  hterm = (H₀ * (t₀/t))^(1/9)
  #rterm = (1 - ((t₀ / t)^(1/18)* (r/R₀)) ^(4/3))^(3/7)
  rterm = (1 - min(1, ((t₀ / t)^(1/18)* (r/R₀))) ^(4/3))^(3/7)
  hterm * rterm
end

# height_at_p(0.0, 0.0)
# height_at_p(0.5, 0.0)
# height_at_p(1.0, 0.0)
# height_at_p(2.0, 0.0)

h₀ = map(x -> height_at_p(x[1], 0), point(s′))

u₀ = construct(PhysicsState, [VectorForm(h₀)], Float64[], [:dynamics_h])
constants_and_parameters = (
  n = n,
  stress_ρ = ρ,
  stress_g = g,
  stress_A = A)

function generate(sd, my_symbol; hodge=GeometricHodge())
  e_vecs = map(edges(sd)) do e
    point(sd, sd[e, :∂v0]) - point(sd, sd[e, :∂v1])
  end
  neighbors = map(vertices(sd)) do v
    union(incident(sd, v, :∂v0), incident(sd, v, :∂v1))
  end
  n_vecs = map(neighbors) do es
    [e_vecs[e] for e in es]
  end
  I = Vector{Int64}()
  J = Vector{Int64}()
  V = Vector{Float64}()
  for e in 1:ne(s)
      append!(J, [s[e,:∂v0],s[e,:∂v1]])
      append!(I, [e,e])
      append!(V, [0.5, 0.5])
  end
  avg_mat = sparse(I,J,V)
  op = @match my_symbol begin
    :♯ => x -> begin
      map(neighbors, n_vecs) do es, nvs
        sum([nv*norm(nv)*x[e] for (e,nv) in zip(es,nvs)]) / sum(norm.(nvs))
      end
    end
    :mag => x -> begin
      norm.(x)
    end
    :avg₀₁ => x -> begin
      avg_mat * x
    end
    :^ => (x,y) -> x .^ y
    :* => (x,y) -> x .* y
    :show => x -> begin
      @show x
      x
    end
    x => error("Unmatched operator $my_symbol")
  end
  return (args...) -> op(args...)
end

sim = eval(gensim(ice_dynamics1D, dimension=1))
fₘ = sim(s, generate)

# function f()
#     tₑ = 300 * 1000
#     prob = ODEProblem(fₘ, u₀, (0, tₑ), constants_and_parameters)
#     @info("Solving")
#     soln = solve(prob, Tsit5())
#     @show soln.retcode
#     @info("Done")
#     return soln.u
# end
#
# f()
