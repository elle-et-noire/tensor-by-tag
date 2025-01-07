using LinearAlgebra

abstract type Model end

βc(m::T) where {T<:Model} = 1 / Tc(m)

linkweight(m::M, β) where {M<:Model} = exp.(-β .* ham2(m).(1:locdim(m), (1:locdim(m))'))

localweight(m::M, β) where {M<:Model} = exp.(-β .* ham1(m).(1:locdim(m)))


mutable struct Potts <: Model
  q::Int
  h
  Potts(q, h=0.0) = new(q, h)
end

locdim(p::Potts) = p.q

ham2(::Potts) = (i, j) -> -Int(i == j)

ham1(p::Potts) = i -> -p.h * cispi(2i / p.q)

Tc(p::Potts) = 1 / log(1 + √p.q)

function central_charge(p::Potts)
  dict = Dict(
    2 => 0.5,
    3 => 0.8,
    4 => 1.0
  )
  get(dict, p.q, missing)
end

function total_quantum_dim(p::Potts)
  dict = Dict(
    2 => 1 + 1 / √2,
    3 => sqrt(3 + 6 / √5),
    4 => 1.5 + √2
  )
  get(dict, p.q, missing)
end