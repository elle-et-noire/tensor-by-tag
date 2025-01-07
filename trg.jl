include("tnlib.jl")
using Plots, Printf

function retag_by_pos_in_plaq!(A::Tensor)
  pos = [:up, :left, :down, :right]

  pos_to_A = Symbol[]
  pos_in_plaq = Symbol[]
  for j in eachindex(pos)
    !(pos[j] in A.ord_tag) && continue
    push!(pos_to_A, pos[j])
    append!(pos_in_plaq, pos[mod1.([j - 1, j + 1], 4)])
  end
  setdiff!(pos_in_plaq, pos_to_A)

  retag!(A, pos_to_A, pos_in_plaq)
end

function trg(; T=2 / log(1 + √2), maxdim=20, topscale=6)
  A = mktensor(
    [exp(-sum([J[j] * J[mod1(j + 1, 4)] for j in 1:4]) / T)
     for J in Iterators.product(fill([-1, 1], 4)...)],
    [:up, :left, :down, :right]
  )

  F = Vector{Tensor}(undef, 4)
  z = 1.0
  ns, zs, λs, loggn = [], [], [], [0.0]

  for scale in 1:topscale
    @time begin
      print("[scale: $(scale-1) -> $scale]")

      F[2], F[4] = bisect(A, [:down, :right], :l, :r; maxdim)
      F[1], F[3] = bisect(A, [:left, :down], :u, :d; maxdim)
      retag_by_pos_in_plaq!.(F)
      A = reduce(contract, F)
      retag!(A, [:u, :l, :r, :d], [:up, :left, :right, :down])

      TrA = trace(A, [:up, :left], [:down, :right]).array[]
      push!(ns, TrA)
      z *= TrA^(1 / 2^(1 + scale))
      push!(zs, z)

      scale > 1 && push!(loggn, log(ns[end]) - 2log(ns[end-1]))

      M = trace(A, [:up], [:down])
      _, S, _ = svd(M, [:left], :iU, :iV)
      push!(λs, diag(S.array))

    end
  end

  println("\nlog(Z)/N = $(log(z))")

  #------- ope coef -------#

  M = trace(A, [:up], [:down])
  _, U = eigen(M, [:left], :iU; maxdim) # small eigvec
  retag!(U, [:left, :iU], [:Ui, :j])
  V = retag(U, [:Ui, :j], [:Vi, :k])

  B = retag(A, [:up, :left, :down, :right], [:Bu, :Bl, :Bd, :Br])
  W = contract(A, [:up, :down], B, [:Bd, :Bu]) # large eigvec
  _, W = eigen(W, [:left, :Bl], :i; maxdim)
  retag!(W, [:left, :Bl], [:Ui, :Vi])

  C = reduce(contract, [W, U, V])
  coef3p(i, j, k) = C[:i=>i, :j=>j, :k=>k] / C[:i=>1, :j=>1, :k=>1]

  opname = ["I", "σ", "ε"]
  for (k, j, i) in Iterators.product(fill(1:3, 3)...)
    println("[", join(opname[[i, j, k]], ","), "]: ", @sprintf("%.5f", coef3p(i, j, k)))
  end

  #------- plot CFT data -------#

  fplot = scatter(1:topscale, log.(zs); legend=false, msw=0,
    xlabel="2log2 L", ylabel="free energy")

  cplot = scatter(2:topscale,
    (6 / pi) .* (log.(getindex.(λs[2:end], 1) ./ ns[2:end]) .- loggn[2:end]);
    legend=false, msw=0, xlabel="2log2 L", ylabel="central charge")
  hline!(cplot, [0.5]; line=:dot)

  xσplot = scatter(1:topscale, log.(getindex.(λs, 1) ./ getindex.(λs, 2)) ./ 2pi;
    legend=false, msw=0, xlabel="2log2 L", ylabel="xσ")
  hline!(xσplot, [1 / 8]; line=:dot)

  xεplot = scatter(1:topscale, log.(getindex.(λs, 1) ./ getindex.(λs, 3)) ./ 2pi;
    legend=false, msw=0, xlabel="2log2 L", ylabel="xε")
  hline!(xεplot, [1]; line=:dot)

  fplot, cplot, xσplot, xεplot
end


return