include("tnlib.jl")
include("model.jl")
using Plots, LsqFit, Printf

# change tensor name
chtn(T::Tensor, AB::Pair{Symbol,Symbol})::Tensor = retag(T, T.ord_tag, replace.(T.ord_tag, AB))
chtn!(T::Tensor, AB::Pair{Symbol,Symbol})::Tensor = retag!(T, T.ord_tag, replace.(T.ord_tag, AB))

# A and B are connected via iAB and iBA
function isometry(A::Tensor, iAB::Vector{Symbol}, oA::Vector{Symbol},
  B::Tensor, iBA::Vector{Symbol}, oB::Vector{Symbol}; maxdim)

  iA = setdiff(A.ord_tag, [iAB; oA])
  AA = contract(A, iA, conj!(chtn(A, :A => :Adag)), replace.(iA, :A => :Adag))
  iB = setdiff(B.ord_tag, [iBA; oB])
  BB = contract(B, iB, conj!(chtn(B, :B => :Bdag)), replace.(iB, :B => :Bdag))
  AABB = contract(AA, [iAB; replace.(iAB, :A => :Adag)], BB, [iBA; replace.(iBA, :B => :Bdag)])
  U, _, _ = svd(AABB, [oA; oB], :Ucoarse, :iV; maxdim)
  chtn!(U, :A => :UA)
  chtn!(U, :B => :UB)
end

# calculate partition function on torus, Klein bottle and RP2
function hotrg(T::Tensor; topscale=6, maxdim=16)
  @assert Set(T.ord_tag) == Set([:u, :l, :d, :r])
  retag!(T, T.ord_tag, Symbol.(:T, T.ord_tag))

  d0 = size(T.array, 1)
  O = delta([d0, d0], [:Oi, :Oj]) # initial spatial-reflection-operator is identity

  # norm, eigval, crosscap, rainbow, reflection operator
  ns, λs, Cis, Ris, Os = [], [], [], [], []

  for scale in 1:topscale # renormalize 4 -> 1 per loop
    @time begin
      print("[scale: $(scale-1) -> $scale]")

      #------- renormalize vertically -------#

      A = chtn(T, :T => :A)
      B = chtn(T, :T => :B)
      AB = contract(A, [:Ad], B, [:Bu]) # B is beneath A

      #------- normalize -------#

      M = trace(AB, [:Au], [:Bd]) # horizontal transfer matrix
      n = trace(M, [:Al, :Bl], [:Ar, :Br]).array[]
      push!(ns, n)
      M.array ./= n
      AB.array ./= n # avoid overflow

      #------- obtain CFT data -------#

      eigvec, eigval, _ = svd(M, [:Al, :Bl], :iU, :iV)
      push!(λs, diag(eigval.array))
      push!(Cis, trace(eigvec, [:Al], [:Bl]).array)
      push!(Ris, contract(eigvec, [:Al, :Bl], O, [:Oi, :Oj]).array)

      #------- truncate -------#

      U = isometry(A, [:Ad], [:Al], B, [:Bu], [:Bl]; maxdim)
      @assert Set(U.ord_tag) == Set([:UAl, :UBl, :Ucoarse])
      V = chtn(U, :U => :V)

      AB = contract(AB, [:Al, :Bl], U, [:UAl, :UBl])
      AB = contract(AB, [:Ar, :Br], V, [:VAl, :VBl])
      T = retag(AB, [:Au, :Ucoarse, :Bd, :Vcoarse], [:Tu, :Tl, :Td, :Tr])

      #------- renormalize `O` -------#

      O1 = retag(O, [:Oi, :Oj], [:UAl, :VBl])
      O2 = retag(O, [:Oi, :Oj], [:UBl, :VAl])
      O = reduce(contract, [U, O1, O2, V])
      retag!(O, [:Ucoarse, :Vcoarse], [:Oi, :Oj])
      push!(Os, O.array)

      #------- check the efficiency of `O` -------#

      Ol = retag(O, [:Oj], [:Tl])
      Or = retag(O, [:Oi], [:Tr])
      OTO = reduce(contract, [Ol, T, Or])
      retag!(OTO, [:Oi, :Oj], [:Tl, :Tr])
      OTO = permuteinds(OTO, OTO.ord_tag, T.ord_tag)

      revT = permuteinds(T, [:Tu, :Tl, :Td, :Tr], [:Td, :Tl, :Tu, :Tr])
      print(" [norm(T-revT)=$(@sprintf("%.3f", norm(T.array - revT.array)))]")
      print(" [norm(OTO-revT)=$(@sprintf("%.3e", norm(OTO.array - revT.array)))]")

      #------- renormalize horizontally -------#

      A = chtn(T, :T => :A)
      B = chtn(T, :T => :B)
      AB = contract(A, [:Ar], B, [:Bl]) # B is to the right of A

      U = isometry(A, [:Ar], [:Au], B, [:Bl], [:Bu]; maxdim)
      @assert Set(U.ord_tag) == Set([:UAu, :UBu, :Ucoarse])
      V = chtn(U, :U => :V)

      AB = contract(AB, [:Au, :Bu], U, [:UAu, :UBu])
      AB = contract(AB, [:Ad, :Bd], V, [:VAu, :VBu])
      T = retag(AB, [:Ucoarse, :Al, :Vcoarse, :Br], [:Tu, :Tl, :Td, :Tr])
    end
  end

  ns, λs, Cis, Ris, Os
end

function vertexcentered_unitcell(; hweight, vweight=hweight, locweight=ones(size(hweight, 1)))
  H = mktensor(hweight, [:H1i, :H2j])
  H1, H2 = bisect(H, [:H1i], :H1j, :H2i)
  V = mktensor(vweight, [:V1i, :V2j])
  V1, V2 = bisect(V, [:V1i], :V1j, :V2i)
  W = mktensor(locweight, [:W])

  vertex = delta(fill(size(hweight, 1), 5), [:V2j, :H2j, :V1i, :H1i, :W])

  T = reduce(contract, [vertex, H1, V1, H2, V2, W])
  retag!(T, [:V2i, :H2i, :V1j, :H1j], [:u, :l, :d, :r])
end

function critical(p::M=Potts(2); topscale=6, maxdim=16) where {M<:Model}

  #------- prepare initial tensor -------#

  T = vertexcentered_unitcell(; hweight=linkweight(p, 1 / Tc(p)))

  #------- check the symmetricity of initial tensor -------#

  perms(l) = isempty(l) ? [l] : [[x; y] for x in l for y in perms(setdiff(l, x))]
  for s in perms(1:4)
    n = norm(T.array - permutedims(T.array, s))
    if n > 1e-15
      println("norm(T-T[$s]) = $n")
    end
  end

  #------- HOTRG -------#

  ns, λs, Cis, Ris, Os = hotrg(T; topscale, maxdim)

  #------- plot central charges -------#

  cplot = scatter(2:topscale, log.(getindex.(λs[2:end], 1) ./ ns[2:end]) ./ pi;
    legend=false, msw=0,
    xlabel="log2 β", ylabel="central charge"
  )
  hline!(cplot, [central_charge(p)]; line=:dot)

  #------- plot Klein Bottle entropy -------#

  kbplot = scatter(2:topscale, [abs2.(getindex.(Cis[2:end], j)) for j in 1:3];
    legend=false, msw=0,
    xlabel="log2 β", ylabel="Klein Bottle entropy"
  )
  hline!(kbplot, [total_quantum_dim(p), 1 - 1 / √2, 0]; line=:dot)

  #------- plot RP2 entropy -------#

  S_rp2 = abs.(getindex.(Ris, 1) .* getindex.(Cis, 1))
  rp2plot = scatter(1:topscale, S_rp2;
    legend=false, msw=0, yscale=:log10,
    xlabel="log2 β", ylabel="RP2 entropy"
  )

  rng = 2:5
  f(x, b) = central_charge(p) / 4 * log(2) * x + b
  fit = curve_fit(
    (x, b) -> f.(x, b[1]),
    rng, log.(S_rp2[rng]), [0.0]
  )
  plot!(rp2plot, 1:topscale, exp.(f.(1:topscale, fit.param[1])); line=:dot)

  println("\n> eigvals of O")
  for scale in 1:topscale
    println("[scale: $scale] +1: ", count(>(0), diag(Os[scale])), ", -1: ", count(<(0), diag(Os[scale])))
  end

  cplot, kbplot, rp2plot, Os
end


return