include("tnlib.jl")

function test()
  Amat = rand(2, 3)
  Bmat = rand(2, 3, 4)
  A = mktensor(Amat, [:iA, :jA])
  B = mktensor(Bmat, [:iB, :jB, :kB])
  old_dims = Tuple[]
  cB = bundleinds(B, [[:kB, :iB]], [:kiB], old_dims)

  sB = ravelinds(cB, [:kiB], [[:kB, :iB]], old_dims)
  sB = permuteinds(sB, [:iB, :jB, :kB], [:jB, :kB, :iB])
  @assert sB.array == B.array
  @assert sB.ord_tag == B.ord_tag

  AB = contract(A, [:jA], B, [:jB])

  Amat = rand(4, 4)
  Bmat = rand(4, 4)
  A = mktensor(Amat, [:iA, :jA])
  B = mktensor(Bmat, [:iB, :jB])
  AB = contract(A, [:jA], B, [:iB])
  @assert AB.array == Amat * Bmat

  println("------- test accepted. -------")
end

test()