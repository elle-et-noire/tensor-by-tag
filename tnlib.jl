import Base: getindex, setindex!, conj!
import LinearAlgebra: svd, eigen
using LinearAlgebra

mutable struct Tensor
  array::AbstractArray
  ord_tag::Vector{Symbol} # conver order to tag
  tag_ord::Dict{Symbol,Int64} # convert tag to order

  Tensor(array, ord_tag, tag_ord) = begin
    @assert length(ord_tag) == length(tag_ord) # prohibit the duplication among tags
    @assert length(ord_tag) == ndims(array)
    new(array, ord_tag, tag_ord)
  end
end

function mktensor(A::AbstractArray,
  tags::Vector{Symbol}=Symbol.("i" .* string.(1:ndims(A))))::Tensor

  Tensor(A, tags, Dict([tags[j] => j for j in eachindex(tags)]))
end

function getindex(A::Tensor, inds...)
  dict = Dict(inds...)
  inds = get.(Ref(dict), A.ord_tag, missing)
  A.array[inds...]
end

function setindex!(A::Tensor, val, inds...)
  dict = Dict(inds...)
  inds = get.(Ref(dict), A.ord_tag, missing)
  A.array[inds...] = val
end

function retag!(A::Tensor, oldtags::Vector{Symbol}, newtags::Vector{Symbol})::Tensor
  @assert length(oldtags) == length(newtags)

  for (ol, nl) in zip(oldtags, newtags)
    i = A.tag_ord[ol]
    delete!(A.tag_ord, ol)
    A.tag_ord[nl] = i
    A.ord_tag[i] = nl
  end
  @assert length(A.ord_tag) == length(A.tag_ord) # prohibit duplication
  A
end

"""specify as `copy=deepcopy` if you want to deepcopy `A.array`"""
function retag(A::Tensor, oldtags::Vector{Symbol}, newtags::Vector{Symbol}; copy=copy)::Tensor
  tags = deepcopy(A.ord_tag)
  for (oldtag, newtag) in zip(oldtags, newtags)
    tags[A.tag_ord[oldtag]] = newtag
  end
  mktensor(copy(A.array), tags)
end

function bundleinds(A::Tensor, oldtags::Vector{Vector{Symbol}},
  bundledtags::Vector{Symbol}, olddims=Tuple[])::Tensor
  @assert length(oldtags) == length(bundledtags)

  oldords = [getindex.(Ref(A.tag_ord), ls) for ls in oldtags] # [[2,4,3], [1,5]]
  oldords_vcat = vcat(oldords...) # [2,4,3,1,5]
  push!(olddims, [size(A.array)[os] for os in oldords]...) # [(20,20,20),(40,40)]

  remaining_ords = setdiff(1:ndims(A.array), oldords_vcat) # [6,7]
  remaining_dims = size(A.array)[remaining_ords] # (40,40)
  remaining_tags = A.ord_tag[remaining_ords] # [:i6, :i7]

  Barray = permutedims(A.array, vcat(oldords_vcat, remaining_ords...))
  newdims = Tuple(vcat(prod.(olddims), remaining_dims...)) # (8000, 1600, 40, 40)
  mktensor(reshape(Barray, newdims), [bundledtags; remaining_tags])
end

function ravelinds(A::Tensor, oldtags::Vector{Symbol},
  finetags::Vector{Vector{Symbol}}, finedims::Vector{Tuple})::Tensor
  @assert length(oldtags) == length(finetags)
  @assert length(oldtags) == length(finedims)

  old_ords = getindex.(Ref(A.tag_ord), oldtags)

  new_dims = Int[]
  new_tags = Symbol[]
  for iA in 1:ndims(A.array)
    i_split = findfirst(==(iA), old_ords)
    if isnothing(i_split)
      push!(new_dims, size(A.array)[iA])
      push!(new_tags, A.ord_tag[iA])
    else
      push!(new_dims, finedims[i_split]...)
      push!(new_tags, finetags[i_split]...)
    end
  end

  mktensor(reshape(A.array, Tuple(new_dims)), new_tags)
end

function permuteinds(A::Tensor, oldtags::Vector{Symbol}, newtags::Vector{Symbol})::Tensor
  @assert length(oldtags) == length(newtags)

  oldords = getindex.(Ref(A.tag_ord), oldtags)
  newords = getindex.(Ref(A.tag_ord), newtags)

  # the form of perm is determined from the properties:
  # - perm(oldord[s], neword[s]) == perm(oldord, neword)
  # - perm(identity, neword) == neword
  perm = newords[invperm(oldords)]

  mktensor(permutedims(A.array, perm), A.ord_tag[perm])
end

"""`oA`---`A`---(`iA`==`iB`)---`B`---`oB`"""
function contract(A::Tensor, iA::Vector{Symbol}, B::Tensor, iB::Vector{Symbol})::Tensor
  @assert length(iA) == length(iB)

  # append common inds to both `iA` and `iB`
  oA = setdiff(A.ord_tag, iA)
  oB = setdiff(B.ord_tag, iB)
  iAB = intersect(oA, oB)
  setdiff!(oA, iAB)
  setdiff!(oB, iAB)
  append!(iA, iAB)
  append!(iB, iAB)

  olddims_A = Tuple[]
  _A = bundleinds(A, [oA, iA], [:__oA__, :__iA__], olddims_A)

  olddims_B = Tuple[]
  _B = bundleinds(B, [iB, oB], [:__iB__, :__oB__], olddims_B)

  _AB = mktensor(_A.array * _B.array, [:__oA__, :__oB__])
  ravelinds(_AB, [:__oA__, :__oB__], [oA, oB], Tuple[olddims_A[1], olddims_B[2]])
end

"""contract the indices of same labels"""
function contract(A::Tensor, B::Tensor)::Tensor
  iAB = intersect(A.ord_tag, B.ord_tag)
  contract(A, iAB, B, iAB)
end

"""take trace over `iA`==`jA`"""
function trace(A::Tensor, iA::Vector{Symbol}, jA::Vector{Symbol})::Tensor
  @assert length(iA) == length(jA)

  remaining_tags = setdiff(A.ord_tag, [iA; jA])
  remaining_ords = getindex.(Ref(A.tag_ord), remaining_tags)
  remaining_dims = size(A.array)[remaining_ords]
  _A = bundleinds(A, [iA, jA], [:__iA__, :__jA__])
  TrA = mapslices(tr, _A.array; dims=getindex.(Ref(_A.tag_ord), [:__iA__, :__jA__]))
  mktensor(reshape(TrA, remaining_dims), remaining_tags)
end

"""[`iA`---`A`---`jA`] == [`iA`---`U`---(`iU`==`iU`)---`S`---(`iV`==`iV`)---`V`---`jA`]"""
function svd(A::Tensor, iA::Vector{Symbol},
  iU::Symbol, iV::Symbol; maxdim=nothing)::Tuple{Tensor,Tensor,Tensor}

  jA = setdiff(A.ord_tag, iA)
  old_dims = Tuple[]
  _A = bundleinds(A, [iA, jA], [:__iA__, :__jA__], old_dims)

  U, S, V = LinearAlgebra.svd(_A.array)

  # truncate
  actual_bonddim = length(S)
  if !isnothing(maxdim)
    actual_bonddim = min(actual_bonddim, maxdim)
  end
  U = mktensor(U[:, 1:actual_bonddim], [:__iA__, iU])
  S = mktensor(Diagonal(S[1:actual_bonddim]), [iU, iV])
  V = mktensor((V')[1:actual_bonddim, :], [iV, :__jA__])

  U = ravelinds(U, [:__iA__], [iA], old_dims[1:1])
  V = ravelinds(V, [:__jA__], [jA], old_dims[2:2])

  U, S, V
end

"""[`iA`---`A`---`jA`] == [`iA`---`U`---(`iU`==`iV`)---`V`---`jA`]"""
function halve(A::Tensor, iA::Vector{Symbol},
  iU::Symbol, iV::Symbol; maxdim=nothing)::Tuple{Tensor,Tensor}

  U, S, V = svd(A, iA, iV, iU; maxdim)
  S.array .= sqrt.(S.array)

  U = contract(U, S)
  V = contract(V, S)

  U, V
end

function conj!(A::Tensor)::Tensor
  conj!(A.array)
  A
end

function eigen(A::Tensor, iA::Vector{Symbol}, iU::Symbol; maxdim)

  jA = setdiff(A.ord_tag, iA)
  old_dims = Tuple[]
  _A = bundleinds(A, [iA, jA], [:__iA__, :__jA__], old_dims)

  vals, vecs = LinearAlgebra.eigen(_A.array)
  # sort!(vals; by=abs, rev=true)
  reverse!(vals)

  # truncate
  actual_bonddim = length(vals)
  if !isnothing(maxdim)
    actual_bonddim = min(actual_bonddim, maxdim)
  end

  U = mktensor(vecs[:, end:-1:end+1-actual_bonddim], [:__iA__, iU])
  U = ravelinds(U, [:__iA__], [iA], old_dims[1:1])

  vals[1:actual_bonddim], U
end

function delta(dims::Vector{Int}, tags::Vector{Symbol})::Tensor
  @assert length(dims) == length(tags)

  mktensor(map(t -> Int(all(==(t[1]), t)),
      Iterators.product(Base.OneTo.(dims)...)), tags)
end


return