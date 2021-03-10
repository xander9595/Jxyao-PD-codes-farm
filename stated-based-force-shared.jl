using SciPy
using BenchmarkTools
using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using SharedArrays
@everywhere using LinearAlgebra

function createCoords(n1::Int, n2::Int, x::Int, y::Int)

    #Fill coords array
    nbnd = 3
    coords = zeros(2, n1 * (n2 + 2 * nbnd))
    dx = x / n1
    cnt = 0

    for i = 1:n2
        for j = 1:n1
            cnt = cnt + 1
            @inbounds coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (-y / 2) + dx / 2 + (i - 1) * dx]

        end
    end
    totint = cnt
    #boundary reaion - bottom
    for i = 1:nbnd
        for j = 1:n1
            cnt = cnt + 1
            @inbounds coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (-y / 2) - dx / 2- (i - 1) * dx]
        end
    end

    totbottom = cnt
    #boundary reaion - top
    for i = 1:nbnd
        for j = 1:n1
            cnt = cnt + 1
            @inbounds coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (y / 2) + dx / 2 + (i - 1) * dx]
        end
    end
    tottop = cnt

    r = dx / (4 * n1) * rand(size(coords)[1], size(coords)[2])
    coords = coords + r  #"random" pertubation of grid
    return coords
end

function compute_internal_force(coords::Array{Float64,2}, disp::Array{Float64,2}, families::Array{Int64,2}, horizon::Float64, mask::BitArray{2} , youngs_modulus = 210e9, poisson_ratio = 0.3)
    #compute the deformed positions of nodes
    def = coords + disp
    # compute deformation state
    tot_nodes,max_num_fam =size(families)
    def_state = zeros(2,tot_nodes, max_num_fam)
    def_mag_state = zeros(tot_nodes, max_num_fam)
    ref_mag_state = zeros(tot_nodes, max_num_fam)
    exten_state = zeros(tot_nodes, max_num_fam)
    def_unit_state = zeros(2,tot_nodes, max_num_fam)
    for i = 1:tot_nodes
        for j = 1:max_num_fam
            if families[i,j] != 0
                def_state[:,i,j] = coords[:,i] + disp[:,i] - coords[:,families[i,j]] - disp[:,families[i,j]]
                def_mag_state[i,j] = norm(def_state[:,i,j])
                ref_mag_state[i,j] = norm((coords[:,i] - coords[:,families[i,j]]))
                exten_state[i,j] = (def_mag_state[i,j] - ref_mag_state[i,j])/ ref_mag_state[i,j]
                def_unit_state[:,i,j] = def_state[:,i,j] / def_mag_state[i,j]
            end
        end
    end

    return def_state,def_unit_state
end

#Distributed computing based on SharedArray

#splits the work/分发任务
@everywhere function myrange(data)
    idx = indexpids(data)
    if idx == 0 # This worker is not assigned a piece
        return 1:0, 1:0
    end
    nchunks = length(procs(data))
    splits = [round(Int, s) for s in range(0, stop=size(data, 1), length=nchunks+1)]
    return splits[idx]+1:splits[idx+1], 1:size(data,2)
end

#kernal function/ 核心函数，也就是要并行计算的任务
@everywhere function cif1_kernal!(def_state,def_mag_state,ref_mag_state, exten_state,def_unit_state, coords, disp, families, irange, jrange)
    for i in irange
        for j in jrange
            if families[i,j] != 0
                def_state[:,i,j] = coords[:,i] + disp[:,i] - coords[:,families[i,j]] - disp[:,families[i,j]]
                def_mag_state[i,j] = norm(def_state[:,i,j])
                ref_mag_state[i,j] = norm((coords[:,i] - coords[:,families[i,j]]))
                exten_state[i,j] = (def_mag_state[i,j] - ref_mag_state[i,j])/ ref_mag_state[i,j]
                def_unit_state[:,i,j] = def_state[:,i,j] / def_mag_state[i,j]
            end
        end
    end
end
@everywhere cif1_kernal_shared!(def_state,def_mag_state,ref_mag_state, exten_state,def_unit_state, coords, disp, families) =
                 cif1_kernal!(def_state,def_mag_state,ref_mag_state, exten_state,def_unit_state,  coords, disp, families, myrange(families)...)


@everywhere function cif2_kernal!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families, irange, jrange )
    for i in irange
        for j in jrange
            if families[i,j] != 0
                iso_exten_state[i,j] = dilatation[i,1] * ref_mag_state[i,j] /3
                dev_exten_state[i,j] = exten_state[i,j] - iso_exten_state[i,j]
                alpha = 15 * shear_modulus / m[i,1]
                iso_force_state[i,j] = (-3.0 * pressure[i,1] / weighted_volume[i,1] * influence_state[i,j] * ref_mag_state[i,j])
                dev_force_state[i,j] = alpha * influence_state[i,j] * dev_exten_state[i,j]
            end
        end
    end
end
@everywhere cif2_kernal_shared!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families) =
                cif2_kernal!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families, myrange(families)...)

@everywhere function cif3_kernal!(force_state_x, force_state_y, force_x, force_y,families, irange, krange )
    for i in irange
        for j in families[i,:]
            for k in krange
                if j>0 && families[j,k] == i
                    force_x[i,1] = force_x[i,1] - force_state_x[j,k]
                    force_y[i,1] = force_y[i,1] - force_state_y[j,k]
                end
            end
        end
    end
    return force_x
end

@everywhere cif3_kernal_shared!(force_state_x, force_state_y, force_x, force_y,families) =
                cif3_kernal!(force_state_x, force_state_y, force_x, force_y,families, myrange(families)...)

#remotecall/调用分发的任务
function cif1_shared!(def_state,def_mag_state,ref_mag_state, exten_state,def_unit_state, coords, disp, families)
    @sync begin
        for p in procs(families)
            @async remotecall_wait(cif1_kernal_shared!, p, def_state,def_mag_state,ref_mag_state, exten_state,def_unit_state, coords, disp, families )
        end
    end
end

function cif2_shared!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families)
    @sync begin
        for p in procs(families)
            @async remotecall_wait(cif2_kernal_shared!, p, iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families )
        end
    end
end

function cif3_shared!(force_state_x, force_state_y, force_x, force_y,families)
    @sync begin
        for p in procs(families)
            @async remotecall_wait(cif3_kernal_shared!, p, force_state_x, force_state_y, force_x, force_y,families )
        end
    end
end

function compute_inrenal_force_shared(coords::Array{Float64,2}, disp::Array{Float64,2}, families::Array{Int64,2}, horizon::Float64, mask::BitArray{2} , youngs_modulus = 210e9, poisson_ratio = 0.3)
    def = coords + disp
    # compute deformation state
    tot_nodes,max_num_fam =size(families)
    def_state = SharedArray(zeros(2,tot_nodes, max_num_fam))
    def_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
    ref_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
    exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
    def_unit_state = SharedArray(zeros(2,tot_nodes, max_num_fam))
    families = SharedArray(families)
    coords = SharedArray(coords)
    disp = SharedArray(disp)
    cif1_shared!(def_state, def_mag_state,ref_mag_state, exten_state,def_unit_state, coords, disp, families)

    influence_state = SharedArray(zeros(tot_nodes, max_num_fam))
    influence_state[mask] = 1 .- ref_mag_state[mask]/horizon
    influence_state[mask] = @. ifelse(exten_state[mask] >= 0.005, 0, influence_state[mask])

    #Compute weighted volume
    weighted_volume = zeros(tot_nodes, max_num_fam)
    weighted_volume[mask] = influence_state[mask] .* ref_mag_state[mask] .* ref_mag_state[mask]
    m = SharedArray(sum(weighted_volume, dims = 2))

    #Compute dilatation
    dilatation = SharedArray(zeros(tot_nodes, 1))
    for i = 1:tot_nodes
        if m[i,1] != 0
         dilatation[i,1]= sum((2/m[i, 1])influence_state[i,:] .* ref_mag_state[i,:] .* exten_state[i,:])
        end
    end

    # Linear peridynamic solid model
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    #Compute the pressure
    pressure = SharedArray(-bulk_modulus * dilatation)
    #Compute the deviatoric extension state
    iso_exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
    dev_exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
    iso_force_state = SharedArray(zeros(tot_nodes, max_num_fam))
    dev_force_state = SharedArray(zeros(tot_nodes, max_num_fam))

    cif2_shared!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state, dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure, weighted_volume, influence_state, families)

    #compute the force on every points/计算每个点受到的合外力
    scalar_force_state = iso_force_state .+ dev_force_state
    force_state_x = SharedArray(scalar_force_state .* def_unit_state[1,:,:])
    force_state_y = SharedArray(scalar_force_state .* def_unit_state[2,:,:])
    force_x = SharedArray(sum(force_state_x, dims = 2))
    force_y = SharedArray(sum(force_state_y, dims = 2))
    cif3_shared!(force_state_x, force_state_y, force_x, force_y,families)

    return force_x
end



const ndivx = 100
const ndivy = 100
const width_x = 10
const len_y = 10

Coords = createCoords(ndivx,ndivy,width_x,len_y)

nodes = Coords'
tree = SciPy.spatial.cKDTree(nodes)

a, families = tree.query(nodes, k=100, eps=0.0, p=2, distance_upper_bound=3.015*width_x/ndivx)
families=@. ifelse(families ==  tree.n, -1, families)
mask = families[:,:] .!= -1
max_family_length = maximum(sum(Int.(families[:,:] .!= -1), dims = 2))
mask = mask[:,2:max_family_length ]
families = families[:, 2:max_family_length ] .+ 1
Disp = 0.0001*rand(size(Coords)[1] , size(Coords)[2])

@benchmark Def_state, Def_unit_state = compute_internal_force(Coords , Disp , families, 3.015*width_x/ndivx, mask)

@benchmark aaa = compute_inrenal_force_shared(Coords , Disp , families, 3.015*width_x/ndivx, mask)
bbb = compute_inrenal_force_shared(Coords , Disp , families, 3.015*width_x/ndivx, mask)
families = SharedArray(families)
Coords = SharedArray(Coords)
Disp = SharedArray(Disp)
tot_nodes,max_num_fam =size(families)
def_state = SharedArray(zeros(2,tot_nodes, max_num_fam))
def_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
ref_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
def_unit_state = SharedArray(zeros(2,tot_nodes, max_num_fam))

aaa = fetch(@spawnat 2 myrange(families))

tot_nodes,max_num_fam =size(families)
force_state_x = SharedArray(zeros(tot_nodes, max_num_fam))
force_state_y = SharedArray(zeros(tot_nodes, max_num_fam))
force_x = SharedArray(sum(force_state_x, dims = 2))
force_y = SharedArray(sum(force_state_y, dims = 2))
