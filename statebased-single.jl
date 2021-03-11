using SciPy
using LinearAlgebra
using BenchmarkTools

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

    influence_state = zeros(tot_nodes, max_num_fam)
    influence_state[mask] = 1 .- ref_mag_state[mask]/horizon
    influence_state[mask] = @. ifelse(exten_state[mask] >= 0.005, 0, influence_state[mask]) #bond break

    #Compute weighted volume
    weighted_volume = zeros(tot_nodes, max_num_fam)
    weighted_volume[mask] = influence_state[mask] .* ref_mag_state[mask] .* ref_mag_state[mask]
    m = sum(weighted_volume, dims = 2)

    #Compute dilatation
    dilatation = zeros(tot_nodes, 1)
    for i = 1:tot_nodes
        if m[i,1] != 0
         dilatation[i,1]= sum((2/m[i, 1])influence_state[i,:] .* ref_mag_state[i,:] .* exten_state[i,:])
        end
    end
    # Linear peridynamic solid model
    bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    #Compute the pressure
    pressure = -bulk_modulus * dilatation
    #Compute the deviatoric extension state
    iso_exten_state = zeros(tot_nodes, max_num_fam)
    dev_exten_state = zeros(tot_nodes, max_num_fam)
    iso_force_state = zeros(tot_nodes, max_num_fam)
    dev_force_state = zeros(tot_nodes, max_num_fam)
    for i = 1:tot_nodes
        for j = 1:max_num_fam
            if families[i,j] != 0
                iso_exten_state[i,j] = dilatation[i,1] * ref_mag_state[i,j] /3
                dev_exten_state[i,j] = exten_state[i,j] - iso_exten_state[i,j]
                alpha = 15 * shear_modulus / m[i,1]
                iso_force_state[i,j] = (-3.0 * pressure[i,1] / weighted_volume[i,1] * influence_state[i,j] * ref_mag_state[i,j])
                dev_force_state[i,j] = alpha * influence_state[i,j] * dev_exten_state[i,j]
            end
        end
    end

    #compute the force on every points/计算每个点受到的合外力
    scalar_force_state = iso_force_state .+ dev_force_state
    force_state_x = scalar_force_state .* def_unit_state[1,:,:]
    force_state_y = scalar_force_state .* def_unit_state[2,:,:]
    force_x = sum(force_state_x, dims = 2)
    force_y = sum(force_state_y, dims = 2)
    for i = 1:tot_nodes
        for j = 1:tot_nodes
            for k = 1:max_num_fam
                if families[j,k] == i
                    force_x[i,1] = force_x[i,1] - force_state_x[j,k]
                    force_y[i,1] = force_y[i,1] - force_state_y[j,k]
                end
            end
        end
    end


    return def_state, def_mag_state, def_unit_state, ref_mag_state, exten_state, influence_state , m, dilatation, scalar_force_state, force_state_x, force_x
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
Disp = 0.0001*zeros(size(Coords)[1] , size(Coords)[2])

@benchmark Def_state, Def_mag_state, Def_unit_state, Ref_mag_state, Exten_state, Influence_state, Weighted_volume, Dilatation, Scalar_force_state, Force_state_x, Force_x = compute_internal_force(Coords , Disp , families, 3.015*width_x/ndivx, mask)
sum(Influence_state,dims = 2)
