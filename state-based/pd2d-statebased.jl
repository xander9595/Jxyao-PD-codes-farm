using SciPy
using BenchmarkTools
using Distributed
using WriteVTK
if nworkers() == 1
    addprocs(15)
end
@everywhere using SharedArrays
@everywhere using LinearAlgebra
include("createCoords.jl")
include("compute_internal_force_shared.jl")

const ndivx = 100
const ndivy = 100
const width_x = 100
const len_y = 100
dx = width_x/ndivx
volume = dx^3
const horizon = 3.015*dx

youngs_modulus = 210e9
poisson_ratio = 0.3
density = 7800
MAX_ITER = 4000
time_step = 1e-5
bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio))
shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
const bc = 9*youngs_modulus/(pi*dx*horizon^3)
dt = 0.8 * sqrt(2.0*density*dx/(pi*(horizon^2)*dx*bc))

Coords, totint, totbottom, tottop = createCoords(ndivx,ndivy,width_x,len_y)
nodes = Coords'
#Find the neighbor points of each points
tree = SciPy.spatial.cKDTree(nodes)
_, families = tree.query(nodes, k=100, eps=0.0, p=2, distance_upper_bound= horizon)
families=@. ifelse(families ==  tree.n, -1, families)
mask = families[:,:] .!= -1
max_family_length = maximum(sum(mask, dims = 2))
mask = mask[:,2:max_family_length ]

# Because in python the index start from 0, julia start from 1
families = families[:, 2:max_family_length ] .+ 1

tot_nodes,max_num_fam =size(families)


Disp = zeros(size(Coords)[1] , size(Coords)[2])

#define PD state
def_state = SharedArray(zeros(2,tot_nodes, max_num_fam))
def_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
ref_mag_state = SharedArray(zeros(tot_nodes, max_num_fam))
exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
def_unit_state = SharedArray(zeros(2,tot_nodes, max_num_fam))
families = SharedArray(families)
Coords = SharedArray(Coords)
my_disp = SharedArray(zeros(2,tot_nodes))
my_vel = SharedArray(zeros(2,tot_nodes))
my_accel_old = zeros(2,tot_nodes)
my_accel = zeros(2,tot_nodes)
my_damage = zeros(tot_nodes,1)

cif1_shared!(def_state, def_mag_state,ref_mag_state, exten_state,def_unit_state, Coords, my_disp, families)

ref_influence_state = SharedArray(zeros(tot_nodes, max_num_fam))

ref_influence_state[mask] .= 1 ./ ref_mag_state[mask] #reference
def_influence_state = SharedArray(copy(ref_influence_state))
#insert a crack
for i = 1:tot_nodes
    for j in 1:max_num_fam
        if families[i,j]!= 0 && Coords[2,families[i,j]] > 0 && Coords[2,i] < 0
            if abs(Coords[1,i]) <=  width_x/6 || abs(Coords[1,families[i,j]]) <=  width_x/6
                def_influence_state[i,j] = 0
            end
        elseif families[i,j] != 0 && Coords[2,families[i,j]] < 0 && Coords[2,i] > 0
            if abs(Coords[1,i]) <=  width_x/6 || abs(Coords[1,families[i,j]]) <=  width_x/6
                def_influence_state[i,j] = 0
            end
        end
    end
end


weighted_volume = zeros(tot_nodes, max_num_fam)
m = SharedArray(zeros(tot_nodes,1)) # sum of weighted_volume


dilatation = SharedArray(zeros(tot_nodes, 1))
pressure = SharedArray(zeros(tot_nodes, 1))

iso_exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
dev_exten_state = SharedArray(zeros(tot_nodes, max_num_fam))
iso_force_state = SharedArray(zeros(tot_nodes, max_num_fam))
dev_force_state = SharedArray(zeros(tot_nodes, max_num_fam))
scalar_force_state = SharedArray(zeros(tot_nodes, max_num_fam))

force_state_x = SharedArray(zeros(tot_nodes, max_num_fam))
force_state_y = SharedArray(zeros(tot_nodes, max_num_fam))

force_x = SharedArray(zeros(tot_nodes,1))
force_y = SharedArray(zeros(tot_nodes,1))

# paraview visualization Initialise
pvd = paraview_collection("pd2d", append=true)

for tt = 1:300
    #println("tt=$(tt)")
    time = tt * time_step
    #bottom boundary condition
    my_vel[1,totint+1:totbottom] .= 0
    my_vel[2,totint+1:totbottom] .= -1
    my_disp[1,totint+1:totbottom] .= 0
    my_disp[2,totint+1:totbottom] .= -1*time
    #top boundary condition
    my_vel[1,totbottom+1:tottop] .= 0
    my_vel[2,totbottom+1:tottop] .= 1
    my_disp[1,totbottom+1:tottop] .= 0
    my_disp[2,totbottom+1:tottop] .= 1*time
    #Clear the internal force vectors
    force_x[:,:] .= 0
    force_y[:,:] .= 0
    #Compute the internal force
    cif1_shared!(def_state, def_mag_state,ref_mag_state, exten_state, def_unit_state, Coords, my_disp, families)
    #bond break--Apply a critical stretch damage model
    def_influence_state[mask] = @. ifelse(exten_state[mask] >= 0.03, 0, def_influence_state[mask])
    weighted_volume[mask] = def_influence_state[mask] .* ref_mag_state[mask] .* ref_mag_state[mask]
    m .= sum(weighted_volume, dims = 2) .* volume
    dilatation[:,:] .= 0
    dilatation_shared!(dilatation, m, def_influence_state, ref_mag_state, exten_state, families)
    pressure .= -bulk_modulus .* dilatation

    cif2_shared!(iso_exten_state, dev_exten_state, iso_force_state, dev_force_state,
                dilatation, ref_mag_state, exten_state, m, shear_modulus, pressure,
                weighted_volume, def_influence_state, families)

    #Integrate nodal forces
    scalar_force_state .= iso_force_state .+ dev_force_state
    force_state_x .= scalar_force_state .* def_unit_state[1,:,:]
    force_state_y .= scalar_force_state .* def_unit_state[2,:,:]
    force_x .= sum(force_state_x, dims = 2)
    force_y .= sum(force_state_y, dims = 2)
    #Subtract the force contribution from i nodes from j
    cif3_shared!(force_state_x, force_state_y, force_x, force_y,families)

    #Compute the nodal acceleration
    my_accel_old[:,:] .= my_accel[:,:]
    my_accel[1,:] .= force_x[:,1] ./ density
    my_accel[2,:] .= force_y[:,1] ./ density

    @. my_vel += 0.5*(my_accel_old + my_accel)*time_step
    @. my_disp += my_vel*time_step + 0.5*my_accel*time_step*time_step


    dmg_state = zeros(tot_nodes,max_num_fam)
    dmg_state[mask] .= def_influence_state[mask]./ref_influence_state[mask]
    my_damage = zeros(tot_nodes,1)
    my_damage .= 1.0 .- sum(dmg_state,dims = 2) ./ sum(mask,dims = 2)

    #plots the data
    if mod(tt-1,50) == 0
        Damage = zeros(ndivx,ndivy)
        DISP_Y = zeros(ndivx,ndivy)
        xyz = zeros( 2, ndivx, ndivy)
        cntnum = 1
        b = sum(my_damage)
        println("my_damage = $b")
        for i = 1:ndivy
            for j= 1:ndivx
                xyz[:,j,i]=Coords[:,cntnum]+my_disp[:,cntnum]
                Damage[j,i] = my_damage[cntnum, 1]
                DISP_Y[j,i] = my_disp[2,cntnum]
                cntnum = cntnum +1
            end
        end
        vtk = vtk_grid("pd2d_$(tt)", xyz)
        vtk["Damage"] = Damage
        vtk["DISP_Y"] = DISP_Y
        pvd[fld(tt,50) + 1] = vtk
    end

end
vtk_save(pvd)
