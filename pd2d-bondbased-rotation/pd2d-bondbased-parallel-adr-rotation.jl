using WriteVTK
using SciPy
using Distributed
using DelimitedFiles
if nworkers() == 1
    addprocs(31)
end
@everywhere using SharedArrays
@everywhere using LinearAlgebra
@everywhere using Parameters
include("createCoords.jl")
include("bond_force.jl")

const ndivx = 100
const ndivy = 100
const width_x = 0.05
const len_y = 0.05
const dx = width_x/ndivx
const nbnd = 3
const totnode = ndivx*(ndivy+2*nbnd)
const maxfam = 100

const delta = 3.015*dx
const radij = dx/2
const thick = dx
const area = dx^2
const vol = dx^3
const emod = 192.0e9
const bc = 9*emod/(pi*thick*delta^3) #bc: Bond constant


@everywhere @with_kw struct para
    ndivx::Int = 100
    ndivy::Int = 100
    width_x::Float64 = 0.05
    len_y::Float64 = 0.05
    dx::Float64 = width_x/ndivx
    nbnd::Int = 3
    totnode::Int = ndivx*(ndivy+2*nbnd)
    delta::Float64 = 3.015*dx
    area::Float64 =dx^2
    vol::Float64 = dx^3
    emod::Float64 = 40.0e9
    thick::Float64 = dx
    radij::Float64 = dx/2
    GIc::Float64 = 160
    GIIc::Float64 = 1634
    possion::Float64 = 1/4
    bc::Float64 = 7.5*emod/(pi*thick*delta^3)
    bk::Float64 = 2.5*emod/(pi*thick*delta^3)
    dens::Float64 = 2500.0
    scr0::Float64 = sqrt(2*GIc/(bc*pi*thick*delta^4))
    rcr0::Float64 = sqrt(240*GIIc/(bc*pi*thick*delta^4))
end

pa= para()


coords, totint, totbottom, tottop = createCoords(ndivx,ndivy,width_x,len_y)


nodes = coords'
tree = SciPy.spatial.cKDTree(nodes)
_, families = tree.query(nodes, k=100, eps=0.0, p=2, distance_upper_bound= delta)
families=@. ifelse(families ==  tree.n, -1, families)
mask = families[:,:] .!= -1
max_family_length = maximum(sum(mask, dims = 2))
mask = mask[:,2:max_family_length ]
# Because in python the index start from 0, julia start from 1
families = families[:, 2:max_family_length ] .+ 1
_,max_num_fam =size(families)
families = SharedArray(families)


fail =SharedArray(ones(Int, totnode, max_num_fam))

#=
#Inserts a crack
for i = 1:totnode
    for j in 1:max_num_fam
        if families[i,j]!= 0 && coords[2,families[i,j]] > 0 && coords[2,i] < 0
            if abs(coords[1,i]) <=  width_x/6 || abs(coords[1,families[i,j]]) <=  width_x/6
                fail[i,j] = 0
            end
        elseif families[i,j] != 0 && coords[2,families[i,j]] < 0 && coords[2,i] > 0
            if abs(coords[1,i]) <=  width_x/6 || abs(coords[1,families[i,j]]) <=  width_x/6
                fail[i,j] = 0
            end
        end
    end
end
=#

stendens = zeros(totnode, 2)
fncst = zeros(totnode,2)
const sedload1 = 9.0 / 16.0 * emod * 1.0e-6
const sedload2 = 9.0 / 16.0 * emod * 1.0e-6
#    Calculation of surface correction factor in x direction
#    by finding the ratio of the analytical strain energy density value
#    to the strain energy density value obtained from PD Theory


#loading 1
disp=zeros(2,totnode)
disp[1,:] = 0.001 * coords[1,:]

for i = 1:totnode
    for j = 1:max_num_fam
        if families[i,j] != 0
            idist = norm(coords[:,families[i,j]]-coords[:,i],2)
            nlength = norm(coords[:,families[i,j]]+disp[:,families[i,j]]-coords[:, i]-disp[:,i],2)
            if idist< (delta-radij)
                fac = 1.0
            elseif idist< (delta+radij)
                fac = (delta+radij - idist)/dx
            else
                fac = 0.0
            end

            stendens[i,1] = stendens[i,1] + 0.5*0.5*bc * (((nlength - idist)/idist)^2)*idist*vol*fac
        end
    end
    fncst[i,1]= sedload1 / stendens[i,1]
end

#loading 2
disp_2 = zeros(2, totnode)
disp_2[2,:] = 0.001*coords[2,:]

for i = 1:totnode
    for j = 1:max_num_fam
        if families[i,j] != 0
            idist = norm(coords[:,families[i,j]]-coords[:,i],2)
            nlength = norm(coords[:,families[i,j]]+disp_2[:,families[i,j]]-coords[:, i]-disp_2[:,i],2)
            if idist< (delta-radij)
                fac = 1.0
            elseif idist < (delta+radij)
                fac = (delta+radij - idist)/dx
            else
                fac = 0.0
            end
            stendens[i,2] = stendens[i,2] + 0.5 * 0.5 * bc * (((nlength - idist)/idist)^2)*idist*vol*fac
        end
    end
    fncst[i,2]= sedload2 / stendens[i,2]
end

Vel = zeros(2 , totnode)
Vel_half = zeros(2 , totnode)
Vel_half_old = zeros(2 , totnode)
Disp = SharedArray(zeros(2, totnode))
acc = zeros(2, totnode)
pforce = SharedArray(zeros(2, totnode))
pforceold = SharedArray(zeros(2, totnode))
dmg =SharedArray(zeros(totnode,1))
bforce = SharedArray(zeros(2, totnode))
coords = SharedArray(coords)
fncst = SharedArray(fncst)
const nt = 1501
endtime = zeros(nt,1)
const dt = 1.0
dens = 8000

#Stable mass vector computation
lambda_ii = 0.25* dt * dt * (pi * (delta)^2 * thick) * bc / dx
mass_vector = ones(2, totnode)*lambda_ii



# paraview visualization Initialise
pvd = paraview_collection("pd2d", append=true)
for tt= 1: nt
    println("tt=$(tt)")

    ctime = tt*dt
    #Application of boundary conditions at the top and bottom edges
    for i = (totint+1):totbottom
        Vel[2,i] = -2.5e-7
        Disp[2, i] = -2.5e-7*tt*dt
    end

    for i = (totbottom+1):tottop
        Vel[2,i] = 2.5e-7
        Disp[2, i] = 2.5e-7*tt*dt
    end

    bondforce_shared!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, pa)

    #Adaptive dynamic relaxation
    cn = 0.0
    cn1 = 0.0
    cn2 = 0.0
    for i = 1:totint
        if Vel_half_old[1,i] != 0
            cn1 = cn1 - Disp[1,i] * Disp[1,i] * (pforce[1,i] / mass_vector[1,i] - pforceold[1,i] / mass_vector[1,i]) / (dt * Vel_half_old[1,i])
        end
        if Vel_half_old[2,i] != 0
            cn1 = cn1 - Disp[2,i] * Disp[2,i] * (pforce[2,i] / mass_vector[2,i] - pforceold[2,i] / mass_vector[2,i]) / (dt * Vel_half_old[2,i])
        end
        cn2 = cn2 + Disp[1,i]*Disp[1,i] + Disp[2,i]*Disp[2,i]
    end

    if cn2 != 0
        if (cn1/cn2) > 0.0
            cn = 2.0*sqrt(cn1/cn2)
        end
    end
    if cn > 2.0
        cn = 1.9
    end

    if tt == 1
        @. Vel_half[:,1:totint] = dt * (pforce[:,1:totint] + bforce[:,1:totint]) / (2.0 * mass_vector[:, 1:totint])
    else
        @. Vel_half[:,1:totint] =((2.0 - cn*dt)*Vel_half_old[:,1:totint] + 2.0*dt*(pforce[:,1:totint] + bforce[:,1:totint])/mass_vector[:, 1:totint])/(2.0 + cn*dt)
    end

    Vel[:,1:totint] = 0.5*(Vel_half_old[:,1:totint] + Vel_half[:,1:totint])
    Disp[:,1:totint] = Disp[:,1:totint] + Vel_half[:,1:totint] * dt


    Vel_half_old[:,1:totint] = Vel_half[:,1:totint]
    pforceold[:,1:totint] = pforce[:,1:totint]



    endtime[tt,1] = ctime


    #plots the data
    if mod(tt-1,50) == 0
        Damage = zeros(ndivx,ndivy)
        DISP_Y = zeros(ndivx,ndivy)
        DISP_X = zeros(ndivx,ndivy)
        xyz = zeros( 2, ndivx, ndivy)
        cntnum = 1
        for i = 1:ndivy
            for j= 1:ndivx
                xyz[:,j,i]=coords[:,cntnum]+Disp[:,cntnum]
                Damage[j,i] = dmg[cntnum, 1]
                DISP_X[j,i] = Disp[1,cntnum]
                DISP_Y[j,i] = Disp[2,cntnum]
                cntnum = cntnum +1
            end
        end
        vtk = vtk_grid("pd2d_$(tt)", xyz)
        vtk["Damage"] = Damage
        vtk["DISP_X"] = DISP_X
        vtk["DISP_Y"] = DISP_Y
        pvd[fld(tt,50) + 1] = vtk

        if tt == 1001
            writedlm("h_disp.csv", DISP_X[:, 50] , ',')
            writedlm("V_disp.csv", DISP_Y[50, :] , ',')
        end
    end


end
vtk_save(pvd)
