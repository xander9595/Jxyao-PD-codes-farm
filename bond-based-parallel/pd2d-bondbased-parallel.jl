using WriteVTK
using SciPy
using Distributed
if nworkers() == 1
    addprocs(15)
end
@everywhere using SharedArrays
@everywhere using LinearAlgebra
@everywhere using Parameters
include("msh.jl")
include("createCoords.jl")
include("bond_force.jl")

const ndivx = 500
const ndivy =500
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
const emod = 192.0e10
const bc = 9*emod/(pi*thick*delta^3) #bc: Bond constant


@everywhere @with_kw struct para
    ndivx::Int = 250
    ndivy::Int = 250
    width_x::Float64 = 0.05
    len_y::Float64 = 0.05
    dx::Float64 = width_x/ndivx
    nbnd::Int = 3
    totnode::Int = ndivx*(ndivy+2*nbnd)
    delta::Float64 = 3.015*dx
    area::Float64 =dx^2
    vol::Float64 = dx^3
    emod::Float64 = 192.0e10
    thick::Float64 = dx
    radij::Float64 = dx/2
    bc::Float64 = 9*emod/(pi*thick*delta^3)
    dens::Float64 = 8000.0
    scr0::Float64 = 0.04472
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
            stendens[i,2] = stendens[i,2] + 0.5*0.5*bc * (((nlength - idist)/idist)^2)*idist*vol*fac
        end
    end
    fncst[i,2]= sedload2 / stendens[i,2]
end

Vel = zeros(2 , totnode)
Disp = SharedArray(zeros(2, totnode))
acc = zeros(2, totnode)
pforce = SharedArray(zeros(2, totnode))
dmg =SharedArray(zeros(totnode,1))
bforce = SharedArray(zeros(2, totnode))
coords = SharedArray(coords)
fncst = SharedArray(fncst)
const nt = 1251
endtime = zeros(nt,1)
const dt = 0.8 * sqrt(2.0*pa.dens*dx/(pi*(delta^2)*dx*bc))
dens = 8000


# paraview visualization Initialise
pvd = paraview_collection("pd2d", append=true)
for tt= 1: nt
    println("tt=$(tt)")
    ctime = tt*dt
    #Application of boundary conditions at the top and bottom edges
    for i = (totint+1):totbottom
        Vel[2,i] = -50.0
        Disp[2, i] = -50.0*tt*dt
    end

    for i = (totbottom+1):tottop
        Vel[2,i] = 50.0
        Disp[2, i] = 50.0*tt*dt
    end

    bondforce_shared!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, pa)



    acc[:,1:totint] = (pforce[:,1:totint]+bforce[:,1:totint]) / dens
    Vel[:,1:totint] = Vel[:,1:totint] +acc[:,1:totint]*dt
    Disp[:,1:totint] = Disp[:,1:totint] + Vel[:,1:totint] *dt

    acc[1,totint+1:tottop] = (pforce[1,totint+1 : tottop] + bforce[1,totint+1:tottop])/dens
    Vel[1,totint+1:tottop] = Vel[1,totint+1:tottop] + acc[1,totint+1:tottop] *dt
    Disp[1,totint+1:tottop] = Disp[1,totint+1:tottop] + Vel[1,totint+1:tottop] *dt
    endtime[tt,1] = ctime


    #plots the data
    if mod(tt-1,50) == 0
        Damage = zeros(ndivx,ndivy)
        DISP_Y = zeros(ndivx,ndivy)
        xyz = zeros( 2, ndivx, ndivy)
        cntnum = 1
        for i = 1:ndivy
            for j= 1:ndivx
                xyz[:,j,i]=coords[:,cntnum]+Disp[:,cntnum]
                Damage[j,i] = dmg[cntnum, 1]
                DISP_Y[j,i] = Disp[2,cntnum]
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
