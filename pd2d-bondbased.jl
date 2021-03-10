
include("msh.jl")
using LinearAlgebra
using WriteVTK



const ndivx = 500
const ndivy = 500
const width_x = 0.05
const len_y = 0.05
const dx = width_x/ndivx
const nbnd = 3
const totnode = ndivx*(ndivy+2*nbnd)
const maxfam = 100
@time m = msh.Mesh(ndivx,ndivy,width_x,len_y)
coords = m.coords
totint = m.totint
totbottom = m.totbottom
tottop = m.tottop
pointfam = m.pointfam
numfam = m.numfam
nodefam = m.nodefam

fail = ones(Int8, totnode, maxfam)


for i = 1:totnode
    for j =  1:numfam[i,1]
        cnode = nodefam[pointfam[i,1]+j-1,1]
        if coords[2, cnode] > 0 && coords[2, i] < 0
            if abs(coords[1,i]) <= len_y/6
                fail[i,j]=0
            elseif abs(coords[1,cnode]) <= len_y/6
                fail[i,j]=0
            end
        elseif coords[2, cnode] < 0 && coords[2, i] > 0
            if abs(coords[1,i]) <= len_y/6
                fail[i,j]=0
            elseif abs(coords[1,cnode]) <= len_y/6
                fail[i,j]=0
            end
        end
    end
end

stendens = zeros(totnode, 2)
const delta = 3.015*dx
const radij = dx/2
const thick = dx
const area = dx^2
const vol = dx^3
const emod = 192.0e10
const bc = 9*emod/(pi*thick*delta^3) #bc: Bond constant
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
    for j = 1:numfam[i,1]
        cnode = nodefam[pointfam[i,1]+j-1,1]
        idist = norm(coords[:,cnode]-coords[:,i],2)
        nlength = norm(coords[:,cnode]+disp[:,cnode]-coords[:, i]-disp[:,i],2)
        if idist< (delta-radij)
            fac = 1.0
        elseif idist< (delta+radij)
            fac = (delta+radij - idist)/dx
        else
            fac = 0.0
        end

        stendens[i,1] = stendens[i,1] + 0.5*0.5*bc * (((nlength - idist)/idist)^2)*idist*vol*fac
    end
    fncst[i,1]= sedload1 / stendens[i,1]
end

#loading 2
disp_2 = zeros(2, totnode)
disp_2[2,:] = 0.001*coords[2,:]

for i = 1:totnode
    for j = 1:numfam[i,1]
        cnode = nodefam[pointfam[i,1]+j-1,1]
        idist = norm(coords[:,cnode]-coords[:,i],2)
        nlength = norm(coords[:,cnode]+disp_2[:,cnode]-coords[:, i]-disp_2[:,i],2)
        if idist< (delta-radij)
            fac = 1.0
        elseif idist < (delta+radij)
            fac = (delta+radij - idist)/dx
        else
            fac = 0.0
        end
        stendens[i,2] = stendens[i,2] + 0.5*0.5*bc * (((nlength - idist)/idist)^2)*idist*vol*fac
    end
    fncst[i,2]= sedload2 / stendens[i,2]
end

Vel = zeros(2 , totnode)
Disp = zeros(2, totnode)
acc = zeros(2, totnode)
pforce = zeros(2, totnode)
dmg = zeros(totnode,1)
bforce = zeros(2, totnode)
const dens = 8000.0
const nt = 1250
endtime = zeros(nt,1)
const dt = 0.8 * sqrt(2.0*dens*dx/(pi*(delta^2)*dx*bc))
const scr0 = 0.04472  #scr0: Critical stretch

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
    for i = 1:totnode
        dmgpar1 = 0.0
        dmgpar2 = 0.0
        pforce[1, i] = 0.0
        pforce[2, i] = 0.0
        for j = 1:numfam[i,1]
            cnode = nodefam[pointfam[i,1]+j-1,1]
            idist = sqrt((coords[1,i]-coords[1,cnode])^2+(coords[2,i]-coords[2,cnode])^2)
            nlength = sqrt((coords[1,i]+Disp[1,i]-Disp[1,cnode]-coords[1,cnode])^2+(coords[2,i]+Disp[2,i]-Disp[2,cnode]-coords[2,cnode])^2)
            #Volume correction
            if idist< (delta-radij)
                fac = 1.0
            elseif idist < (delta+radij)
                fac = (delta+radij - idist)/dx
            else
                fac = 0.0
            end
            if abs(coords[2,cnode]-coords[2,i]) <= dx/10
                theta = 0.0
            elseif abs(coords[1,cnode]-coords[1,i]) <= dx/10
                theta = 90.0*pi/180.0
            else
                theta = atan(abs(coords[2,cnode]-coords[2,i])/abs(coords[1,cnode]-coords[1,i]))
            end
            #Determination of the surface correction between two material points
            scx = (fncst[i,1] + fncst[cnode,1]) / 2.0
            scy = (fncst[i,2] + fncst[cnode,2]) / 2.0
            scr = 1.0 /(((cos(theta))^2 / (scx)^2) + ((sin(theta))^2 / (scy)^2))
            scr = sqrt(scr)

            if fail[i,j] == 1
                #Calculation of the peridynamic force in x and y directions
                #acting on a material point i due to a material point j
                dforce1 = bc * (nlength - idist)/idist*vol*scr*fac* (coords[1,cnode]+Disp[1,cnode]-coords[1,i]-Disp[1,i])/nlength
                dforce2 = bc * (nlength - idist)/idist*vol*scr*fac* (coords[2,cnode]+Disp[2,cnode]-coords[2,i]-Disp[2,i])/nlength
            else
                dforce1 = 0
                dforce2 = 0
            end

            pforce[1,i] = pforce[1,i] + dforce1
            pforce[2,i] = pforce[2,i] + dforce2
            #Definition of a no-fail zone
            if abs((nlength - idist)/idist) >scr0
                if abs(coords[2,i]) < len_y/4
                    fail[i,j] = 0
                end
            end
            dmgpar1 = dmgpar1 + fail[i,j] * vol * fac
            dmgpar2 = dmgpar2 + vol*fac
        end
        dmg[i,1] =1- dmgpar1/dmgpar2
    end

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

