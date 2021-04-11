using SciPy
using Distributed
@everywhere using SharedArrays
@everywhere using LinearAlgebra
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

@everywhere function bondforce_kernal!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, p, irange, jrange)
    for i in irange
        dmgpar1 = 0.0
        dmgpar2 = 0.0
        pforce[1, i] = 0.0
        pforce[2, i] = 0.0
        for j in jrange
            if families[i,j] != 0
                idist = sqrt((coords[1,i]-coords[1,families[i,j]])^2+(coords[2,i]-coords[2,families[i,j]])^2)
                nlength = sqrt((coords[1,i]+Disp[1,i]-Disp[1,families[i,j]]-coords[1,families[i,j]])^2+(coords[2,i]+Disp[2,i]-Disp[2,families[i,j]]-coords[2,families[i,j]])^2)
                #Volume correction
                if idist< (p.delta-p.radij)
                    fac = 1.0
                elseif idist < (p.delta+p.radij)
                    fac = (p.delta+p.radij - idist)/p.dx
                else
                    fac = 0.0
                end
                if abs(coords[2,families[i,j]]-coords[2,i]) <= p.dx/10
                    theta = 0.0
                elseif abs(coords[1,families[i,j]]-coords[1,i]) <= p.dx/10
                    theta = 90.0*pi/180.0
                else
                    theta = atan(abs(coords[2,families[i,j]]-coords[2,i])/abs(coords[1,families[i,j]]-coords[1,i]))
                end
                #Determination of the surface correction between two material points
                scx = (fncst[i,1] + fncst[families[i,j],1]) / 2.0
                scy = (fncst[i,2] + fncst[families[i,j],2]) / 2.0
                scr = 1.0 /(((cos(theta))^2 / (scx)^2) + ((sin(theta))^2 / (scy)^2))
                scr = sqrt(scr)

                if fail[i,j] == 1
                    #Calculation of the peridynamic force in x and y directions
                    #acting on a material point i due to a material point j
                    dforce1 = p.bc * (nlength - idist)/idist*p.vol*scr*fac* (coords[1,families[i,j]]+Disp[1,families[i,j]]-coords[1,i]-Disp[1,i])/nlength
                    dforce2 = p.bc * (nlength - idist)/idist*p.vol*scr*fac* (coords[2,families[i,j]]+Disp[2,families[i,j]]-coords[2,i]-Disp[2,i])/nlength
                else
                    dforce1 = 0
                    dforce2 = 0
                end

                pforce[1,i] = pforce[1,i] + dforce1
                pforce[2,i] = pforce[2,i] + dforce2
                #Definition of a no-fail zone
                if abs((nlength - idist)/idist) >p.scr0
                    if abs(coords[2,i]) < p.len_y/4
                        fail[i,j] = 0
                    end
                end
                dmgpar1 = dmgpar1 + fail[i,j] *p.vol * fac
                dmgpar2 = dmgpar2 + p.vol*fac
            end
        end
        dmg[i,1] =1- dmgpar1/dmgpar2
    end
end

@everywhere bondforce_kernal_shared!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, p) =
                bondforce_kernal!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, p, myrange(families)...)


@everywhere function bondforce_shared!(dmg, pforce, bforce, families, coords, Disp, fncst, fail, p)
    @sync begin
    for i in procs(families)
        @async remotecall_wait(bondforce_kernal_shared!, i, dmg, pforce, bforce, families, coords, Disp, fncst, fail, p)
    end
end
end
