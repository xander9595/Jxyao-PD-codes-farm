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
        tmprad = norm(coords[:,i])
        #if tmprad > 0.005
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

                    #compute local shear strain
                    X = zeros(1,3)
                    U = zeros(1,2)
                    for k in families[i,:]
                        if k !=0 && norm(coords[:,k]+Disp[:,k]-Disp[:,i]-coords[:,i]) < 1.5*p.dx
                            X = vcat(X, [coords[:,k]' 1 ])
                            U = vcat(U, Disp[:,k]')
                        end
                    end

                    nij = coords[:,families[i,j]] + Disp[:,families[i,j]]-coords[:,i] - Disp[:,i]

                    nij = nij / nlength
                    X = X[2:end,:]
                    U = U[2:end,:]
                    rij = zeros(2)
                    if det(X' * X) != 0
                        T = inv(X' * X)*X'
                        M = T * U
                        tmp_ma1 = [[1 0 0] ; [0 0 0] ; [0 0.5 0]]
                        tmp_ma2 = [[0 0 0] ; [0 1 0] ; [0.5 0 0]]
                        tmp_ma3 = [[(nij[1]-nij[1]^3) (-nij[1]*nij[2]^2) (nij[2]-2*nij[2]*nij[1]^2)];
                                    [(-nij[2]*nij[1]^2) (nij[2]-nij[2]^3) (nij[1] - 2*nij[1]*nij[2]^2)]]
                        eij = tmp_ma1 * M[:,1] + tmp_ma2 * M[:,2]
                        rij = tmp_ma3 * eij
                    end


                    if fail[i,j] == 1
                        #Calculation of the peridynamic force in x and y directions
                        #acting on a material point i due to a material point j
                        dforce1 = p.bc * (nlength - idist)/idist*p.vol*fac* (coords[1,families[i,j]]+Disp[1,families[i,j]]-coords[1,i]-Disp[1,i])/nlength
                        dforce2 = p.bc * (nlength - idist)/idist*p.vol*fac* (coords[2,families[i,j]]+Disp[2,families[i,j]]-coords[2,i]-Disp[2,i])/nlength

                        dforce1 += p.bk * rij[1] * p.vol * fac
                        dforce2 += p.bk * rij[2] * p.vol * fac
                    else
                        dforce1 = 0
                        dforce2 = 0
                    end

                    pforce[1,i] = pforce[1,i] + dforce1
                    pforce[2,i] = pforce[2,i] + dforce2

                    #Definition of a no-fail zone
                    #=
                    if abs((nlength - idist)/idist) > p.scr0 # && abs(norm(rij)) > 0.003
                        if abs(coords[2,i]) < p.len_y/4
                            fail[i,j] = 0
                        end
                    end
                    =#

                    dmgpar1 = dmgpar1 + fail[i,j] *p.vol * fac
                    dmgpar2 = dmgpar2 + p.vol*fac
                end
            end
            dmg[i,1] =1- dmgpar1/dmgpar2
        #end
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
