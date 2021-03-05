module msh

using LinearAlgebra

struct Mesh
    coords::Array{Float64,2}
    totint::Int32
    totbottom::Int32
    tottop::Int32
    pointfam::Array{Int64,2}
    numfam::Array{Int64,2}
    nodefam::Array{Int64,2}

    function Mesh(n1, n2, x, y)
        @assert (n1 > 0 && n2 > 0) "Mesh arguments must be positive"
        coords, totint, totbottom,tottop = createCoords(n1, n2, x, y)
        pointfam, numfam, nodefam = createFam(coords, n1, n2, x, y)
        new(coords, totint, totbottom, tottop, pointfam, numfam, nodefam)
    end
end

function createCoords(n1, n2, x, y)

    #Fill coords array
    nbnd = 3
    coords = zeros(2, n1 * (n2 + 2 * nbnd))
    dx = x / n1
    cnt = 0

    for i = 1:n2
        for j = 1:n1
            cnt = cnt + 1
            coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (-y / 2) + dx / 2 + (i - 1) * dx]

        end
    end
    totint = cnt
    #boundary reaion - bottom
    for i = 1:nbnd
        for j = 1:n1
            cnt = cnt + 1
            coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (-y / 2) - dx / 2- (i - 1) * dx]
        end
    end

    totbottom = cnt
    #boundary reaion - top
    for i = 1:nbnd
        for j = 1:n1
            cnt = cnt + 1
            coords[1:2, cnt] =
                [(-x / 2) + dx / 2 + (j - 1) * dx, (y / 2) + dx / 2 + (i - 1) * dx]
        end
    end
    tottop = cnt

    r = dx / (4 * n1) * rand(size(coords)[1], size(coords)[2])
    coords = coords + r  #"random" pertubation of grid
    return coords, totint, totbottom, tottop
end

function createFam(coords, n1, n2, x, y)
    nbnd = 3
    totnode = n1 * (n2 + 2 * nbnd)
    dx = x/n1
    delta = 3.015*dx
    pointfam = zeros(Int, totnode, 1)
    numfam = zeros(Int, totnode, 1)
    nodefam = zeros(Int, 100*totnode, 1)
    for i = 1:totnode
        if i == 1
            pointfam[i,1] = 1
        else
            pointfam[i,1] = pointfam[i-1,1] + numfam[i-1,1]
        end
        for j= 1:totnode
            if i != j
                idist = sqrt((coords[1,i]-coords[1,j])^2+(coords[2,i]-coords[2,j])^2)
                if idist <= delta
                    numfam[i,1] = numfam[i,1] + 1
                    nodefam[pointfam[i,1]+numfam[i,1]-1, 1] = j
                end
            end
        end
    end

    return pointfam, numfam, nodefam
end

end
