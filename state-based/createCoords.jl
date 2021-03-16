
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
    return coords, totint, totbottom, tottop
end
