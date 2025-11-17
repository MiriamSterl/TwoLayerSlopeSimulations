using GeophysicalFlows, Printf
using FourierFlows: TwoDGrid, makefilter, rfft, irfft
using Statistics
using Random: seed!
using NPZ

function simulation(slope,spinupdays,rundays,field)

    println("Simulation with s = "*slope*", strong mu & U started")

    n = 256                             # 2D resolution = n²
    stepper = "FilteredRK4"             # timestepper
        dt = 200                        # timestep
    nsteps = (spinupdays+rundays)*86400/dt # total number of time-steps
    nsubs_output = 18                    # number of time-steps for saving output (nsteps must be a multiple of nsubs_output)

    L = 7e5                  # domain size
    μ = 3e-6                 # bottom drag
    β = 0                    # the y-gradient of planetary PV

    nlayers = 2              # number of layers
    f₀, g = 1e-4, 9.81       # Coriolis parameter and gravitational constant
    H = [1000, 4000]         # the rest depths of each layer
    ρ = [1027.6, 1028]       # the density of each layer
    gp = g*(ρ[2] - ρ[1])/ρ[2]
    b2 = -1
    b1 = b2 + gp
    b = [b1, b2]             # buoyancy
    s = parse(Float64,slope) # the bottom slope

    U = zeros(nlayers) # the imposed mean zonal flow in each layer
    U[1] = 0.2
    U[2] = 0.0

    x, y = gridpoints(TwoDGrid(nx=n, Lx=L, ny=n, Ly=L))

    dev = CPU()
    prob = MultiLayerQG.Problem(nlayers, dev;
                                nx=n, Lx=L, f₀=f₀, H=H, b=b, U=U, μ=μ, β=β, eta=nothing,
                                topographic_pv_gradient=(0,s*f₀/H[2]),
                                linear=false,
                                dt=dt, stepper=stepper, aliased_fraction=0)

    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y

    F1 = f₀^2/(H[1]*gp)
    F2 = f₀^2/(H[2]*gp)
    F = f₀^2/(sum(H)*gp)
    println(@sprintf("1/sqrt(F1), 1/sqrt(F2), 1/sqrt(F) = %1.1f, %1.1f, %1.1f km", 1e-3/sqrt(F1), 1e-3/sqrt(F2), 1e-3/sqrt(F)))

    Ld1 = sqrt(gp*H[1])/abs(f₀)
    dx = grid.dx
    log = @sprintf("Ld1 = %1.3f", Ld1)
    println(log)
    println(@sprintf("Ld1/dx = %1.1f", Ld1/dx))
    println(@sprintf("Ld1/L = %1.3f", Ld1/L))
    println(@sprintf("s * L/2/H2 = %1.3f", s*L/H[2]/2))
    println(@sprintf("s * Ld1/2/H2 = %1.3f", s*Ld1/H[2]/2))

    if field=="1"
        seed = 1234
    elseif field=="2"
        seed = 4321
    elseif field=="3"
        seed = 3142
    else
        seed = nothing
    end

    seed!(seed) # reset of the random number generator for reproducibility
    qmag = 3e-6 #2*U[1]/Ld
    q₀  = qmag * randn(grid.nx, grid.ny, nlayers)
    q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
    q₀  = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2
    MultiLayerQG.set_q!(prob, q₀)

    E = Diagnostic(MultiLayerQG.energies, prob; nsteps=nsteps)
    diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.



    startwalltime = time()
    fname = "simulation_s"*slope*"_strongmu_field"*field
    get_qh(prob) = prob.vars.qh
    get_u(prob) = prob.vars.u
    get_v(prob) = prob.vars.v
    out_fine = Output(prob,"../../Results/Results_GeophysicalFlows/SmallLd/"*fname*"_equilibrium.jld2", (:qh,get_qh),(:u,get_u),(:v,get_v))
    out_coarse = Output(prob,"../../Results/Results_GeophysicalFlows/SmallLd/"*fname*"_spinup.jld2", (:qh,get_qh),(:u,get_u),(:v,get_v))
    saveproblem(out_fine)
    saveproblem(out_coarse)

    outname_energy = "../../Results/Results_GeophysicalFlows/SmallLd/energies/"*fname*"_energies.jld2"

    function get_energies(prob)
        (KE1, KE2), PE = MultiLayerQG.energies(prob)
        return KE1, KE2, PE
    end

    if isfile(outname_energy); rm(outname_energy); end
    out_energy = Output(prob, outname_energy, (:E, get_energies))
    saveproblem(out_energy)



    for jout = 1:(nsteps/nsubs_output)
        stepforward!(prob, diags, nsubs_output)
        MultiLayerQG.updatevars!(prob)

        if jout%240 == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
            logdata = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, mean(U): %.2f m/s, walltime: %.2f min",
                        clock.step, clock.t/86400, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], mean(sqrt.(vars.u.^2 + vars.v.^2)), (time()-startwalltime)/60)
            println(logdata)
            saveoutput(out_energy)
            #if clock.t > (spinupdays-rundays/2)*86400 && clock.t < (spinupdays+rundays/2)*86400
            if clock.t < spinupdays*86400
                saveoutput(out_coarse)
            end
        end
        
        if clock.t > spinupdays*86400
            saveoutput(out_fine)
        end
    end


    println("Simulation with s = "*slope*" finished")
end
