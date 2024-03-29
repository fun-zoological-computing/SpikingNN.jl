using SpikingNN
using Plots
using UnicodePlots
using OhMyREPL
using SnoopCompile
using Cthulhu
using Revise
using Infiltrator: @infiltrate
using Cthulhu: @descend
using Base: @noinline # already exported, but be explcit
using Pkg
#Pkg.add("https://github.com/JuliaDebug/Cthulhu.jl")
#using StatsBase
using Random
# simulation parameters
T = 1000

# neuron parameters
vᵣ = 0
τᵣ = 1.0
vth = 1.0

# create population
# neuron 1, 2 excite neuron 3
# neuron 3 inhibits neuron 1, 2
#weights_ = [ 0  0  1;
#            0  0  1;
#           -5 -5  0]
#random = StatsBase.random
weights_ = reshape(rand(100),(10,10))
#weights = []
#append!(weights,weights_)
#append!(weights,weights_)
#append!(weights,weights_)
#we
UnicodePlots.spy(weights_) |> display

@show(size(weights_))

pop = Population(weights_; cell = () -> LIF(τᵣ, vᵣ),
                          synapse = Synapse.Alpha,
                          threshold = () -> Threshold.Ideal(vth))

@show(pop)
#descend(f, tt)
# create input currents
low = ConstantRate(0.1)
high = ConstantRate(0.99)

switch(t; dt = 1) = (t < Int(T/2)) ? low(t; dt = dt) : high(t; dt = dt)

n1synapse = QueuedSynapse(Synapse.Alpha())
n2synapse = QueuedSynapse(Synapse.Alpha())

excite!(n1synapse, filter(x -> x != 0, [low(t) for t = 1:T]))
excite!(n2synapse, filter(x -> x != 0, [switch(t) for t = 1:T]))

# simulate
voltages = Dict([(i, Float64[]) for i in 1:length(pop)])
cb = () -> begin
    for id in 1:size(pop)
        push!(voltages[id], getvoltage(pop[id]))
    end
end

Cthulhu.@descend simulate!(pop, T; cb = cb, inputs = [n1synapse, n2synapse,n2synapse,n2synapse,n2synapse,n2synapse,n2synapse,n2synapse, n2synapse,n2synapse, (t; dt) -> 0])
@time outputs = simulate!(pop, T; cb = cb, inputs = [n1synapse, n2synapse,n2synapse,n2synapse,n2synapse,n2synapse,n2synapse,n2synapse, n2synapse,n2synapse, (t; dt) -> 0])

rasterplot(outputs) | display
#title!("Raster Plot")
#xlabel!("Time (sec)")
