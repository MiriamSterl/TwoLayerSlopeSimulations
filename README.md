Code accompanying the manuscript "Asymmetric effects of topographic slopes on Lagrangian and Eulerian eddy diffusivities in two-layer QG flow" submitted to *JGR: Oceans* ([preprint][https://essopenarchive.org/users/744129/articles/1361790-asymmetric-effects-of-topographic-slopes-on-lagrangian-and-eulerian-eddy-diffusivities-in-two-layer-qg-flow?commit=384d86b515e79279f1fb6d333b494b6989a3806f]).

The script `simulation_GeophysicalFlows.jl` is run with the following parameters:
- `slope`: `["-7e-3", "-5e-3", "-3e-3","-2e-3","-1e-3","-7e-4","-5e-4","-3e-4","-2e-4","-1e-4", "0","1e-4","2e-4","3e-4","5e-4","7e-4","1e-3","2e-3","3e-3","5e-3","7e-3"]`
- `spinupdays`: 1000
- `rundays`: 670
- `field`: `["1", "2", "3"]` (for each slope, the simulations is seeded with three different random seeds)

The script `simulation_Parcels.py` is run for each (slope,field) combination above, with `rundays` set to 400.

The scripts starting with `compute` are used for analysis of the GeophysicalFlows and Parcels output. The output is then plotted in the notebooks starting with `plot`.

The folder `packages` contains lists of all the packages with their version numbers used for these scripts:
- `Manifest.toml`: overview of Julia packages used to run GeophysicalFlows.
- `py3_parcels.yml`: environment used to run Parcels.
- `2lqg.yml`: environment used to run all `compute` and `plot` scripts.
