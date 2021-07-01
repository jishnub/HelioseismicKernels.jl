# HelioseismicKernels

Time-distance sensitivity kernels in spherical geometry, accounting for line-of-sight projection and differences in line-formation heights. See [Bhattacharya et al. (2020)](https://arxiv.org/pdf/2009.07056.pdf) and [Bhattacharya (2020)](https://arxiv.org/pdf/2011.02180.pdf) for details of the code.

The code may be run on standalone machines, however performance will be better if it is run on HPC clusters. To this extent it depends on MPI through [MPI.jl](https://github.com/JuliaParallel/MPI.jl), and one might need to follow the instructions of the package to build it if a system-provided MPI installation is to be used. Typically one might need to load the MPI module on the cluster before building the package in order to use the system MPI. Dependencies would be automatically installed if the package is instantiated as

```
HelioseismicKernels/ $ julia --project
```

```julia
julia> using Pkg

julia> Pkg.instantiate()
```

The package may also be run interactively without using MPI. This is useful for tests, but won't be as performant.

# Running the code

## Evaluting Green functions

The first step is to evaluate Green functions that go into evaluating the kernel.
