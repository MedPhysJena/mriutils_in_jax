# mriutils in jax

Meant for some helpful utils and functions

## Installation

To install a specific tag / release directly from git:

```sh
pip install git+https://github.com/MedPhysJena/mriutils_in_jax.git@v0.2.6
```

You can omit the tag at the end (i.e. drop `@v0.2.5`) to checkout and install the current commit.

You can also checkout the repository locally and install it in the developmental mode ("editable installation"):

```sh
git clone https://github.com/MedPhysJena/mriutils_in_jax.git
pip install -e mriutils_in_jax
```

This will make all your local changes to the source code transparently available to the python environment.

Furthermore, a number of dependencies are shifted to extras (see [`pyproject.toml`](./pyproject.toml). 
Thus, to handle image registration in jax, specify `"registration"` optional dependency:

```sh
pip install -e "mriutils_in_jax[registration]"
```

You can list multiple extras as a comma-separated list in the brackets.

### GPU support

It is fully outsourced to jax. By default the dependencies mention the CPU only version of jax. 
You can simply specify `gpu` extra when installing this package, or learn more about jax installation [here](https://docs.jax.dev/en/latest/installation.html#installation).

In certain scenarios, one may encounter data that is too large for a single available GPU.
In this case, consider setting `JAX_PLATFORM_NAME=cpu` environment variable to force jax to
perform analysis on the CPU (and thus use RAM, which is often more permissive than that of a GPU)

## Usage

### Registration

Installing this package will make `register-echoes-to-last` command available within the python environment.
Consider running `register-echoes-to-last --help`. Additionally, the bulk of the functionality can be
used from within python using `mriutils_in_jax.register.register_complex_data`. Please, check its help.
