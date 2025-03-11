# mriutils in jax

Meant for some helpful utils and functions

## Installation

To install the module directly from git run

```sh
pip install git+https://github.com/MedPhysJena/mriutils_in_jax.git
```

You can also checkout the repository manually and install it in the developmental mode ("editable installation"):

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
