import jax.numpy as jnp
import nibabel as nib
import numpy as onp
import pytest

from mriutils_in_jax.utils import take


INP = onp.arange(6, dtype=float).reshape((2, 3, 1))
EXP = jnp.arange(3, dtype=float)[:, None]


@pytest.fixture
def dataobj(tmp_path):
    # Create a temporary file path for the NIfTI file.
    nifti_path = tmp_path / "temp.nii.gz"

    # Create a NIfTI image with the dummy data.
    nifti_img = nib.nifti1.Nifti1Image(INP, onp.eye(4))
    nib.save(nifti_img, nifti_path)

    # Yield the file path for use in a test.
    yield nib.nifti1.load(nifti_path).dataobj

    # Cleanup: remove the file after the test run.
    if nifti_path.exists():
        nifti_path.unlink()


@pytest.mark.parametrize("inp", [INP, jnp.array(INP)])
def test_take(inp):
    res = take(inp, indices=0, axis=0)
    assert isinstance(res, jnp.ndarray)
    assert jnp.allclose(res, EXP)


def test_take_dataobj(dataobj):
    res = take(dataobj, indices=0, axis=0)
    assert isinstance(res, jnp.ndarray)
    assert jnp.allclose(res, EXP)
