"""
Produce integration test data, which is tested by the `test_integration_features.py`
tests. One thing to note here is that all redshifts are reasonably high.

This is necessary, because low redshifts mean that neutral fractions are small,
and then numerical noise gets relatively more important, and can make the comparison
fail at the tens-of-percent level.
"""
import glob
import hashlib
import os
import sys

import h5py
from powerbox import get_power

from py21cmfast import AstroParams
from py21cmfast import CosmoParams
from py21cmfast import FlagOptions
from py21cmfast import UserParams
from py21cmfast import run_coeval
from py21cmfast import run_lightcone

SEED = 12345
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")
DEFAULT_USER_PARAMS = {"HII_DIM": 50, "DIM": 150, "BOX_LEN": 100}
OPTION_NAMES = [
    "redshift",
    "z_step_factor",
    "z_heat_max",
    "HMF",
    "interp_perturb_field",
    "USE_MASS_DEPENDENT_ZETA",
    "SUBCELL_RSD",
    "INHOMO_RECO",
    "USE_TS_FLUCT",
    "M_MIN_in_Mass",
    "USE_FFTW_WISDOM",
]
OPTIONS = (
    [12, 1.02, 35, 1, False, False, False, False, False, False, False],
    [11, 1.05, 35, 1, False, False, False, False, False, False, False],
    [30, 1.02, 40, 1, False, False, False, False, False, False, False],
    [13, 1.05, 25, 0, False, False, False, False, False, False, False],
    [16, 1.02, 35, 1, True, False, False, False, False, False, False],
    [14, 1.02, 35, 1, False, True, False, False, False, False, False],
    [9, 1.02, 35, 1, False, False, True, False, False, False, False],
    [10, 1.03, 35, 2, False, False, False, True, False, False, False],
    [15, 1.02, 35, 3, False, False, False, False, True, False, False],
    [20, 1.02, 45, 2, False, False, False, False, False, True, False],
    [35, 1.02, 35, 1, False, False, False, False, False, False, True],
)


def get_defaults(kwargs, cls):
    return {k: kwargs.get(k, v) for k, v in cls._defaults_.items()}


def get_all_defaults(kwargs):
    flag_options = get_defaults(kwargs, FlagOptions)
    astro_params = get_defaults(kwargs, AstroParams)
    cosmo_params = get_defaults(kwargs, CosmoParams)
    user_params = get_defaults(kwargs, UserParams)
    return user_params, cosmo_params, astro_params, flag_options


def get_all_options(**kwargs):
    user_params, cosmo_params, astro_params, flag_options = get_all_defaults(kwargs)
    user_params.update(DEFAULT_USER_PARAMS)
    return {
        "redshift": kwargs.get("redshift", 7),
        "user_params": user_params,
        "cosmo_params": cosmo_params,
        "astro_params": astro_params,
        "flag_options": flag_options,
        "regenerate": True,
        "z_step_factor": kwargs.get("z_step_factor", None),
        "z_heat_max": kwargs.get("z_heat_max", None),
        "use_interp_perturb_field": kwargs.get("use_interp_perturb_field", False),
        "random_seed": SEED,
    }


def produce_coeval_power_spectra(**kwargs):
    options = get_all_options(**kwargs)

    coeval = run_coeval(**options)
    p, k = get_power(
        coeval.brightness_temperature.brightness_temp,
        boxlength=coeval.brightness_temperature.user_params.BOX_LEN,
    )

    return k, p, coeval


def produce_lc_power_spectra(**kwargs):
    options = get_all_options(**kwargs)

    lightcone = run_lightcone(max_redshift=options["redshift"] + 2, **options)

    p, k = get_power(
        lightcone.brightness_temp, boxlength=lightcone.lightcone_dimensions
    )

    return k, p, lightcone


def get_filename(**kwargs):
    fname = (
        f"power_spectra_z{kwargs['redshift']:.2f}_Z{kwargs['z_heat_max']}_"
        f"{hashlib.md5(str(kwargs).encode()).hexdigest()}.h5"
    )

    return os.path.join(DATA_PATH, fname)


def produce_power_spectra_for_tests(**kwargs):
    k, p, coeval = produce_coeval_power_spectra(**kwargs)
    k_l, p_l, lc = produce_lc_power_spectra(**kwargs)

    fname = get_filename(**kwargs)

    # Need to manually remove it, otherwise h5py tries to add to it
    if os.path.exists(fname):
        os.remove(fname)

    with h5py.File(fname, "w") as fl:
        for k, v in kwargs.items():
            fl.attrs[k] = v

        fl.attrs["HII_DIM"] = coeval.brightness_temperature.user_params.HII_DIM
        fl.attrs["DIM"] = coeval.brightness_temperature.user_params.DIM
        fl.attrs["BOX_LEN"] = coeval.brightness_temperature.user_params.BOX_LEN

        fl["power_coeval"] = p
        fl["k_coeval"] = k

        fl["power_lc"] = p_l
        fl["k_lc"] = k_l

        fl["xHI"] = lc.global_xHI
        fl["Tb"] = lc.global_brightness_temp

    print(f"Produced {fname} with {kwargs}")


if __name__ == "__main__":

    # Remove files that are there, unless -k cmd line arg specified.
    if len(sys.argv) == 1 or sys.argv[-1] != "-k":
        for fl in glob.glob(os.path.join(DATA_PATH, "*")):
            os.remove(fl)

    for option_set in OPTIONS:
        options = {name: option for name, option in zip(OPTION_NAMES, option_set)}
        produce_power_spectra_for_tests(**options)
