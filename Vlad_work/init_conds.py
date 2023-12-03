import py21cmfast as cm21
import os
from py21cmfast import cache_tools


if not os.path.exists('_init_conds_cache'):
        os.mkdir('_init_conds_cache')

if not os.path.exists('initial_conditions_folder'):
        os.mkdir('initial_conditions_folder')

cm21.config['direc'] = '_init_conds_cache'



HII_DIM = int(400)
DIM = int(3*HII_DIM)
Mpc = 2000

init_cond = cm21.initial_conditions(
                                    user_params = {"DIM": DIM ,"HII_DIM": HII_DIM, "BOX_LEN": Mpc, "USE_INTERPOLATION_TABLES": True},
                                    cosmo_params = cm21.CosmoParams._defaults_,
                                    random_seed=4321
                                    )     

init_cond.save(fname='initial_conditions_folder/init_cond_DIM1200_2000Mpc.h5')                  