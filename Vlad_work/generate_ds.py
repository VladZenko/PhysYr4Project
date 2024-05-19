import py21cmfast as p21c
from py21cmfast import plotting
import os
import matplotlib.pyplot as plt
from py21cmfast import cache_tools
import numpy as np
import cv2
import ObserverLightcone
import random


random_integers = [random.randint(1, 100) for _ in range(100)]

box_len = 500
pix_side = 64

counter = 0



for i in random_integers:
    
    cache_tools.clear_cache(direc="_cache")

    lightcone = p21c.run_lightcone(
        redshift = 8.9, #6.0
        max_redshift = 12.1, #15.0
        random_seed=i,
        astro_params = p21c.AstroParams({"HII_EFF_FACTOR":120.0}),
        user_params = {"HII_DIM": pix_side, "BOX_LEN": box_len},
        lightcone_quantities=("brightness_temp", 'density'),
        global_quantities=("brightness_temp", 'density', 'xH_box'),
        direc='_cache'
    )

    sim_lc_obj = ObserverLightcone.Lightcone_21cmFAST(lightcone)

    obs_lc_obj = sim_lc_obj.convert_to_obs_space(pix_side)

    counter += 1

    np.save('./dataset/lc/lc_{}.npy'.format(counter), obs_lc_obj[0])

    print('{}/100 DONE'.format(counter))


