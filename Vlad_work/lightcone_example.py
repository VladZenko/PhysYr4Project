import py21cmfast as p21c
from py21cmfast import plotting
import matplotlib.pyplot as plt
import matplotlib
import os

print(f"21cmFAST version is {p21c.__version__}")

initial_conditions = p21c.InitialConditions(direc='cm21fast/init_conds',
                                            filename='init_cond_DIM690_2000Mpc.h5')

p21c.config['direc'] = '_cache_lc'

lightcone = p21c.run_lightcone(
    redshift = 8.0,
    max_redshift = 13.0,
    astro_params={"t_STAR": 1.0, }, # Double the default value
    init_box = initial_conditions,
    direc='_cache_lc'
)

lc_default = lightcone.save(direc='_cache_lc')

EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap',\
             [(0, 'white'),(0.33, 'yellow'),(0.5, 'orange'),(0.68, 'red'),\
              (0.83333, 'black'),(0.9, 'blue'),(1, 'cyan')])

fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(211)
img1 = getattr(lightcone, 'brightness_temp')[1]
ax1.imshow(img1, cmap='magma')
ax1.set_xticks([])
ax2 = fig.add_subplot(212)
img2 = getattr(lightcone, 'brightness_temp')[1]
ax2.imshow(img2, cmap=EoR_colour)

plt.tight_layout()
fig.subplots_adjust(hspace = 0.05)
plt.show()
