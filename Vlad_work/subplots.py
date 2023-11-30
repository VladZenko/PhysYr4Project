import py21cmfast as p21c
from py21cmfast import plotting
import matplotlib.pyplot as plt
import os

print(f"21cmFAST version is {p21c.__version__}")

lightcone = p21c.run_lightcone(
    redshift = 7.0,
    max_redshift = 12.0,
    user_params = {"HII_DIM":100, "BOX_LEN": 600},
    lightcone_quantities=("brightness_temp", 'density'),
    global_quantities=("brightness_temp", 'density', 'xH_box'),
    direc='_cache'
)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

plotting.lightcone_sliceplot(lightcone, fig = fig, ax = ax1)
plotting.lightcone_sliceplot(lightcone, fig = fig, ax = ax2)
plt.show()

fig, axs = plt.subplots(3,1,
            figsize=())
for ii, lightcone_quantity in enumerate(lightcone_quantities):
    axs[ii].imshow(getattr(lightcone_fid, lightcone_quantity)[1],
                   vmin=vmins[ii], vmax=vmaxs[ii],cmap=cmaps[ii])
    axs[ii].text(1, 0.05, lightcone_quantity,horizontalalignment='right',verticalalignment='bottom',
            transform=axs[ii].transAxes,color = 'red',backgroundcolor='white',fontsize = 15)
    axs[ii].xaxis.set_tick_params(labelsize=0)
    axs[ii].yaxis.set_tick_params(labelsize=0)
plt.tight_layout()
fig.subplots_adjust(hspace = 0.01)