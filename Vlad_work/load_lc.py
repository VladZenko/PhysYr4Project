


import py21cmfast as p21c
from py21cmfast import plotting
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np

print(f"21cmFAST version is {p21c.__version__}")

lightcone1 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_default.h5')
lightcone2 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_default.h5')
lightcone3 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_HII60.h5')
lightcone4 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_tSTAR_0.1.h5')
lightcone5 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_HII_120.h5')
lightcone6 = p21c.LightCone.read('_cache_lc/LightCone_z8_z13_tSTAR_1.h5')

p21c.config['direc'] = '_cache_lc'

texts = ["HII_EFF_FACTOR: 30 ; t_STAR: 0.5","HII_EFF_FACTOR: 30 ; t_STAR: 0.5",
         "HII_EFF_FACTOR: 60 ; t_STAR: 0.5","HII_EFF_FACTOR: 30 ; t_STAR: 0.1",
         "HII_EFF_FACTOR: 120 ; t_STAR: 0.5","HII_EFF_FACTOR: 30 ; t_STAR: 1.0"]

plotting.lightcone_sliceplot(lightcone1)

fig = plt.figure(figsize=(15, 10))

fig.subplots_adjust(hspace=0.05)

new_ticks = np.arange(8.0, 13.0, 1)

for i in range(6):
    ax = fig.add_subplot(3, 2, i + 1)

    # Replace this with your actual image data
    img = getattr(eval(f'lightcone{i+1}'), 'brightness_temp')[1]

    cmap_choice = 'bone' if i % 2 == 0 else 'plasma'
    image_display = ax.imshow(img, cmap=cmap_choice)

    # Create an area for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar in the created area
    fig.colorbar(image_display, cax=cax)
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_xticks(np.linspace(0, (img.shape[1] - 1), 6))
    ax.set_xticklabels(np.arange(8.0, 13.1, 1.0))
    
    ax.text(0.95, 0.05, texts[i],
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes, color='green', backgroundcolor='k', fontsize=10)

plt.tight_layout()
plt.show()