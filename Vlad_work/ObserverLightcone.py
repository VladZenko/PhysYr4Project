import py21cmfast as p21c
from py21cmfast import plotting
import os
import matplotlib.pyplot as plt
from py21cmfast import cache_tools
import numpy as np
from scipy import interpolate
import cv2
from tqdm import tqdm




class Lightcone_21cmFAST():

    def __init__(self, lightcone_object):
        
        self.lightcone = lightcone_object

    def convert_to_obs_space(self):

        distances = getattr(self.lightcone, 'lightcone_distances')
    

        dist_max = np.max(distances)

        theta = np.arctan((getattr(self.lightcone, 'lightcone_dimensions')[0]/2)/dist_max)

        simulation = getattr(self.lightcone,'brightness_temp')

        img_profile = getattr(self.lightcone,'brightness_temp')[0,:,:]

        frames = []

        for i in range(len(distances)):

            dist = distances[len(distances)-1-i]
            half_box_mpc = np.tan(theta)*dist
            half_box_pixels = int(((np.shape(img_profile)[0]/2)/(np.tan(theta)*dist_max))*half_box_mpc)
            
            pixels_from_edge = int(np.shape(img_profile)[0]/2-half_box_pixels)
            
            if i==0:
                frame = np.copy(simulation[:,:,np.shape(img_profile)[1]-1-i])
            else:
                frame = np.copy(simulation[pixels_from_edge:-pixels_from_edge,
                                        pixels_from_edge:-pixels_from_edge,
                                        np.shape(img_profile)[1]-1-i])
                
                frame = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_NEAREST)
                
            
            frames.append(frame)

        lightcone_observer = np.stack(np.copy(frames), axis=2)
        lightcone_observer = np.flip(lightcone_observer, axis=2)


        #----------------------------------------------------------------------------

        z_axis = getattr(self.lightcone, 'lightcone_redshifts')
        nu_axis = 1420/(1+z_axis) #MHz
        nu_ax_lin = np.linspace(nu_axis[0], nu_axis[-1], len(nu_axis)) #MHz

        def find_closest_values(val, arr):
            # Assuming arr is sorted. If not, sort it first
            arr.sort()

            for i in range(len(arr)):
                if arr[i] >= val:
                    # Handle the case where the value is smaller than all elements in arr
                    if i == 0:
                        return None, arr[0]
                    # Return the closest smaller and larger values
                    return arr[i - 1], arr[i]

            # Handle the case where the value is larger than all elements in arr
            return arr[-1], None

        interpolated_lightcone = np.zeros_like(lightcone_observer)

        interpolated_lightcone[:,:,0] = lightcone_observer[:,:,-1]
        interpolated_lightcone[:,:,-1] = lightcone_observer[:,:,0]



        for i in tqdm(range(np.shape(lightcone_observer)[-1]-2)):

            idx = i+1
            
            smaller, larger = find_closest_values(nu_ax_lin[idx], nu_axis)

            smaller_norm = 0
            larger_norm = larger - smaller
            pos_norm = nu_ax_lin[idx] - smaller

            idx_low = np.argwhere(nu_axis==smaller)
            idx_high = np.argwhere(nu_axis==larger)


            array1 = lightcone_observer[:,:,idx_low[0][0]]
            array2 = lightcone_observer[:,:,idx_high[0][0]]


            # Locations
            loc1 = smaller_norm/larger_norm
            loc2 = larger_norm/larger_norm
            new_loc = pos_norm/larger_norm
        

            x = np.array([loc1, loc2])

            z = np.array([array1, array2])

            # Interpolating each point
            interpolated_array = np.zeros_like(array1)
            for i in range(np.shape(array1)[0]):
                for j in range(np.shape(array1)[1]):
                    interp_func = interpolate.interp1d(x, z[:, i, j], kind='nearest')
                    interpolated_array[i, j] = interp_func(new_loc)

            interpolated_lightcone[:,:,idx] = interpolated_array

            
        interpolated_lightcone = np.flip(interpolated_lightcone, axis=2)

        return interpolated_lightcone, nu_ax_lin, getattr(self.lightcone, 'lightcone_dimensions')[0]



