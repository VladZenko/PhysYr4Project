import numpy as np
import configparser
#from tqdm.autonotebook import tqdm
import scipy.interpolate
import scipy.ndimage
import scipy.fft as fft
from scipy import signal
import pylab as pl
import time
import requests
from bs4 import BeautifulSoup
import re
import random
import matplotlib.pyplot as plt
import cv2





class ForeGsim:
    def __init__(self, configFile="", ngrid=256, imgsize=10, nfreq=1, numin=150.0,
                 nustep=0.5, nexgal=13640809, sync_pli=-2.5, ff_pli=-3.,
                 exgal_pli=-0.7, psf_n_pixel = [10,10],psf_fwhm = [1,1],
                 version = "", seed = None):
        """
        Initialises an instance of the ForeGsym class.

        Params
        -------
        configFile : text
            A file containing all the parameters to pass to the class.
        imgsize : int
            The size of the output image in degrees.
        ngrid : int
            The size of the grid to simulate.
        nfreq : int
            The total number of frequency to generate sims for.
        numin : float
            The minimum frequency at which to generate a sim.
        nustep : float
            The jump between ajacent frequencies.
        nexgal : int
            The number of extragalactic galaxies.
        sync_pli : float
            The powerlaw index for the synchrotron distribution.
        ff_pli : float
            The powerlaw index for the free-free distribution.
        exgal_pli : float
            The powerlaw index for the extragalactic distribution.
        gauss_n_pixel : int 2D list
            The pixel size of the 2D Gaussian.
        gauss_fwhm : int 2D list
            The FWHM of the Gaussian along each axis.
        version : str
            empty or "original". If empty, it will use latest version, if 'vibor', it will use the original code

        """

        if configFile:
            try:
                config = configparser.ConfigParser()
                config.read(configFile)

            except IOError as io:
                print("An error occured trying to read the configFile.")
                # logger.exception("An error occured trying to read the configFile.")
        else:

            self.ngrid = ngrid
            self.imgsize = imgsize

            self.nfreq = nfreq
            self.numin = numin
            self.nustep = nustep
            self.nu = numin + nustep * np.arange(nfreq)

            self.nexgal = nexgal

            self.exgal_pli = exgal_pli
            self.sync_pli = sync_pli
            self.ff_pli = ff_pli

            self.psf_n_pixel = psf_n_pixel
            self.psf_fwhm = psf_fwhm

            self.version = version

            if seed is not None:
                np.random.seed(seed)
            else:
                pass



    def _gauss3d(self, pli, test=False, ran_field=None):
        """
        Generates a 3D random Gaussian field

        Params
        ------

        pli : float
            Powerlaw index for random field.
        test : boolean
            Changes between testing and production modes.
        ran_field : array
            Predefined random field used for testing.

        Returns
        -------
        array
        """


        n = self.ngrid

        kx = np.zeros([n])
        kx[0 : int(n / 2) + 1] = np.arange(int(n / 2) + 1)
        temp = np.arange(int(n / 2) - 1) + 1
        kx[int(n / 2) + 1 :] = -temp[::-1]

        kz = ky = kx

        if test:
            if ran_field is None:
                print("Random field not defined")
                return None
            else:
                print("Using test random field")
                phi = 2 * np.pi * ran_field

        else:
            print("Generating random field")
            phi = 2 * np.pi * np.random.rand(n, n, n)

        if self.version == "original":
            '''kamp = np.zeros([n, n, n])
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        kamp[i, j, k] = np.sqrt(kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2)'''
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            kamp = np.sqrt(kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2)
            kamp[np.where(kamp == 0)] = 0.0000001

            amp = kamp ** pli



            FTmapR = np.sqrt(amp) * np.cos(phi)
            FTmapI = np.sqrt(amp) * np.sin(phi)
            '''for i in range(1, int(n / 2)):
                for j in range(1, n):
                    for k in range(1, n):
                        FTmapR[n - i, n - j, n - k] = FTmapR[i, j, k]
                        FTmapI[n - i, n - j, n - k] = FTmapI[i, j, k]'''
            # Define slice indices
            i_slice = slice(1, int(n/2))  # Slices 1 to n/2 for first dimension
            j_slice = slice(1, n)         # Slices 1 to n for second and third dimensions

            # Apply mirroring
            FTmapR[-i_slice, -j_slice, -j_slice] = FTmapR[i_slice, j_slice, j_slice]
            FTmapI[-i_slice, -j_slice, -j_slice] = FTmapI[i_slice, j_slice, j_slice]
            # %%time
            '''i1 = 0
            i2 = int(n / 2)
            for j in range(1, int(n / 2)):
                for k in range(1, n):
                    FTmapR[i1, n - j, n - k] = FTmapR[i1, j, k]
                    FTmapI[i1, n - j, n - k] = -FTmapI[i1, j, k]
                    FTmapR[i2, n - j, n - k] = FTmapR[i2, j, k]
                    FTmapI[i2, n - j, n - k] = -FTmapI[i2, j, k]'''
            i1 = 0
            i2 = int(n / 2)

            # j slice excludes the first element (as original loop starts from 1)
            j_slice = slice(1, int(n / 2))
            k_slice = slice(1, n)

            # Mirroring for i1 and i2
            FTmapR[i1, n - j_slice, n - k_slice] = FTmapR[i1, j_slice, k_slice]
            FTmapI[i1, n - j_slice, n - k_slice] = -FTmapI[i1, j_slice, k_slice]
            FTmapR[i2, n - j_slice, n - k_slice] = FTmapR[i2, j_slice, k_slice]
            FTmapI[i2, n - j_slice, n - k_slice] = -FTmapI[i2, j_slice, k_slice]

            '''j1 = 0
            j2 = int(n / 2)
            for i in range(1, int(n / 2)):
                for k in range(1, n):
                    FTmapR[n - i, j1, n - k] = FTmapR[i, j1, k]
                    FTmapI[n - i, j1, n - k] = -FTmapI[i, j1, k]
                    FTmapR[n - i, j2, n - k] = FTmapR[i, j2, k]
                    FTmapI[n - i, j2, n - k] = -FTmapI[i, j2, k]'''
            j1 = 0
            j2 = int(n / 2)

            i_slice = slice(1, int(n / 2))

            # Mirroring for j1 and j2
            FTmapR[n - i_slice, j1, n - k_slice] = FTmapR[i_slice, j1, k_slice]
            FTmapI[n - i_slice, j1, n - k_slice] = -FTmapI[i_slice, j1, k_slice]
            FTmapR[n - i_slice, j2, n - k_slice] = FTmapR[i_slice, j2, k_slice]
            FTmapI[n - i_slice, j2, n - k_slice] = -FTmapI[i_slice, j2, k_slice]

            '''k1 = 0
            k2 = int(n / 2)
            for j in range(1, int(n / 2)):
                for i in range(1, n):
                    FTmapR[n - i, n - j, k1] = FTmapR[i, j, k1]
                    FTmapI[n - i, n - j, k1] = -FTmapI[i, j, k1]
                    FTmapR[n - i, n - j, k2] = FTmapR[i, j, k2]
                    FTmapI[n - i, n - j, k2] = -FTmapI[i, j, k2]'''
            k1 = 0
            k2 = int(n / 2)

            j_slice = slice(1, int(n / 2))

            # Mirroring for k1 and k2
            FTmapR[n - i_slice, n - j_slice, k1] = FTmapR[i_slice, j_slice, k1]
            FTmapI[n - i_slice, n - j_slice, k1] = -FTmapI[i_slice, j_slice, k1]
            FTmapR[n - i_slice, n - j_slice, k2] = FTmapR[i_slice, j_slice, k2]
            FTmapI[n - i_slice, n - j_slice, k2] = -FTmapI[i_slice, j_slice, k2]

#----------------------------------------------------
            
            i1 = 0
            j1 = 0
            i2 = int(n / 2)
            j2 = int(n / 2)
            i3 = 0
            j3 = int(n / 2)
            i4 = int(n / 2)
            j4 = 0
            
            for k in range(1, int(n / 2)):
                FTmapR[i1, j1, n - k] = FTmapR[i1, j1, k]
                FTmapI[i1, j1, n - k] = -FTmapI[i1, j1, k]
                FTmapR[i2, j2, n - k] = FTmapR[i2, j2, k]
                FTmapI[i2, j2, n - k] = -FTmapI[i2, j2, k]
                FTmapR[i3, j3, n - k] = FTmapR[i3, j3, k]
                FTmapI[i3, j3, n - k] = -FTmapI[i3, j3, k]
                FTmapR[i4, j4, n - k] = FTmapR[i4, j4, k]
                FTmapI[i4, j4, n - k] = -FTmapI[i4, j4, k]

            k1 = 0
            j1 = 0
            k2 = int(n / 2)
            j2 = int(n / 2)
            k3 = 0
            j3 = int(n / 2)
            k4 = int(n / 2)
            j4 = 0
            for i in range(1, int(n / 2)):
                FTmapR[n - i, j1, k1] = FTmapR[i, j1, k1]
                FTmapI[n - i, j1, k1] = -FTmapI[i, j1, k1]
                FTmapR[n - i, j2, k2] = FTmapR[i, j2, k2]
                FTmapI[n - i, j2, k2] = -FTmapI[i, j2, k2]
                FTmapR[n - i, j3, k3] = FTmapR[i, j3, k3]
                FTmapI[n - i, j3, k3] = -FTmapI[i, j3, k3]
                FTmapR[n - i, j4, k4] = FTmapR[i, j4, k4]
                FTmapI[n - i, j4, k4] = -FTmapI[i, j4, k4]

            i1 = 0
            k1 = 0
            i2 = int(n / 2)
            k2 = int(n / 2)
            i3 = 0
            k3 = int(n / 2)
            i4 = int(n / 2)
            k4 = 0
            for j in range(1, int(n / 2)):
                FTmapR[i1, n - j, k1] = FTmapR[i1, j, k1]
                FTmapI[i1, n - j, k1] = -FTmapI[i1, j, k1]
                FTmapR[i2, n - j, k2] = FTmapR[i2, j, k2]
                FTmapI[i2, n - j, k2] = -FTmapI[i2, j, k2]
                FTmapR[i3, n - j, k3] = FTmapR[i3, j, k3]
                FTmapI[i3, n - j, k3] = -FTmapI[i3, j, k3]
                FTmapR[i4, n - j, k4] = FTmapR[i4, j, k4]
                FTmapI[i4, n - j, k4] = -FTmapI[i4, j, k4]

            FTmapR[0, 0, 0] = 0.000001
            FTmapI[0, 0, 0] = 0.0

            FTmapI[int(n / 2), 0, 0] = 0.0
            FTmapI[0, int(n / 2), 0] = 0.0
            FTmapI[0, 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), 0] = 0.0
            FTmapI[0, int(n / 2), int(n / 2)] = 0.0
            FTmapI[int(n / 2), 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), int(n / 2)] = 0.0

            FTmap = FTmapR + 1j * FTmapI

            IFTmap = fft.ifftn(FTmap)

            Rmap = np.real(IFTmap)
            Rmap = Rmap / np.std(Rmap)



        elif self.version == "":
            kamp = np.zeros([n, n, n])
            i,j,k = np.meshgrid(range(n),range(n),range(n))
            kamp[i,j,k] = np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2)
            kamp[np.where(kamp == 0)] = 0.0000001

            amp = kamp ** pli

            FTmapR = np.sqrt(amp) * np.cos(phi)
            FTmapI = np.sqrt(amp) * np.sin(phi)

            i,j,k = np.meshgrid(range(1, int(n / 2)),range(1,n),range(1,n))
            FTmapR[n - i, n - j, n - k] = FTmapR[i, j, k]
            FTmapI[n - i, n - j, n - k] = FTmapI[i, j, k]


            i,j,k = np.meshgrid([0,int(n / 2)],range(1, int(n / 2)),range(1,n))
            FTmapR[i, n - j, n - k] = FTmapR[i, j, k]
            FTmapI[i, n - j, n - k] = -FTmapI[i, j, k]


            i,j,k = np.meshgrid(range(1, int(n / 2)),[0,int(n / 2)],range(1,n))
            FTmapR[n - i, j, n - k] = FTmapR[i, j, k]
            FTmapI[n - i, j, n - k] = -FTmapI[i, j, k]

            i,j,k = np.meshgrid(range(1,n),range(1, int(n / 2)),[0,int(n / 2)])
            FTmapR[n - i, n - j, k] = FTmapR[i, j, k]
            FTmapI[n - i, n - j, k] = -FTmapI[i, j, k]

            ilist = [0,int(n / 2),0,int(n / 2)]
            jlist = [0,int(n / 2),int(n / 2),0]
            i,j,k = np.meshgrid(ilist,jlist,range(1, int(n / 2)))
            FTmapR[i, j, n - k] = FTmapR[i, j, k]
            FTmapI[i, j, n - k] = -FTmapI[i, j, k]


            jlist = [0,int(n / 2),int(n / 2),0]
            klist = [0,int(n / 2),0,int(n / 2)]
            i,j,k = np.meshgrid(range(1, int(n / 2)),jlist,klist)
            FTmapR[n - i, j, k] = FTmapR[i, j, k]
            FTmapI[n - i, j, k] = -FTmapI[i, j, k]

            ilist = [0,int(n / 2),0,int(n / 2)]
            klist = [0,int(n / 2),int(n / 2),0]
            i,j,k = np.meshgrid(ilist,range(1, int(n / 2)),klist)
            FTmapR[i, n - j, k] = FTmapR[i, j, k]
            FTmapI[i, n - j, k] = -FTmapI[i, j, k]


            FTmapR[0, 0, 0] = 0.000001
            FTmapI[0, 0, 0] = 0.0

            FTmapI[int(n / 2), 0, 0] = 0.0
            FTmapI[0, int(n / 2), 0] = 0.0
            FTmapI[0, 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), 0] = 0.0
            FTmapI[0, int(n / 2), int(n / 2)] = 0.0
            FTmapI[int(n / 2), 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), int(n / 2)] = 0.0

            FTmap = FTmapR + 1j * FTmapI

            IFTmap = fft.ifftn(FTmap)

            Rmap = np.real(IFTmap)
            Rmap = Rmap / np.std(Rmap)

        else:
            raise Exception('Unknown version given')



        # The transpose is to much the format of the output of the original IDL code.
        return Rmap.T

    def _random_powerLaw(self, power, x_range=[0, 1], size=1):
        """
        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        Params
        ------

        power : float
        x_range : list
        size : int

        Returns
        -------
        array
        """
        a, b = x_range
        g = power + 1
        r = np.random.random(size=size)
        ag, bg = a ** g, b ** g
        return (ag + (bg - ag) * r) ** (1.0 / g)

    def _gauss2d_psf(self, n_pixel=[10, 10], FWHM=[1, 1]):
        """
        Normalised 2D Gaussian

        Params
        ------

        n_pixel : int 2D list
            The pixel size of the 2D Gaussian.
        FWHM : int 2D list
            The FWHM of the Gaussian along each axis.

        Returns
        -------
        array

        """

        nx_pix = n_pixel[0]
        ny_pix = n_pixel[1]

        sigma_x = FWHM[0] / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = FWHM[1] / (2 * np.sqrt(2 * np.log(2)))

        x = np.linspace(-nx_pix / 2, nx_pix / 2, nx_pix)
        y = np.linspace(-ny_pix / 2, ny_pix / 2, ny_pix)

        xx, yy = np.meshgrid(x, y)

        g = np.exp(-((xx ** 2) / (2 * sigma_x ** 2) + (yy ** 2) / (2 * sigma_y ** 2)))

        return g / np.sum(g)

    def _congrid(self, a, newdims, method="linear", centre=False, minusone=False):
        """
        Taken from scipy recipe book:

        Arbitrary resampling of source array to new dimension sizes.
        Currently only supports maintaining the same number of dimensions.
        To use 1-D arrays, first promote them to shape (x,1).

        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a VAX/VMS
        routine of the same name.

        method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates

        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin


        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
        """
        if not a.dtype in [np.float64, np.float32]:
            a = a.astype(float)

        m1 = np.cast[int](minusone)
        ofs = np.cast[int](centre) * 0.5
        old = np.array(a.shape)
        ndims = len(a.shape)
        if len(newdims) != ndims:
            print(
                "[congrid] dimensions error. "
                "This routine currently only support "
                "rebinning to the same number of dimensions."
            )
            return None
        newdims = np.asarray(newdims, dtype=float)
        dimlist = []

        if method == "neighbour":
            for i in range(ndims):
                base = np.indices(newdims)[i]
                dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
            cd = np.array(dimlist).round().astype(int)
            newa = a[list(cd)]
            return newa

        elif method in ["nearest", "linear"]:
            # calculate new dims
            for i in range(ndims):
                base = np.arange(newdims[i])
                dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
            # specify old dims
            olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

            # first interpolation - for ndims = any
            mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
            newa = mint(dimlist[-1])

            trorder = [ndims - 1] + [*range(ndims - 1)]
            for i in range(ndims - 2, -1, -1):
                newa = newa.transpose(trorder)

                mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
                newa = mint(dimlist[i])

            if ndims > 1:
                # need one more transpose to return to original dimensions
                newa = newa.transpose(trorder)

            return newa
        
        elif method in ["spline"]:
            oslices = [slice(0, j) for j in old]
            oldcoords = np.ogrid[oslices]
            nslices = [slice(0, j) for j in list(newdims)]
            newcoords = np.mgrid[nslices]

            newcoords_dims = range(np.rank(newcoords))
            # make first index last
            newcoords_dims.append(newcoords_dims.pop(0))
            newcoords_tr = newcoords.transpose(newcoords_dims)
            # makes a view that affects newcoords

            newcoords_tr += ofs

            deltas = (np.asarray(old) - m1) / (newdims - m1)
            newcoords_tr *= deltas

            newcoords_tr -= ofs

            newa = scipy.ndimage.map_coordinates(a, newcoords)
            return newa
        else:
            print(
                "Congrid error: Unrecognized interpolation type.\n",
                "Currently only 'neighbour', 'nearest','linear',",
                "and 'spline' are supported.",
            )
            return None

    def _rebin2D(self, a, new_shape=[None, None], mean=True):
        """
        Reduces the number of dimensions of an array through rebining.

        Params
        ------

        a : array
            Original array to be changed
        new_shape : list
            Dimensions of new array. Has to be a multiple of original array size.
        mean : boolean
            Changes the output array from a regular sum to mean.

        Returns
        -------
        array

        """

        new_shape = np.array(new_shape)
        factor_tmp = a.shape / new_shape
        rebin_factor = factor_tmp.astype(int)
        check = factor_tmp == rebin_factor
        '''if check[0] and check[1]:
            if mean:
                b = a.reshape(
                    new_shape[0], rebin_factor[0], new_shape[1], rebin_factor[1]
                )  # reshape into 4D array
                b = b.mean(1).mean(
                    2
                )  # Take mean across 1 axis to give a 3D array, then take mean accros last axis to get 2D array

            else:
                b = a.reshape(
                    new_shape[0], rebin_factor[0], new_shape[1], rebin_factor[1]
                )  # reshape into 4D array
                b = b.sum(1).sum(
                    2
                )  # Take sum across 1 axis to give a 3D array, then take sum accros last axis to get 2D arra

        else:
            print("New shape is not a multiple of old shape.")
            b = None'''
        if all(check):  # More concise way to check both conditions
            b = a.reshape(new_shape[0], rebin_factor[0], new_shape[1], rebin_factor[1])  # reshape into 4D array

            # Choose operation based on 'mean'
            operation = np.mean if mean else np.sum
            b = operation(operation(b, axis=1), axis=2)  # Apply operation twice to reduce to 2D

        else:
            print("New shape is not a multiple of old shape.")
            b = None

        return b

    def read_IDL_file(self, fname, shape=(256, 256, 256)):
        """
        Written specifically to read in IDL output arrays

        Params
        ------
        fname : str
            IDL file name
        shape : tuple
            Shape of array in IDL file.

        Returns
        -------
        array
        """
        file = open(fname, "r")
        lines = file.readlines()

        # remove front space and \n character
        lines = [lines[i][:-2].strip() for i in range(len(lines))]

        flatten_lines = np.array(lines).flatten()

        # separate strings into list and join all lists together into one array
        stacked_lines = np.hstack(np.char.split(flatten_lines, sep="  "))

        # remove elements of array with empty strings
        cleaned_lines = stacked_lines[stacked_lines != ""].astype(float)

        return cleaned_lines.reshape(shape)

    def gen_sync_map(self, test=False, ran_field_asyn=None, ran_field_plisyn=None):
        """
        Generates a synchrotron emmission foreground map.

        Params
        ------

        test : boolean
            Changes between testing and production modes.
        ran_field_asyn : array
            Predefined random field used for testing.
        ran_field_plisyn : array
            Predefined random field used for testing.

        Returns
        -------
        array

        """

        pli = self.sync_pli

        Asyn = self._gauss3d(pli=pli, test=test, ran_field=ran_field_asyn)
        PLIsyn = self._gauss3d(pli=pli, test=test, ran_field=ran_field_plisyn)

        norm = np.std(np.sum(Asyn, 2))
        Asyn = Asyn * 3.0 / norm  # @150MHz
        PLIsyn = -2.55 + PLIsyn * 0.1

        syn = np.zeros((self.ngrid, self.ngrid, self.nfreq))
        for i in range(self.nfreq):
            syntemp = Asyn * (self.nu[i] / 150.0) ** PLIsyn
            syn[:, :, i] = np.sum(syntemp, 2).T
            # Transpose to much orginal IDL output format
            print(("SYN @" + str(self.nu[i]) + "MHz: DONE!").split())

        return syn

    def gen_freefree_map(self, test=False, ran_field_aff=None, ran_field_pliff=None):
 

        pli = self.ff_pli

        Aff = self._gauss3d(pli=pli, test=test, ran_field=ran_field_aff)
        PLIff = self._gauss3d(pli=pli, test=test, ran_field=ran_field_pliff)

        norm = np.std(np.sum(Aff, 2))
        Aff = Aff * 0.03 / norm  # @150MHz
        PLIff = -2.15 + PLIff * 0.05

        ff = np.zeros((self.ngrid, self.ngrid, self.nfreq))
        for i in range(self.nfreq):
            fftemp = Aff * (self.nu[i] / 150.0) ** PLIff
            ff[:, :, i] = np.sum(
                fftemp, 2
            ).T  # Transpose to much orginal IDL output format
            print(("FF @" + str(self.nu[i]) + "MHz: DONE!").split())

        return ff

    def gen_snr_map(self):

        url = 'https://www.mrao.cam.ac.uk/surveys/snrs/snrs.data.html'

        # Fetch the content from the URL
        response = requests.get(url, verify=False)
        content = response.content

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        pre_tag = soup.find('pre')

        pre_contents = pre_tag.text

        lines = pre_contents.strip().split("\n")
        lines = [element for element in lines if element]
        lines_val = lines[4:-1]

        angle_max = self.imgsize

        snr_in_5x5 = 1.5
        snr_in_angle = int(np.round((snr_in_5x5/25)*angle_max**2))

        grid = self.ngrid*6
        random_points = [(random.randint(0, grid-1), random.randint(0, grid-1)) for _ in range(snr_in_angle)]

        sample_idx = [random.randint(0, 302) for _ in range(snr_in_angle)]

        a_values_list = []
        b_values_list = []
        flux_values_list = []
        spectral_idx = []

        for i in sample_idx:

            test_line = lines_val[i]
            
            columns = re.split(r' {2,}| (?=[+ - S F C])', test_line)
            if columns[0]=='':
                    columns.remove(columns[0])
            if columns[5]=='?':
                    columns[5] = 'S'

            columns = [s.replace('?', '') for s in columns]

            ang_size = columns[4] 
            flux = columns[6]
            spctrl_idx = columns[7]
            
            if spctrl_idx.strip() == '' or spctrl_idx.strip() == 'varies':
                    spctrl_idx = 0.5 # random value, can be substituted by mean/mode value from the column

            if '>' in flux:
                flux = flux.replace('>','')
            if flux.strip() == '':
                    flux = 7 # random value, can be substituted by mean/mode value from the column

            if 'x' in ang_size:
                a = float(ang_size.split('x')[0])
                b = float(ang_size.split('x')[1])
            else:
                r = float(ang_size)
                a = r
                b = r

            a_values_list.append(a/2)
            b_values_list.append(b/2)
            flux_values_list.append(float(flux))
            spectral_idx.append(float(spctrl_idx))

       
        nu_ref = float(1000) # 1 GHz to MHz

        spctrl_idx_array = np.array(spectral_idx, dtype=float)
        flux_values_list_array = np.array(flux_values_list, dtype=float)

        def flux_to_brghtnss_temp(freq, flux):

            freq_hz = freq * 10**6 # Hz
            c0 = 299792458 # m/s
            flux_mJy = flux * 10**3 # mJy
            Theta = 3 * 60 # arcsec, FoV of the instrument not from the BOX_LEN!!! Take as 3 arcmin

            lmbda = (c0/freq_hz) * 10**2 # cm
            S_nu = flux_mJy

            T = 1.360 * lmbda**2 * S_nu * (Theta**2)**(-1) # in K

            return T

        def draw_multiple_ellipses_on_grid(coordinates, a_values, b_values, angle_max, grid_size, T):

            canvas_angular_size_arcminutes = angle_max * 60

            # Scale factor: number of pixels per arcminute
            scale_factor = grid_size / canvas_angular_size_arcminutes

            grid = np.zeros((grid_size, grid_size))

            for (center_x, center_y), a, b, T in zip(coordinates, a_values, b_values, T):
                # Convert ellipse dimensions to grid units
                semi_major_axis_pixels = int(round(a * scale_factor))
                semi_minor_axis_pixels = int(round(b * scale_factor))

                # Calculate the range for the ellipse in both x and y directions
                x_range = np.arange(center_x - semi_minor_axis_pixels, center_x + semi_minor_axis_pixels + 1)
                y_range = np.arange(center_y - semi_major_axis_pixels, center_y + semi_major_axis_pixels + 1)

                # Ensure the ranges are within the grid boundaries
                x_range = x_range[(x_range >= 0) & (x_range < grid_size)]
                y_range = y_range[(y_range >= 0) & (y_range < grid_size)]

                # Fill pixels within the ellipse
                for x_pixel in x_range:
                    for y_pixel in y_range:
                        if ((x_pixel - center_x)**2 / semi_minor_axis_pixels**2 + 
                            (y_pixel - center_y)**2 / semi_major_axis_pixels**2) <= 1:
                            grid[y_pixel, x_pixel] = T

            return grid
       
        map = np.zeros((self.ngrid, self.ngrid, self.nfreq))

        for i in range(len(self.nu)):
            F_at_freq = flux_values_list_array*((self.nu[i]/nu_ref)**spctrl_idx_array)
            T = flux_to_brghtnss_temp(self.nu[i], F_at_freq)
            # Draw the ellipses
            map_slice = draw_multiple_ellipses_on_grid(random_points, a_values_list, b_values_list, angle_max, grid, T)
            map_slice = cv2.resize(map_slice, (self.ngrid, self.ngrid), interpolation=cv2.INTER_NEAREST)

            map[:,:,i] = map_slice
            map[:, :, i] -= np.mean(map[:, :, i])
            print(("SNR @" + str(self.nu[i]) + "MHz: DONE!").split())
            
        return map


    def gen_exgal_map(self, test = False, test_phi = None, test_dist = None):
        start_time = time.time()

        nexgal = self.nexgal
        imgsize = self.imgsize
        exgal_pli = self.exgal_pli

        if test == True:
            phi = test_phi
            dist = test_dist
        else:
            phi = 2 * np.pi * np.random.rand(nexgal)
            dist = self._random_powerLaw(power=exgal_pli, x_range=[0.005, imgsize], size=nexgal)

        time_phi_dist = time.time()
        print("Phi and Dist generation time: ", time_phi_dist - start_time)

        x = np.zeros(nexgal)
        y = np.zeros(nexgal)

        cos_phi = np.cos(phi[1:])
        sin_phi = np.sin(phi[1:])

        dx = np.cumsum(np.concatenate(([0], dist[1:] * cos_phi)))  # Cumulative sum for x displacements
        dy = np.cumsum(np.concatenate(([0], dist[1:] * sin_phi)))  # Cumulative sum for y displacements

        # Update x and y coordinates
        x = dx % imgsize  # Adjust for overflows
        y = dy % imgsize  # Adjust for overflows

        # Handle the conditions where x and y are greater or smaller than the imgsize / 2
        x[x > imgsize / 2] -= imgsize
        y[y > imgsize / 2] -= imgsize
        x[x < -imgsize / 2] += imgsize
        y[y < -imgsize / 2] += imgsize

        

        time_loop = time.time()
        print("loop time: ", time_loop - time_phi_dist)

        ngridtmp = 8*self.ngrid 
        res = imgsize / (ngridtmp - 1)

        xtmp = np.round((x + imgsize / 2) / res).astype(int)
        ytmp = np.round((y + imgsize / 2) / res).astype(int)

        n_pixel = self.psf_n_pixel
        FWHM = self.psf_fwhm

        psf = self._gauss2d_psf(n_pixel=n_pixel, FWHM=FWHM)

        time_gauss = time.time()
        print("gauss2d time: ", time_gauss - time_loop)

        bmaj = 2.0 * 60.0
        bmin = 2.0 * 60.0

        c = 2.99792458 * 1e8
        kb = 1.3806503 * 1e-23
        arcsectorad = np.pi / 648000.0
        solidang = (np.pi * (bmaj * arcsectorad) * (bmin * arcsectorad) / (4.0 * np.log(2.0)))

        exgal_data = np.loadtxt("exgalFG.1muJy.unr", usecols=(2, 3))

        time_load = time.time()
        print("load data time: ", time_load - time_gauss)

        extgalSJy = exgal_data[:, 0]
        extgalSpli = exgal_data[:, 1]

        exgal = np.zeros((self.ngrid, self.ngrid, self.nfreq))

        '''imgtmp = np.zeros((ngridtmp, ngridtmp, self.nfreq))
        
        imgtmp[xtmp, ytmp, :] = np.power((extgalSJy[:, np.newaxis] * (self.nu / 151.0)), extgalSpli[:, np.newaxis])
        imgtmp[np.where(imgtmp > 0.05 )] = 0
        #print(np.shape(imgtmp[xtmp, ytmp, :]))
        #print(np.shape(extgalSJy[:, np.newaxis]))
        #print(np.shape(self.nu))
        #print(np.shape(extgalSpli[:, np.newaxis]))'''


        for j in range(self.nfreq):
            imgtmp = np.zeros((ngridtmp, ngridtmp))
            imgtmp[xtmp, ytmp] = (extgalSJy * (self.nu[j] / 151.0) ** extgalSpli)
          
            #Flux cut
            imgtmp[np.where(imgtmp > 0.05 )] = 0

            imgtmpnu = self._rebin2D(a=imgtmp, new_shape=[self.ngrid, self.ngrid],mean = False)

            imgtmpnuC = signal.convolve2d(imgtmpnu, psf, mode='same')
            

            imgnu = self._rebin2D(a = imgtmpnuC,new_shape = [self.ngrid,self.ngrid],mean = False)

            
            exgal[:, :, j] = ((imgnu * 1e-26) / solidang)* c ** 2 / (2.0 * kb * (self.nu[j] * 1e6) ** 2)
            exgal[:, :, j] -= np.mean(exgal[:, :, j])

            # Normalize values larger than 24K or less than -24 to 24 or -24 respectively
            exgal[:, :, j] = np.clip(exgal[:, :, j], -24, 24)
        

        time_loop2 = time.time()
        print("main loop time: ", time_loop2 - time_load)

        return exgal

