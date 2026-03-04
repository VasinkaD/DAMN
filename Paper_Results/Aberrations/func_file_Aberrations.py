"""
psf_toolbox.py
This module provides tools for generating point spread functions (PSFs) and rendering emitter distributions convolved with a PSF.

Functions:
    - psf_high_NA_vector: Generates a 2D in-focus scalar PSF with Zernike aberrations and √cosθ apodisation.
    - scaled_psf_vector: Returns a 50x50 PSF, normalized, with fixed pupil sampling and pixel size.

Dependencies:
    numpy, cv2, matplotlib, scipy.ndimage, h5py
"""

import numpy as np
import cv2
from numpy.fft import fftshift, ifft2, ifftshift
import scipy.ndimage as ndi
import h5py
from typing import Tuple

def psf_high_NA_vector(N_pupil=256, pad_factor=8,
                       wavelength=0.580,        # µm
                       NA=1.40, n=1.5,
                       pol='circ',
                       A_def=0.0, A_ast=0.0, A_ast_y=0.0,
                       A_coma_x=0.0, A_coma_y=0.0,
                       A_trefoil_x=0.0, A_trefoil_y=0.0, A_sph=0.0):
    """
    Vectorial in-focus PSF via Debye–Wolf integral with Zernike aberrations.

    Parameters
    ----------
    N_pupil    : samples across aperture diameter
    pad_factor : zero-padding factor for image plane sampling
    wavelength : wavelength [µm]
    NA, n      : numerical aperture, immersion index
    pol        : 'x', 'y', or 'lin45' input polarization
    Aberration coeffs in waves at the pupil edge.
    
    Returns
    -------
    psf : (M,M) normalised intensity
    x,y : coordinates in µm
    """
    k = 2 * np.pi / wavelength
    M  = int(N_pupil * pad_factor)
    P  = np.zeros((M, M), dtype=complex)

    mid    = M // 2
    half   = N_pupil // 2
    sl     = slice(mid - half, mid + half)
    u,v    = np.mgrid[-1:1:N_pupil*1j, -1:1:N_pupil*1j]
    rho    = np.sqrt(u**2 + v**2)
    theta  = np.arctan2(v, u)
    mask   = rho <= 1

    # # Map pupil radius rho -> sinθ
    sin_t_max = NA / n
    sin_t     = rho * sin_t_max
    sin_t[~mask] = 0.0
    cos_t     = np.sqrt(1 - sin_t**2)

    # Zernike polynomials (unnormalised)
    Z20 = 2*rho**2 - 1
    Z22 = rho**2 * np.cos(2*theta)
    Z2m2 = rho**2 * np.sin(2*theta)
    Z31 = (3*rho**3 - 2*rho) * np.cos(theta)
    Z3m1 = (3*rho**3 - 2*rho) * np.sin(theta)
    Z33 = rho**3 * np.cos(3*theta)
    Z3m3 = rho**3 * np.sin(3*theta)
    Z40 = 6*rho**4 - 6*rho**2 + 1

    # Normalize Zernikes over pupil
    for Z in (Z20, Z22, Z2m2, Z31, Z3m1, Z33, Z3m3, Z40):
        Z /= np.sqrt(np.mean(Z[mask]**2))

    phase = 2 * np.pi * (
        A_def       * Z20 +
        A_ast       * Z22 +
        A_ast_y     * Z2m2 +
        A_coma_x    * Z31 +
        A_coma_y    * Z3m1 +
        A_trefoil_x * Z33 +
        A_trefoil_y * Z3m3 +
        A_sph       * Z40
    )

    # Vectorial pupil field (Richards–Wolf)
    if pol.lower() == 'x':
        Ex_pupil = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_pupil = (1 - cos_t) * np.sin(2*theta)
    elif pol.lower() == 'y':
        Ex_pupil = (1 - cos_t) * np.sin(2*theta)
        Ey_pupil = cos_t - (1 - cos_t) * np.cos(2*theta)
    elif pol.lower() == 'lin45':
        Ex_lin = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_lin = (1 - cos_t) * np.sin(2*theta)
        Ex_pupil = (Ex_lin + Ey_lin) / np.sqrt(2)
        Ey_pupil = (Ey_lin + (cos_t - (1 - cos_t) * np.cos(2*theta))) / np.sqrt(2)
    elif pol.lower() == 'circ':
        # Circular polarization = (x-pol + i y-pol) / sqrt(2)
        Ex_x = cos_t + (1 - cos_t) * np.cos(2*theta)
        Ey_x = (1 - cos_t) * np.sin(2*theta)
        Ex_y = (1 - cos_t) * np.sin(2*theta)
        Ey_y = cos_t - (1 - cos_t) * np.cos(2*theta)
        Ex_pupil = (Ex_x + 1j * Ex_y) / np.sqrt(2)
        Ey_pupil = (Ey_x + 1j * Ey_y) / np.sqrt(2)
    else:
        raise ValueError("pol must be 'x', 'y', 'lin45', or 'circ'")

    # Apodisation = sqrt(cosθ) from Debye–Wolf theory
    apod = np.sqrt(cos_t) * mask

    # Apply phase and apodisation
    Px = apod * Ex_pupil * np.exp(1j*phase)
    Py = apod * Ey_pupil * np.exp(1j*phase)

    # Insert into padded array
    Px_full = np.zeros_like(P)
    Py_full = np.zeros_like(P)
    Px_full[sl, sl] = Px
    Py_full[sl, sl] = Py

    # FFT to image plane
    fx = fftshift(ifft2(ifftshift(Px_full)))
    fy = fftshift(ifft2(ifftshift(Py_full)))

    # Intensity from vector components
    psf = np.abs(fx)**2 + np.abs(fy)**2
    psf /= psf.sum()

    # Coordinates in µm
    df = (NA / wavelength) / (N_pupil / 2)   # cycles/µm
    dx = 1 / (M * df)                        # µm per pixel
    coords = (np.arange(M) - mid) * dx

    return psf, coords, coords

def scaled_psf_vector(px_size_um: float, 
                      NA: float = 1.4, 
                      n: float = 1.5, 
                      wavelength: float = 0.580, 
                      N_pupil: int = 512, 
                      N_out = 50, 
                      **A_kwargs) -> Tuple[np.ndarray, float]:
    """
    Generate a 50x50 PSF from psf_high_NA_vector with given real pixel size.
    Warning: actual pixel size might differ, check the return value

    Parameters
    ----------
    px_size_um : float
        Pixel size in µm for the final 50x50 image.
    NA, n : float
        Numerical aperture and refractive index.
    wavelength : float
        Wavelength in µm.
    N_pupil : int
        Sampling settings for PSF simulation.
    **A_kwargs : dict
        Aberration coefficients for psf_high_NA_vector.

    Returns
    -------
    np.ndarray : 50x50 PSF normalized to sum = 1.
    dx : object plane pixel size in um
    """

    # Generate high-resolution PSF
    pad = (wavelength*0.5/NA)/px_size_um
    psf, x, y = psf_high_NA_vector(N_pupil=N_pupil, 
                                   pad_factor=pad, 
                                   wavelength=wavelength, 
                                   NA=NA, n=n, 
                                   **A_kwargs)
    
    # Desired physical FOV
    fov_um = N_out * px_size_um
    
    # Crop
    def crop_center(psf, x, N_out):
        mid = psf.shape[0] // 2
        half = N_out // 2
        
        sl = slice(mid - half, mid + half)
        return psf[sl, sl], x[sl]
    
    psf_crop, x_crop = crop_center(psf, x, N_out)
    psf_crop /= psf_crop.sum()
    dx = x_crop[1] - x_crop[0]
    
    # Normalize intensity
    return psf_crop, dx
