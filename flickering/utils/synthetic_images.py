import numpy as np
from scipy.interpolate import interp1d

def generate_cell_image(
    shape: tuple,
    center: tuple,
    radii: np.ndarray,
    profile_type: str = "gaussian_gradient",
    sigma: float = 2.0,
    amplitude: float = 1.0,
    background: float = 0.5,
    noise_sigma: float = 0.05
):
    """
    Generate a synthetic cell image with a given radial profile and fluctuations.
    
    Args:
        shape: (height, width) of the image.
        center: (y, x) center of the cell.
        radii: array of radii for each angle (assumed 0 to 2pi).
        profile_type: "gaussian" or "gaussian_gradient".
        sigma: width of the radial profile.
        amplitude: peak intensity (or max gradient).
        background: constant background value.
        noise_sigma: standard deviation of Gaussian noise.
    
    Returns:
        np.ndarray: synthetic image.
    """
    y, x = np.ogrid[:shape[0], :shape[1]]
    cy, cx = center
    
    dy = y - cy
    dx = x - cx
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) % (2 * np.pi)
    
    # Interpolate radii for each pixel's angle
    angles = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    # Handle wrapping
    r_interp = interp1d(angles, radii, kind='linear', fill_value="extrapolate", bounds_error=False)
    
    # Target radius for each pixel
    r0 = r_interp(angle)
    
    dr = dist - r0
    
    if profile_type == "gaussian":
        intensity = np.exp(-(dr**2) / (2 * sigma**2))
    elif profile_type == "gaussian_gradient":
        # Intensity proportional to the radial gradient of a Gaussian
        # I = -dr * exp(-dr^2 / 2sigma^2)
        intensity = - (dr / sigma) * np.exp(-(dr**2) / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    image = background + amplitude * intensity
    if noise_sigma > 0:
        image += np.random.normal(0, noise_sigma, shape)
        
    return image

def generate_fluctuating_radii(mean_radius, n_rays, modes):
    """
    Generate radii with sinusoidal fluctuations.
    
    Args:
        mean_radius: base radius.
        n_rays: number of angular samples.
        modes: dict of {mode_number: amplitude} or list of (mode, amplitude, phase).
    """
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.full_like(angles, mean_radius)
    
    if isinstance(modes, dict):
        for mode, amp in modes.items():
            radii += amp * np.cos(mode * angles)
    elif isinstance(modes, list):
        for mode, amp, phase in modes:
            radii += amp * np.cos(mode * angles + phase)
            
    return radii
