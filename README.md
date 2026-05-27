# static-interpolation
Fast interpolation of panelized detector data, optimized for **static sampling positions**.

## Main features
* Plan first -> execute __fast__ approach.
* Mask aware interpolation.
* Native support for panelized data
* [EXtra-geom](https://github.com/European-XFEL/EXtra-geom) integration.

## Installation
It is available on pipy so you can simply pip install the package
```bash
pip install static-interpolation
```

## Usage
Let me give two examples 

### General example
```python
import numpy as np
import static_interpolation as si
from matplotlib import pyplot as plt

# simple cubic interpolation using 1 panel

# generate test data
rng = np.random.RandomState(12345)
nx,ny = 5,5
n_samples = 100
n_images = 20
data = rng.random((n_images,nx,ny))


# define image layout
layout = si.data_structures.ImageLayout.from_shape((nx,ny))

# define sampling points
sampling_points = np.stack(np.meshgrid(np.arange(n_samples)*(nx-1)/n_samples,np.arange(n_samples)*(ny-1)/n_samples,indexing='ij'),axis=-1)
samples = si.data_structures.SamplingGrid(points = sampling_points[None,...],n_panels=1)

# options
options = si.config.InterpolationPolicy()
options.method = options.Method.linear

# Instanciate interpolatiors
interp_cubic = si.interpolators.StaticInterpolator(layout,samples)
interp_linear = si.interpolators.StaticInterpolator(layout,samples,options)

# Execute interpolations for all 20 images
out = interp_cubic(data)
out_linear = interp_linear(data)

# Plot Result
fig,axs = plt.subplots(1,3,figsize=(10,5))
axs[0].imshow(data[0])
axs[0].set_title("Data")
axs[1].imshow(out[0])
axs[1].set_title("Cubic interpolation")
axs[2].imshow(out_linear[0])
axs[2].set_title("Linear interpolation")
plt.show()
```
![simple_interpolation](docs/images/simple_interpolation.png)

### AGIPD_1M Detector & polar grid on Ewald's sphere
```python
import static_interpolation as si
import numpy as np
from extra_geom import AGIPD_1MGeometry
from matplotlib import pyplot as plt

geom_agipd = AGIPD_1MGeometry.from_quad_positions(quad_pos=[
    (-525, 625), # in mm
    (-550, -10),
    (520, -160),
    (542.5, 475),
])

# Interpolation options
opt = si.config.InterpolationPolicy()
opt.method = opt.Method.cubic
opt.masking = opt.Masking.Strict()


# Things that are not in geom but needed
detector_distance = 0.2 # in meters
xray_energy = 7000 # in eV
nr,nphi = (512,2048)


# Instanciate the interpolator
agipd_interp = si.AGIPD_1MInterpolator.from_polar_ewald(geom_agipd,
                                         nr,
                                         nphi,
                                         xray_energy,
                                         detector_distance,
                                         max_q=None,#1e10,
                                         policy=opt)


# make test data
rng = np.random.RandomState(123456)

shape_agipd = (150,)+geom_agipd.expected_data_shape
data = rng.random(shape_agipd)
data*=np.arange(1,17)[None,:,None,None]
pixpos = geom_agipd.get_pixel_positions()
px, py, pz = np.moveaxis(pixpos, -1, 0)  # Separate x, y, z coordinates
angle = np.arctan2(py, px)
wedge_mask_agipd = (np.pi * 5/8 < angle) & (angle < np.pi * 7/8)
masks = (rng.random(shape_agipd)>0.005) & ~wedge_mask_agipd[None,...]


# Optional: pre-define output arrays 
out = np.zeros((150,nr,nphi),float)
out_masks = np.zeros(out_agipd.shape,bool)

# Interpolate
agipd_interp(data,
             masks,
             out = out,
             out_masks=out_masks)


# Plotting
shape=out[0].shape
rVal=np.arange(0,shape[0])
phiVal= np.linspace(0,2*np.pi,shape[1],endpoint=False)
r,phi=np.meshgrid(rVal,phiVal)

fig = plt.figure(figsize=(32,12))
axs = [fig.add_subplot(121, projection='polar'),fig.add_subplot(122, projection='polar')]
for i,ax in enumerate(axs):
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(linewidth = 2)
    if i==0:
        pltarr = out[0].copy()
        pltarr[~out_masks[0]]=np.nan
        im = ax.pcolormesh(phi,r,np.swapaxes(pltarr,0,1),cmap=plt.get_cmap('inferno'))
        fig.colorbar(im,extend='both')
        ax.set_title("Interpolation result",fontsize = 40)
    else:
        im = ax.pcolormesh(phi,r,np.swapaxes(out_masks[0],0,1),cmap=plt.get_cmap('inferno'))
        fig.colorbar(im,extend='both')
        ax.set_title("Interpolation mask",fontsize = 40)
        
plt.show()
```
![simple_interpolation](docs/images/agipd_interpolation.png)


