# Options
Interpolation specific options are handled via an instance of [`InterpolationPolicy`][static_interpolation.config.InterpolationPolicy]{target="_blank"}.
Default options can be obtained via
```python
from static_interpolation.config import InterpolationPolicy
policy = InterpolationPolicy()
```
Currently selected options are stored as lower case attributes, e.g.
```python
policy.method
```
The possible values of an option are accessible via uppercase attributes, e.g.
```python
policy.Method.linear
policy.Method.cubic
# or equivalently
InterpolationPolicy.Method.linear
InterpolationPolicy.Method.cubic
```

## Method
Defines the interpolation type.

* `Methond.linear`: [Bilinear interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation){target="blank"}
* `Methond.cubic`: (default) [Bicubic (Catmull-Rom)](https://en.wikipedia.org/wiki/Bicubic_interpolation){target="_blank"} interpolation with $\alpha = 0.5$


<figure>
  <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
    <img src="../images/linear_kernel.png" alt="Image 1" style="flex: 1 1 20rem; width=300px; max-width: 100%; height: auto;">
    <img src="../images/cubic_kernel.png" alt="Image 2" style="flex: 1 1 20rem; width=300px; max-width: 100%; height: auto;">
  </div>
  <figcaption style="text-align: center;">Shows which data points of an underlying pixel grid are used in linear and cubic interpolation to computation of the value at the orange dot. Used pixels are shaded in blue and their centers are marked by black dots.</figcaption>
</figure>

## Boundary
All sampling points that lie outside of the underlying data range(pixel grid) are ignored.  
Sampling points which lie inside the data range but whose kernel contains points outside of it are affected by the following options.

<figure markdown>
 ![Image 1](../images/boundary.png){ width="300"  }
  <figcaption> Example of a sampling point whose kernel crosses the data boundary. Read shaded areas lie outside of the data range (pixel grid) but are needed to interpolate the orange sample. </figcaption>
</figure>

* `Boundary.reject`: Do not interpolate samples for which data points outside the data range is needed.
* `Boundary.extrapolate_nearest`: Data values outside of the data range are are filled by the value of the nearest pixel.
* `Boundary.extrapolate_linear`: (default) Data values outside of the data range are are filled by linear extrapolation from the nearest points in the data range.

## Masking
This option specifies how masked data points(pixels) should be treated.

* `Masking.Strict()`: (default)  A Sampling point is considered masked if any of the data points needed for its interpolation is masked. E.g. for cubic(linear) interpolation this means that an output sample point is masked, if any of its nearest 4x4 (2x2) pixel values on the detector is masked.
* `Masking.MeanFill(max_masked:int,mask_nearest:bool)`: Try to fill in masked data values by taking the mean of the 8 surrounding pixels.  
   __max_masked__: is an integer in 1,...,7 (default=3), specifying how many surrounding pixels are at most allowed to be masked.  
   __mask_nearest__: is a boolean flag (default=True). If it is set, then sampling points whose nearest data point is masked (before trying to fill their values) are always masked as well.
   
<figure markdown>
![Image 1](../images/meanfill.png){ width="300"  }
	   <figcaption> Consider the center red pixel to be masked. MeanFill tries to fill the masked value by taking the mean of its 8 surrounding pixels (blue area). It only does so if less or equal to `max_masked` of the blue pixels are masked as well.</figcaption>
   </figure>



## OverlapMode
If there are n image panels e.g.  
`layout = ImageLayout(n_panels=n,num_x=6,num_y=4)`  
then each sampling point has to be defined in pixel coordinates of each of the n panels.
E.g. consider m sampling points, then the sampling grid shape has to be (n,m,2).


It can happen that one of the m sampling points lies in the data
range of more than one of the n panels. This is a problem since then 
there is no unique association between the underlying data and this sampling point,
which means that there is more than one possible interpolation result.

This option specifies what to do in such a case.

* `OverlapMode.error`: (default) Raises a ValueException if `SamplingGrid` contains points that lie in the data range of more than one panel.
* `OverlapMode.first`: Use the "first" data panel (by panel id) in whose data range a given sample point lies.
