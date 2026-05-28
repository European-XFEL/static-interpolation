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
The possible vaues of an option are accessible via uppercase attributes, e.g.
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

## Boundary

* `Boundary.reject`: Do not interpolate samples for which data outside the data range is needed. For cubic(linear) interpolation the data in a 4x4(2x2) pixel area around the nearest pixel center of a given sample is needed.
* `Boundary.extrapolate_nearest`: Image data values outside of the data range are are filled by the value of the nearest pixel.
* `Boundary.extrapolate_linear`: (default) image data vlaues outside of the data range are are filled by linear extrapolation from the nearest points in the data range.

## Masking

* `Masking.Strict()`: (default)  A Sampling point is considered masked if any of the datapoints needed for its interpolation is masked. E.g. for cubic(linear) interpolation this means that an output sample point is masked, if any of its nearest 4x4 (2x2) pixel values on the detector is masked.
* `Masking.MeanFill(max_masked:int,mask_nearest:bool)`: Try to fill in masked data values by taking the mean of the 8 surrounding pixels  
   |o|o|o|  
   |o|x|o|  
   |o|o|o|  
   __max_masked__: is an integer in 1,...,7 specifying how many surrounding pixels are at most allowed to be masked.  
   __mask_nearest__: is a boolean flag. If it is set then samplest whose nearest data point is masked are also masked irrespective of the datapoint could have been filled by mean of the surrounding 8 pixels or not.

## OverlapMode

* `OverlapMode.error`: (default)
* `OverlapMode.first`:
