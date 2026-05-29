# AGIPD Detector
The Adaptive-Gain Integrating Pixel Detector (AGIPD) deployed at EuXFEL has a special Interpolation class
that propperly takes the double sized pixels at its ASIC boundaries into account.
```python
from static_interpolation import AGIPD_1MInterpolator
```
An example usage can be found [here](index.md#agipd-ewald-example)  
[Extra-Geom page on AGIPD](https://extra-geom.readthedocs.io/en/latest/agipd_geometry.html){target="_blank"}
