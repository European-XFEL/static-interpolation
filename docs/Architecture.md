# Scope and Architecture

## Scope
This library is build focusing on 2D interpolations that satisfy the following constraints

1. Interpolation output must be __linear__ in the original image data.

	In equations each output of the interpolation $v$ must be computable via
	$$\\displaystyle v = \\sum_i d_i w_i $$
	where $d_i$ the data values and $w_i$ are weights that must be independent from the data values.
	
2. Data is given on panels consisting of square pixels  
   (or multiples of this pixel size like e.g. 2x1 pixels with twice the width).

3. Image data changes frequently while the sampling points stay fixed.

## Architecture
Static-interpolation treates an interpolation via 5 separate components.

1. A data structure describing the input [`ImageLayout`][static_interpolation.data_structures.ImageLayout]{target="_blank"}, e.g.

	```python
	from static_interpolation.data_structures import ImageLayout
	
	layout = ImageLayout(n_panels=3,num_x=64, num_y=16) 
	# Describes an input with two pixel panels of shape 64x16.
	```
	
2. A data structure dercribing the [`SamplingGrid`][static_interpolation.data_structures.SamplingGrid]{target="_blank"}, e.g.

	```python
	from static_interpolation.data_structures import SamplingGrid
	
	points = np.zeros((3,10,2),float)
	points[...,0] = (np.random.rand(10)*64)[None,...]
	points[...,1] = (np.random.rand(10)*16)[None,...]
	samples = SamplingGrid(points=points,n_panels=3) 
	# Describes 10 random sampling points on the entire detector 64x16.
	```
	
3. A [`CoordinateMapper`][static_interpolation.coordinate_mappers]{target="_blank"}. that maps sample points to image panel coordinates, e.g.

	```python
	from static_interpolation.coordinate_mappers import IdentityMapper
	
	idmapper = IdentityMapper()
	# Simplest possible coordinate mapper.
	assert idmapper.map(samples)==samples
	```

4.  An [`InterpolationPlanner`][static_interpolation.planning.InterpolationPlanner]{target="_blank"}, that precomputes the weights $w_i$ and stors them in an InterpolationPlan, e.g.

	```python
	from static_interpolation.planning import InterpolationPlanner
	from static_interpolation.config import InterpolationPolicy

	policy = InterpolationPolicy()
	planner = InterpolationPlanner()
	plan = planner.build(samples,layout,policy)
	```
	The policy object simply stores settings like whether to use linear or cubic interpolation.
	
5. An [`InterpolationEngine`][static_interpolation.engines.InterpolationEngine]{target="_blank"}, that takes an InterpolationPlan and actually performs the interpolation, e.g.

	```python
	from static_interpolation.engines import NumbaEngine
   
    engine = NumbaEngine(plan,layout,policy)
   
    # given some input data one can then perfom the interpolation via
    data = np.random.rand(layout.shape)
    engine(data)
	```







