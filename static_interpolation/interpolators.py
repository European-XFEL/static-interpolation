import numpy as np
from extra_geom.base import DetectorGeometryBase
from extra_geom.detectors import AGIPD_1MGeometry

from .config import InterpolationPolicy
from .data_structures import ImageLayout,SamplingGrid,AGIPD_1MLayout,JUNGFRAU_4MLayout
from .coordinate_mappers import CoordinateMapper,EwaldSphereMapper,IdentityMapper
from .engines import InterpolationEngine,NumbaEngine
from .planning import InterpolationPlanner
from .utils import get_max_q

#----------------------
#   User facing API
_MISSING = object()
class StaticInterpolator:
    """User Facing Interoplation Class"""
    fixed_layout_class = _MISSING
    def __init__(self,
                 sample_grid:SamplingGrid,
                 layout:ImageLayout|type[None] = type(None),
                 policy:InterpolationPolicy|None = None,
                 mapper:CoordinateMapper|None = None,
                 engine:type[InterpolationEngine] = NumbaEngine):
        if policy is None:
            policy = InterpolationPolicy()
        if mapper is None:
            mapper = IdentityMapper()
            
        if not isinstance(layout,ImageLayout):
            if not issubclass(self.fixed_layout_class,ImageLayout):
                raise TypeError('layout is needed for instanciation when fixed_layout_class is _MISSING.')
            else:
                layout = self.fixed_layout_class()
                
        self.layout = layout
        self.sample_grid = sample_grid
        self.policy = policy
        self.mapper = mapper
        self.mapped_samples = self.mapper.map(sample_grid, layout)            
            
        planner = InterpolationPlanner()
        self.plan = planner.build(
            sample_grid=self.mapped_samples,
            layout=layout,
            policy=policy
        )        
        self.engine = engine(self.plan,layout,policy)
        
    @classmethod
    def from_polar_ewald(cls:type,
                         geom:DetectorGeometryBase,
                         n_radial_samples = 32,
                         n_angular_samples = 256,
                         xray_energy:float = 10000,
                         sample_detector_distance:float = 0,
                         max_q:float|None = None,
                         policy:InterpolationPolicy|None=None,
                         engine:type[InterpolationEngine] = NumbaEngine):
        
        if issubclass(cls.fixed_layout_class,ImageLayout):
            layout = cls.fixed_layout_class()
        else:
            layout = ImageLayout.from_shape(geom.expected_data_shape)
        mapper = EwaldSphereMapper.from_geometry(geom,sample_detector_distance,xray_energy)
        if max_q is None:
            max_q = get_max_q(geom,sample_detector_distance,xray_energy,pad = True)
        n_panels = len(geom.modules)
        sampling_grid = SamplingGrid.from_uniform_polar(n_panels,(n_radial_samples,n_angular_samples),max_radius=max_q,)
        return cls(sampling_grid,layout=layout,policy=policy,mapper=mapper,engine=engine)

    def __call__(self,data,masks=None,out=None,out_masks=None):
        return self.engine(data,masks = masks,out=out,out_masks=out_masks)

    
class AGIPD_1MInterpolator(StaticInterpolator):
    """ StaticInterpolator for the AGIPD_1M detector
    """
    fixed_layout_class = AGIPD_1MLayout

class JUNGFRAU_4MInterpolator(StaticInterpolator):
    """ StaticInterpolator for the JUNGFRAU_4M detector
    """
    fixed_layout_class = JUNGFRAU_4MLayout
