import numpy as np
from extra_geom.base import DetectorGeometryBase
from extra_geom.detectors import AGIPD_1MGeometry

from .config import InterpolationPolicy
from .data_structures import ImageLayout,SamplingGrid,AGIPD_1MLayout
from .coordinate_mappers import CoordinateMapper,EwaldSphereMapper,IdentityMapper
from .engines import InterpolationEngine,NumbaEngine
from .planning import InterpolationPlanner
from .utils import get_max_q

#----------------------
#   User facing API
class StaticInterpolator:
    """User Facing Interoplation Class"""
    def __init__(self,
                 layout:ImageLayout,
                 sample_grid:SamplingGrid,
                 policy:InterpolationPolicy|None = None,
                 mapper:CoordinateMapper|None = None,
                 engine:type[InterpolationEngine] = NumbaEngine):
        if policy is None:
            policy = InterpolationPolicy()
        if mapper is None:
            mapper = IdentityMapper()
        self.layout = layout
        self.sample_grid = sample_grid
        self.policy = policy
        self.mapper = mapper
        mapped_samples = self.mapper.map(sample_grid, layout)
        planner = InterpolationPlanner()
        self.plan = planner.build(
            sample_grid=mapped_samples,
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
        layout = ImageLayout.from_shape(geom.expected_data_shape)
        mapper = EwaldSphereMapper.from_geometry(geom,sample_detector_distance,xray_energy)
        if max_q is None:
            max_q = get_max_q(geom,sample_detector_distance,xray_energy,pad = True)
        n_panels = len(geom.modules)
        sampling_grid = SamplingGrid.from_uniform_polar(n_panels,(n_radial_samples,n_angular_samples),max_radius=max_q,)
        return cls(layout,sampling_grid,policy=policy,mapper=mapper,engine=engine)

    def __call__(self,data,masks=None,out=None,out_masks=None):
        return self.engine(data,masks = masks,out=out,out_masks=out_masks)
    
class AGIPD_1MInterpolator(StaticInterpolator):
    """ StaticInterpolator for the AGIPD_1M detector
    """
    def __init__(self,
                 sampling_grid:SamplingGrid,
                 policy:InterpolationPolicy|None = None,
                 mapper:CoordinateMapper|None = None,
                 engine:type[InterpolationEngine] = NumbaEngine):

        layout = AGIPD_1MLayout()
        super().__init__(layout,sampling_grid,policy,mapper,engine)
        
    @classmethod
    def from_polar_ewald(cls:type,
                         geom:AGIPD_1MGeometry,
                         n_radial_samples = 32,
                         n_angular_samples = 256,
                         xray_energy:float = 10000,
                         sample_detector_distance:float = 0,
                         max_q:float|None = None,
                         policy:InterpolationPolicy|None=None,
                         engine:type[InterpolationEngine] = NumbaEngine):
        if not isinstance(geom, AGIPD_1MGeometry):
            raise ValueError(f'geom is not an AGIPD_1MGeometry instance but of type {type(geom)}.')
            
        mapper = EwaldSphereMapper.from_geometry(geom,sample_detector_distance,xray_energy)
        if max_q is None:
            max_q = get_max_q(geom,sample_detector_distance,xray_energy,pad = True)
        n_panels = len(geom.modules)
        sampling_grid = SamplingGrid.from_uniform_polar(n_panels,(n_radial_samples,n_angular_samples),max_radius=max_q,)
        return cls(sampling_grid,policy=policy,mapper=mapper,engine=engine)
