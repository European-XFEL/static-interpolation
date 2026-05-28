import numpy as np
from extra_geom.base import DetectorGeometryBase
from extra_geom.detectors import AGIPD_1MGeometry

from .config import InterpolationPolicy
from .data_structures import ImageLayout,SamplingGrid
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
    def __init__(self,
                 sampling_grid:SamplingGrid,
                 policy:InterpolationPolicy|None = None,
                 mapper:CoordinateMapper|None = None,
                 engine:type[InterpolationEngine] = NumbaEngine):
        
        logical_data_shape = (16,526,128)
        data_shape = (16,512,128)
        layout_logical = ImageLayout.from_shape(logical_data_shape)
        layout = ImageLayout.from_shape(data_shape)
        # compute stencils & weights with square logical pixels
        super().__init__(layout_logical,sampling_grid,policy,mapper,engine)

        # change all data indexes from (16,526,128) format to (512,128)
        self.plan.weight_indices[...] = self.agipd_logical_to_physical_indexing(self.plan.weight_indices)
        if self.plan.mean_fill_indices is not None:
            self.plan.mean_fill_indices[...] = self.agipd_logical_to_physical_indexing(self.plan.mean_fill_indices)
        if self.plan.nearest_data_id is not None:
            self.plan.nearest_data_id[...] = self.agipd_logical_to_physical_indexing(self.plan.nearest_data_id)

        # adjusting to the proper data_shape
        self.layout = layout
        self.engine.layout = layout

    @staticmethod
    def agipd_logical_to_physical_indexing(ids):
        '''converts logical pixel indexing, where double with pixels get two virtual pixels assigned,
            to the physical indexing where it is just one pixel.'''
        idm,idx,idy = np.unravel_index(ids,(16,526,128))
        o1 = (idx//66)*2
        o2 = (idx%66)//64
        idx = idx-o1-o2
        ids = np.ravel_multi_index((idm,idx,idy),dims=(16,512,128))
        return ids
    
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
