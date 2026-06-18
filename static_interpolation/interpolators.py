import numpy as np
from extra_geom.base import DetectorGeometryBase
from extra_geom.detectors import AGIPD_1MGeometry
import pickle
from dataclasses import dataclass
from numpy.typing import NDArray

from .config import InterpolationPolicy
from .data_structures import ImageLayout,SamplingGrid,AGIPD_1MLayout,JUNGFRAU_4MLayout,SamplingMeshRegular
from .coordinate_mappers import CoordinateMapper,EwaldSphereMapper,IdentityMapper
from .engines import InterpolationEngine,NumbaEngine
from .planning import InterpolationPlanner,InterpolationPlan,InterpolationPlannerMeshRegular
from .utils import get_max_q

@dataclass(frozen=True)
class InterpolationStruct:
    """ Data version of a StaticInterpolator.
    
    Attributes:
        plan (InterpolationPlan): Dataclass containing all precomputed arrays and information needed to perform the interpolation.
        data_shape (tuple[int,int,int]) : input data shape
        logical_shape (tuple[int,int]|None) : logical input data shape
        sampling_points (NDArray): original sampling points.
        policy (bytes): Pickled InterpolationPolicy object.
    """
    plan:InterpolationPlan
    data_shape:tuple[int,int,int]
    logical_shape:tuple[int,int,int]|None
    sampling_points: NDArray
    policy: bytes
    
    

#----------------------
#   User facing API
_MISSING = object
class StaticInterpolator:
    """User Facing Interoplation Class"""
    fixed_layout_class = _MISSING
    def __init__(self,
                 sampling_grid:SamplingGrid,
                 layout:ImageLayout|type[None] = type(None),
                 policy:InterpolationPolicy|None = None,
                 mapper:CoordinateMapper|None = None,
                 engine:type[InterpolationEngine] = NumbaEngine,
                 plan:InterpolationPlan|None = None):
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
        self.sampling_grid = sampling_grid
        self.policy = policy
        self.mapper = mapper

        if isinstance(plan,InterpolationPlan):
            self.plan = plan
        else:
            mapped_samples = self.mapper.map(sampling_grid, layout)            

            if policy.method == policy.Method.area:
                if not isinstance(sampling_grid,SamplingMeshRegular):
                    raise ValueError(f'When using method="area" the sampling_grid musst be of type SamplingMeshRegular but provided type is {type(sampling_grid)}')
                planner = InterpolationPlannerMeshRegular()
                self.plan = planner.build(
                    mapped_grid=mapped_samples,
                    layout=layout,
                    policy=policy
                )
            else:
                planner = InterpolationPlanner()
                self.plan = planner.build(
                    mapped_grid=mapped_samples,
                    layout=layout,
                    policy=policy
                )
            
        self.engine = engine(self.plan,layout,policy)
        
    @property
    def struct(self):
        return InterpolationStruct(
            plan = self.plan,
            data_shape = self.layout.data_shape,
            logical_shape = self.layout.logical_shape,
            sampling_points = self.sampling_grid.points,
            policy = pickle.dumps(self.policy)
        )
    
    @classmethod
    def from_struct(cls,
                    struct:InterpolationStruct,
                    engine:type[InterpolationEngine] = NumbaEngine):
        policy = pickle.loads(struct.policy)
        return cls(SamplingGrid(n_panels=struct.data_shape[0],
                                points = struct.sampling_points),
                   layout = ImageLayout.from_shape(struct.data_shape,struct.logical_shape),
                   policy=policy,
                   engine = engine,
                   plan = struct.plan
                   )
        
        
    @classmethod
    def from_polar_ewald(cls:type,
                         geom:DetectorGeometryBase,
                         n_radial_samples = 32,
                         n_angular_samples = 256,
                         xray_energy:float = 10000,
                         detector_origin:NDArray = np.array([0.0,0.0,0.0]),
                         max_q:float|None = None,
                         policy:InterpolationPolicy|None=None,
                         engine:type[InterpolationEngine] = NumbaEngine):
        if policy is None:
            policy = InterpolationPolicy()
        if issubclass(cls.fixed_layout_class,ImageLayout):
            layout = cls.fixed_layout_class()
        else:
            layout = ImageLayout.from_shape(geom.expected_data_shape)
        mapper = EwaldSphereMapper.from_geometry(geom,detector_origin,xray_energy)
        if max_q is None:
            max_q = get_max_q(geom,detector_origin,xray_energy,pad = True)
        n_panels = len(geom.modules)
        sample_class = SamplingGrid if policy.method!=policy.Method.area else SamplingMeshRegular
        samples = sample_class.from_uniform_polar(n_panels,(n_radial_samples,n_angular_samples),max_radius=max_q,)
        return cls(samples,layout=layout,policy=policy,mapper=mapper,engine=engine)

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
