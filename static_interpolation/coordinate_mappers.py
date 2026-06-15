import abc
import numpy as np
from numpy.typing import NDArray
from extra_geom.base import DetectorGeometryBase
from .data_structures import ImageLayout,SamplingGrid,SamplingMeshRegular
from .utils import polar_scattering_coordinates_to_pixel_coordinates

#----------------------
# Sample to Pixelcoord
#      Mapping
class CoordinateMapper(abc.ABC):
    """ Abstract base class for mapping SamplingPoints onto pixel coordinates of the image layout.
    """
    @abc.abstractmethod
    def map(self,sampling_grid:SamplingGrid,layout:ImageLayout)->SamplingGrid:
        """ Maps points of a given sampling_grid onto pixel coordinates in the given image layout.
        
        Args:
            sampling_grid (SamplingGrid): input sampling_grid.
            layout (ImageLayout): input image layout.
        
        Returns:
            NDArray: sampling_grid.points in pixel coordinates.
        """
        pass
    
class IdentityMapper(CoordinateMapper):
    """Simplest possible mapper, it does nothing.
    """
    def map(self, sample_grid:SamplingGrid, layout:ImageLayout)->SamplingGrid:
        if not (sample_grid.n_panels == layout.n_panels):
            raise ValueError(f'Mismatch between sample_grid.n_panels ({sample_grid.n_panels}) and layout.n_panels ({layout.n_panels}).')
        return sample_grid
class GeometryMapper(CoordinateMapper):
    """
    Mapper Base class for all Mappers that use panel coordinate definitions and/or DetectorGeometryBase instances.

    Attributes:
        origins (NDArray): (n_panels,3),np.float64|  Starting corner positions of the pixel panels.
        origins_center (NDArray): (n_panels,3),np.float64| Pixel center of the pixel in the starting corner of each panel.
        xdirs (NDArray): (n_panels,3),np.float64|  Vectors pointing along the x-direction edge of the first pixel in each panel.
                         Their length needs to be equal to the pixel length.
        ydirs (NDArray): (n_panels,3),np.float64|  Vectors pointing along the y-direction edge of the first pixel in each panel.
                         Their length needs to be equal to the pixel length.
        n_panels (int): Number of panels.        
    """
    def __init__(self,
                 origins:NDArray|None = None,
                 xdirs:NDArray|None = None,
                 ydirs:NDArray|None = None):
        
        all_arrays = isinstance(origins,np.ndarray) and isinstance(xdirs,np.ndarray) and isinstance(ydirs,np.ndarray)
        any_arrays = isinstance(origins,np.ndarray) or isinstance(xdirs,np.ndarray) or isinstance(ydirs,np.ndarray)
        if all_arrays:
            if not (origins.shape==xdirs.shape==ydirs.shape):
                raise ValueError(f'Provided origins,xdirs and ydirs do not all have the same shapes.')
            xnorms = np.linalg.norm(xdirs,axis=-1)
            ynorms = np.linalg.norm(ydirs,axis=-1)
            if (xnorms==0).any():
                raise ValueError(f'There are xdirs with norm 0.')
            if (ynorms==0).any():
                raise ValueError(f'There are ydirs with norm 0.')
        elif any_arrays:
            raise ValueError(f'Mismatching input types: origins({type(origins)}),xdirs({type(xdirs)}) and ydirs({type(ydirs)}) ')

        self.origins = origins
        self.xdirs = xdirs
        self.ydirs = ydirs
        if all_arrays:
            self.n_panels = origins.shape[0]
        else:
            self.n_panels = None

    @property
    def origins_center(self):
        """ center positions of the first pixel of each panel"""
        return self.origins + (self.xdirs+self.ydirs)*0.5
        
    def validate(self,sampling_grid:SamplingGrid,layout:ImageLayout):
        """Validates self.n_panels against the given sampling_grid.n_panels and layout.n_panels.
           On Mismatch raises a ValueError
        Args:
            sampling_grid (SamplingGrid): Input sampling grid
            layout (ImageLayout): Input image layout
        """
        if not isinstance(sampling_grid,SamplingGrid):
            raise ValueError(f"sampling_grid has to be an instance of SamplingGrid but given object is of type {type(sampling_grid)}")
        if not (self.n_panels == sampling_grid.n_panels == layout.n_panels):
            raise ValueError(f'Mismatch between self.n_panels({self.n_panels}) of sampling_grid.n_panels ({sampling_grid.n_panels}) and layout.n_panels ({layout.n_panels}).')
        
    @staticmethod
    def parse_geometry(geom:DetectorGeometryBase,detector_origin:NDArray|None = None)->tuple[NDArray,NDArray,NDArray]:
        """ Translates a Detector geometry + sample detector origin into origin,xdirs,ydirs 
        Args:
            geom (DetectorGeometryBase): extra_geom.detectors.DetectorGeometryBase instance whose modules are interpreted as image panels.
            detector_origin (NDArray): x,y,z coordinates of the detector origin.
        Returns:
            tuple(NDarray,NDArray,NDArray): origins,xdirs,ydirs
        """
        pixel_size = geom.pixel_size
        n_mod = len(geom.modules)
        origins = np.zeros((n_mod,3),dtype=float)
        xdirs = np.zeros((n_mod,3),dtype=float)
        ydirs = np.zeros((n_mod,3),dtype=float)
        for i,m in enumerate(geom.modules):
            if isinstance(m,(list,tuple)):
                # igonore tiles in the geometry and just pic the first one
                m=m[0] 
            origins[i] = m.corner_pos
            xdirs[i] = m.ss_vec
            ydirs[i] = m.fs_vec
            xdirs[i] *= pixel_size/np.linalg.norm(xdirs[i]) 
            ydirs[i] *= pixel_size/np.linalg.norm(ydirs[i])
            
        if detector_origin is not None:
            origins[...]+=np.asarray(detector_origin)
        return origins,xdirs,ydirs

    @classmethod
    def from_geometry(cls,geom:DetectorGeometryBase,detector_origin:NDArray|None = None):
        """ Constructor based on DetectorGeometryBase instance.

        Args:
            geom (DetectorGeometryBase): geometry instance
            detector_origin (NDArray): x,y,z coordinates of the detector origin.
        
        Returns:
            GeometryMapper: Class instance
        """        
        origins,xdirs,ydirs = cls.parse_geometry(geom,detector_origin)
        return cls(origins,xdirs,ydirs)
    
    def map(self, sampling_grid: SamplingGrid, layout: ImageLayout) -> SamplingGrid:
        raise NotImplementedError
class ZProjectionMapper(GeometryMapper):
    ''' Projetcts 2D coordinates in a plane orthogonal to the z axis onto the pixel panels.
    '''
    def map(self, sampling_grid: SamplingGrid, layout: ImageLayout)->SamplingGrid:
        """ Takes a SamplingGrid and an image layout and uses the classes origins,xdir and ydir to create a new SamplingGrid.
            The points of the new grid are created by extending a line from each old point along the z-Axis until it hits a panel plane.
            New points are represented in units of pixel sizes.
        Args:
            sampling_grid (SamplingGrid): input SamplingGrid
            layout (ImageLayout): input ImageLayout
        
        Returns:
            SamplingGrid: Sampling grid with points in pixel units.
        
        """
        self.validate(sampling_grid,layout)
        normals = np.array([np.cross(x,y) for x,y in zip(self.xdirs,self.ydirs)])
        orthogonal_to_z = np.isclose(np.dot(normals,np.array([0,0,1])),0)
        if orthogonal_to_z.any():
            raise ValueError(f'There are panels whose normals are orthogonal the z-axis.')

        out_class = type(sampling_grid)
        out = out_class(sampling_grid.points.copy(),sampling_grid.n_panels)
        new_points = sampling_grid.ravel()
        out_points = out.ravel()
        zdir = np.array([0,0,1],dtype=np.float64)
        origins = self.origins_center
        for opts,pts,o,x,y in zip(out_points,new_points,origins,self.xdirs,self.ydirs):            
            # base points of lines parallel to the z axis
            line_base = np.zeros(pts.shape[:-1]+(3,),dtype=np.float64)
            line_base[...,:2] = pts
            
            # compute normal of detector plane
            normal = np.cross(x,y)
            normal /= np.linalg.norm(normal)
            
            # compute intersection of lines along z ayis with pixel plane
            length = (o-line_base).dot(normal)/zdir.dot(normal)

            # get pixel coordinate of sphericical scattering grid points.
            points = line_base + length[...,None]*zdir
            
            # set origin to plane origin
            points -= o

            # go to pixel coordinates
            x_size = np.linalg.norm(x)
            y_size = np.linalg.norm(y)
            opts[...] = (points @ (np.array([x/x_size**2,y/y_size**2]).T))
        return out
class EwaldSphereMapper(GeometryMapper):
    '''
    Maps polar coordinates on the Ewald's sphere onto the pixel planes.
    
    Arguments:
        xray_energy (float): X-ray energy in electron Volt [eV].
    '''
    def __init__(self, origins, xdirs, ydirs, xray_energy):
        super().__init__(origins,xdirs,ydirs)
        self.xray_energy = xray_energy 

    @classmethod
    def from_geometry(cls,geom:DetectorGeometryBase,detector_origin:NDArray,xray_energy:float):
        """ Overriding baseclass constructor from_geometry to take the extra xray_energy variable into account

        Args:
            geom (DetectorGeometryBase): extra_geom DetectorGeometryBase instance. Its modules define the planes.
            detector_origin (NDArray): x,y,z coordinates of detector origin in meters [m].
            xray_energy (float): X-ray energy in electron Volt [eV].

        Returns:
            EwaldSphereMapper: [description]
        """
        origins,xdirs,ydirs = cls.parse_geometry(geom,detector_origin)
        return cls(origins,xdirs,ydirs,xray_energy)
    
    def map(self, sampling_grid:SamplingGrid, layout:ImageLayout):
        self.validate(sampling_grid,layout)
        out_class = type(sampling_grid)
        out = out_class(sampling_grid.points.copy(),sampling_grid.n_panels)
        new_points = sampling_grid.ravel()
        out_points = out.ravel()
        origins = self.origins_center
        for opts,pts,o,x,y in zip(out_points,new_points,origins,self.xdirs,self.ydirs):
            opts[:] = polar_scattering_coordinates_to_pixel_coordinates(pts,o,x,y,self.xray_energy)        
        return out
