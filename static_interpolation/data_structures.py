import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

@dataclass(frozen=True)
class ImageLayout:
    """ Defines multi-panel image layout.
    and implements data standardization routines.

    Attributes:
        n_panels (int) : Number of panels
        num_x (int): Number of pixels in x-direction
        num_y (int) : Number of pixels in y-direction
    """
    n_panels: int
    num_x: int
    num_y: int
    @classmethod
    def from_shape(cls:type, data_shape:tuple[int,...]):
        """ImageLayout constructor using a shape tuple.
    
        Args:
            data_shape (tuple[int,...]): Shape tuple of length 2 or 3. Tuples of length 2 are interpreted as single panel shape, only specifying num_x and num_y.
        Returns:
            ImageLayout: instance
        """
        if len(data_shape) == 2:
            return cls(n_panels=1,num_x = data_shape[0],num_y = data_shape[1])
        elif len(data_shape) == 3:
            return cls(*data_shape)
        else:
            raise ValueError(f'Only data_shapes of length 2 or 3 are support but given length is {len(data_shape)}')
        
    def normalize(self, data:NDArray)->NDArray:
        """Normalizes input data shape to (n_images,n_panels,num_x,num_y)
        setting n_images and/or n_panels to 1 if necessary.
        Forces output data array to be c_contiguous.
        Possible intput shapes are:
        1. (num_x,num_y) -> (1,1,num_x,num_y)
        2. self.n_panels=1 and (n_images,num_x,num_y) -> (n_image,1,num_x,num_y)
        3. self.n_panels!=1 and (n_panels,num_x,num_y -> (1,n_panels,num_x,num_y)
        4. (n_image,n_panels,num_x,num_y) -> (n_image,n_panels,num_x,num_y)
        
        Args:
            data (NDArray):  Input data set.
        Returns:
            NDArray: reshaped version of data (No copy if input was C contiguous )
        """
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError(f"Input data is not C_CONTIGUOUS.")
        
        data_shape = data.shape
        len_data_shape = len(data_shape)
        if len_data_shape == 2:
            data = data[None,None,...]
            if data_shape!=(self.num_x,self.num_y):
                raise ValueError(f'Input data has wrong shape, expected length 2 shape is {(self.num_x,self.num_y)} but {data_shape} was given.')
        elif len_data_shape == 3:
            if self.n_panels==1:
                data = data[:,None,:,:]
                if data_shape[1:]!=(self.num_x,self.num_y):
                    raise ValueError(f'Input data has wrong shape, expected length 3 shape for 1 panel is {(...,self.num_x,self.num_y)} but {data_shape} was given.')
            else:
                data = data[None,...]
                if data_shape!=(self.n_panels,self.num_x,self.num_y):
                    raise ValueError(f'Input data has wrong shape, expected length 3 shape for multipanel data is {(self.n_panels,self.num_x,self.num_y)} but {data_shape} was given.')
        elif len_data_shape == 4:
            if data_shape[1:]!=(self.n_panels,self.num_x,self.num_y):
                raise ValueError(f'Input data has wrong shape, expected length 4 shape data is {(...,self.n_panels,self.num_x,self.num_y)} but {data_shape} was given.')
        else:
            raise ValueError(f'Length of data shape has to be 2, 3 or 4 but length is {len_data_shape}.')
        return data
    def ravel(self, data:NDArray)->NDArray:
        """ Ravel the image dimensions n_panel,num_x,num_y.
        (n_images,n_panels,num_x,num_y) -> (n_images,N) with N = n_panels*num_x*num_y
        Args:
            data (NDArray): Input data set
        Returns:
            NDArray: Reshaped data set
        """
        if data.shape[1:]!=self.data_shape:
            raise ValueError(f'Data shape is not normalized try to run .normalize on data first.')            
        return data.reshape(data.shape[0],-1)
    @property
    def data_shape(self)->tuple[int,...]:
        """
        tuple[int,int,int]: Data shape as tuple (n_panels,num_x,num_y) 
        """
        return (self.n_panels,self.num_x,self.num_y)
    
@dataclass(frozen=True)
class SamplingGrid:
    """ Defines sampling point collections.
    Attributes:
        points (NDArray): Has standard shape (n_panels,...,2).
        n_panels (int): Number of panels.
    """
    points: np.ndarray      # canonical shape: (panel, ..., 2)
    n_panels: int

    def __post_init__(self):
        """ Forces self.points to be C-contiguous and checks points shape.
        """
        if self.n_panels <= 0:
            raise ValueError(f"n_panels must be > 0, got {self.n_panels}.")

        points = self.points

        if not points.flags["C_CONTIGUOUS"]:
            raise ValueError(f"Sampling points are not C_CONTIGUOUS.")

        if points.ndim < 3 or points.shape[0] != self.n_panels or points.shape[-1] != 2:
            raise ValueError(
                f"points must have shape (n_panels, ..., 2), got {points.shape}."
            )

    @classmethod
    def from_shared_points(cls, points: np.ndarray, n_panels: int):
        """ SamplingGrid constructor for panel-independent points of shape (..., 2)."""
        points = np.ascontiguousarray(points)
        
        if n_panels <= 0:
            raise ValueError(f"n_panels must be > 0, got {n_panels}.")
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError(
                f"Shared points must have shape (..., 2), got {points.shape}."
            )
        
        panel_points = np.broadcast_to(points, (n_panels,) + points.shape).copy()
        return cls(points=panel_points, n_panels=n_panels)
            
            
    @classmethod
    def from_uniform_polar(cls:type,n_panels:int,shape:tuple[int,int],max_radius:float,endpoint:bool=False):
        """ Constructor of SamplingGrid for a uniform polar grid.
        
        Args:
            n_panels (int): number of panels
            shape (tuple(int,int)): shape of the uniform polar grid (number_of_radial_points,number_of_angular_points)
            max_radius (float): Maximal radial distance 
            endpoint (bool): Whether max_radius or max_radius*(number_of_radial_points-1)/number_of_radial_points is the last radial point. 
        """
        rs = np.linspace(0,max_radius,shape[0],endpoint=endpoint)
        phis = np.linspace(0,2*np.pi,shape[1],endpoint=False)
        points = np.stack(np.meshgrid(rs,phis,indexing="ij"),axis=-1)
        return cls.from_shared_points(points=points,n_panels=n_panels)
    
    def ravel(self)->NDArray:
        """ Returns self.points in the standard shape (n_panels,N,2)
        Returns:
            NDArray: reshaped version of self.points
        """
        return self.points.reshape(self.n_panels,-1,2)    
    @property
    def output_shape(self)->tuple[int,...]:
        """Returns the expected interpolation output shape.
        Follows the premis that panels are not allowed to overlap. So the output shape is self.points.shape[1:-1].
        
        Returns:
            tuple(int,...): Shape of interpolation output 
        """
        return self.points.shape[1:-1]
