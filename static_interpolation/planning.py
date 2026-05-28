from numba import njit
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from dataclasses import dataclass
from .data_structures import SamplingGrid,ImageLayout
from .config import InterpolationPolicy

#---------------------
#   Precomputation
#   Plan creation
@dataclass(frozen=True)
class InterpolationPlan:
    """ Dataclass containing all precomputed arrays and information needed to perform the interpolation.
    
    Arguments:
        weight_indices (NDArray):(M,), data indice associated to a specific weight.
        weight_values (NDArray): (M,), weight values.
        n_weights_per_sample (NDArray): (n_valid,), number of weight_indices,weight_values used for the interpolation at a given sampling point.
        valid_sample_ids (NDArray): (n_valid,) Flattened indices of sample points whose mapped coordinates lie on a data plane.
        valid_sample_mask (NDArray): (n_samples,) Boolean mask whose true values indicate sampling points are mapped onto a data plane. (np.nonzero(valid_sample_mask)[0]==valid_sample_ids)
        output_shape (tuple[int,...]): Shape of the fineal interpolated image.
        mean_fill_indices (None|NDArray): 3x3 neighbor indices of each pixel on each data plane.
        nearest_data_id (None|NDArray): (n_valid,) pixel index of the nearest data point for each sampling point.
    """
    weight_indices: NDArray      #(M) | (n_valid, stencil_size)
    weight_values: NDArray      #(M) | (n_valid, stencil_size)
    n_weights_per_sample: NDArray #(n_valid)
    valid_sample_ids: NDArray   # (n_valid,) Ids of samples that lie within the data planes.
    valid_sample_mask: NDArray   # (n_samples,) Mask over all samples that is true if sample lies in one of the data planes.
    output_shape: tuple
    mean_fill_indices: None|NDArray # (n_data_points,8)
    nearest_data_id: None|NDArray # (n_valid)
    
    
@njit()
def cubic_kernel(t:float)->float:
    """
    Cubic interpolation kernel (Keys / Catmull–Rom-like with a = -0.5).
    
    Parameters
       t (float): scalar distance
    
    Returns
       float: Kernel value at t
    """
    a = -0.5
    x = abs(t)
    if x <= 1.0:
        return ((a + 2) * x**3 - (a + 3) * x**2 + 1.0)
    elif x < 2.0:
        return (a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a)
    else:
        return 0.0 
@njit()
def linear_kernel(t:float)->float:
    """
    Linear interpolation kernel.
    
    Parameters
       t (float): scalar distance
    
    Returns
       float: Kernel value at t.
    """
    return max(1.0 - abs(t), 0.0)

_CUBIC_LINEAR_CONTINUATION_BY_1_MATRIX = np.array([
    [ 2.0,  0.0, 0.0, 0.0],
    [ 0.0,  1.0, 0.0, 0.0],
    [-1.0,  0.0, 1.0, 0.0],
    [ 0.0,  0.0, 0.0, 1.0],
], dtype=np.float64)
_CUBIC_LINEAR_CONTINUATION_BY_1_MATRIX_REVERSED = np.ascontiguousarray(
    _CUBIC_LINEAR_CONTINUATION_BY_1_MATRIX[::-1,::-1]
)
_CUBIC_LINEAR_CONTINUATION_BY_2_MATRIX = np.array([
    [ 3.0,  0.0, 0.0, 0.0],
    [-2.0,  0.0, 0.0, 0.0],
    [ 0.0,  2.0, 1.0, 0.0],
    [ 0.0, -1.0, 0.0, 1.0],
], dtype=np.float64)
_CUBIC_LINEAR_CONTINUATION_BY_2_MATRIX_REVERSED = np.ascontiguousarray(
    _CUBIC_LINEAR_CONTINUATION_BY_2_MATRIX[::-1,::-1]
)
@njit()
def _cubic_indices_weights_1d_linear_continuation(i0: int, w: NDArray, n: int):
    """
    Return 4 sample indices and 4 weights for cubic interpolation
    with offsets [-1, 0, 1, 2], using linear continuation beyond
    the image border.

    Assumes pixel-center convention with valid coordinate range
    [-0.5, n-0.5).
    """
    idx = np.empty(4, dtype=np.int64)
    ww = np.empty(4, dtype=np.float64)
        
    # x in [-0.5, 0)  -> i0 = -1
    # stencil [-2, -1, 0, 1]
    # p[-2] = 3*p[0] - 2*p[1] = p[0] - 2*(p[1]-p[0])
    # p[-1] = 2*p[0] - p[1] = p[0] - 1*(p[1]-p[0])
    # sum = p[-2]w[0]+p[-1]w[1]+p[0]w[2]+p[1]w[3]
    #     = (3p[0]-2p[1])w[0] + (2p[0]-p[1])w[1] + p[0]w[2] + p[1]w[3]
    #     =  p[0](3w[0]+2w[1]+w[2]) + p[1](-2w[0])-w[1]+w[3])
    if i0 == -1:
        idx[:] = [0, 1, 0, 1]
        ww[:] = _CUBIC_LINEAR_CONTINUATION_BY_2_MATRIX@w
        return idx, ww
    
    # x in [0, 1) -> i0 = 0
    # stencil [-1, 0, 1, 2]
    # p[-1] = 2*p[0] - p[1]
    #     = (2p[0]-p[1])w[0] + p[0]w[1] + p[1]w[2] + p[2]w[3]
    #     = p[0](2*w[0]+w[1]) + p[1](w[2]-w[0]) + p[2]w[3]
    if i0 == 0:
        idx[:] = [0, 0, 1, 2]
        ww[:] = _CUBIC_LINEAR_CONTINUATION_BY_1_MATRIX@w
        return idx, ww
    
    # x in [n-2, n-1) -> i0 = n-2
    # stencil [n-3, n-2, n-1, n]
    # p[n] = 2*p[n-1] - p[n-2]
    if i0 == n - 2:
        idx[:] = [n - 3, n - 2, n - 1, n - 1]
        ww = _CUBIC_LINEAR_CONTINUATION_BY_1_MATRIX_REVERSED@w
        return idx, ww

    # x in [n-1, n-0.5) -> i0 = n-1
    # stencil [n-2, n-1, n, n+1]
    # p[n]   = 2*p[n-1] - p[n-2]
    # p[n+1] = 3*p[n-1] - 2*p[n-2]
    if i0 == n - 1:
        idx[:] = [n - 2, n - 1, n - 2, n - 1]
        [n - 1, n - 2, n - 1, n - 2]
        ww = _CUBIC_LINEAR_CONTINUATION_BY_2_MATRIX_REVERSED@w
        return idx, ww
    
    # interior
    idx[:] = [i0 - 1, i0, i0 + 1, i0 + 2]
    ww[:] = w
    return idx, ww
@njit()
def _linear_indices_weights_1d_linear_continuation(i0: int, w: NDArray, n: int):
    """
    Return 2 sample indices and 2 weights for linear interpolation
    with offsets [0, 1], using linear continuation beyond the image border.

    Assumes pixel-center convention with valid coordinate range
    [-0.5, n-0.5).
    """
    idx = np.empty(2, dtype=np.int64)
    ww = np.empty(2, dtype=np.float64)

    # x in [-0.5, 0) -> i0 = -1
    # stencil [-1, 0]
    # p[-1] = 2*p[0] - p[1]
    if i0 == -1:
        idx[:] = [0, 1]
        ww[:] = [2.0 * w[0] + w[1], -w[0]]
        return idx, ww

    # x in [n-1, n-0.5) -> i0 = n-1
    # stencil [n-1, n]
    # p[n] = 2*p[n-1] - p[n-2]
    if i0 == n - 1:
        idx[:] = [n - 2, n - 1]
        ww[:] = [-w[1], w[0] + 2.0 * w[1]]
        return idx, ww

    # interior
    idx[:] = [i0, i0 + 1]
    ww[:] = w
    return idx, ww

class InterpolationPlanner:
    """ Planner that creates InterpolationPlan instances.
        Currently supports linear and cubic interpolation.
    """
    def __init__(self):
        self._build_linear = self._make_builder(np.array([0,1]),linear_kernel)
        self._build_cubic = self._make_builder(np.array([-1,0,1,2]),cubic_kernel)
        
    def _make_builder(self,offsets:NDArray,kernel:Callable)->Callable:
        """ creates njit routines to generate weight_indices, weight_values etc.
            This allows to share most of the code between linear and cubic interpolation while still beeing
            numba njit routines that are part of a python class.
        
        Args:
            offsets (NDArray): Array containing stencil offsets (0,1) for linear interpolation (-1,0,1,2) for cubic. 
            kernel (Callable): linear_kernel or cubic_kernel
        
        Returns:
            Callable: Builder for linear or cubic interpolation weights.
        """
        @njit()
        def build(sample_points:NDArray,data_shape:tuple,linear_continuation:np.bool_,error_on_overlap:np.bool_,xlim:tuple,ylim:tuple):
            n_panels,num_x,num_y = data_shape
            valid_sample_mask = np.zeros(sample_points.shape[:2],dtype=np.bool_)

            point_processed = np.zeros(valid_sample_mask.shape[1:],dtype=np.bool_)
            for i,pts in enumerate(sample_points):
                for j,(x,y) in enumerate(pts):
                    point_in_data = (xlim[0]<=x<xlim[1]) and (ylim[0]<=y<ylim[1])
                    # The points_process mask ensures that a single sample point can at
                    # most be mapped to a single plane (the first one in whose data range it lies.)
                    if point_in_data:
                        if not point_processed[j]:
                            valid_sample_mask[i,j] = True
                            point_processed[j] = True
                        elif error_on_overlap:
                            raise ValueError(f'There are sampling points that belong to multiple data panels.')
                    
            valid_sample_ids = np.nonzero(valid_sample_mask)
            N = len(valid_sample_ids[0])            
            
            len_offsets = len(offsets)
            stencil_size = len_offsets**2
            idxs = np.empty((N, stencil_size), dtype=np.int64)
            weights = np.empty((N, stencil_size), dtype=np.float64)
            nearest_data_id = np.empty(N, dtype=np.int64)
            n_weights_per_sample = np.full(N,stencil_size, dtype=np.int64)
            
            wx = np.empty(len_offsets, dtype=np.float64)
            wy = np.empty(len_offsets, dtype=np.float64)
            for i,(panel_id,point_id) in enumerate(zip(*valid_sample_ids)):
                x, y  = sample_points[panel_id,point_id]
                x_round = max(min(int(np.floor(x+0.5)),num_x-1),0)
                y_round = max(min(int(np.floor(y+0.5)),num_y-1),0)
                nearest_data_id[i] = panel_id*num_x*num_y + x_round*num_y + y_round
                x0,y0 = int(np.floor(x)),int(np.floor(y)) 
                for io,o in enumerate(offsets):
                    wx[io] = kernel(x - np.float64(x0) - np.float64(o))
                    wy[io] = kernel(y - np.float64(y0) - np.float64(o))
                # normalize separable weights
                sx,sy = wx.sum(),wy.sum()
                if sx != 0.0:
                    wx /= sx
                if sy != 0.0:
                    wy /= sy
                    
                k = 0
                
                # do clamping or linear continuation
                if linear_continuation:
                    if len_offsets == 4:
                        x_idx, x_w = _cubic_indices_weights_1d_linear_continuation(x0 , wx, num_x)
                        y_idx, y_w = _cubic_indices_weights_1d_linear_continuation(y0,  wy, num_y)
                    elif len_offsets == 2:
                        x_idx, x_w = _linear_indices_weights_1d_linear_continuation(x0, wx, num_x)
                        y_idx, y_w = _linear_indices_weights_1d_linear_continuation(y0, wy, num_y)
                else:
                    # old clamp behavior
                    x_idx = np.empty(len_offsets, dtype=np.int64)
                    y_idx = np.empty(len_offsets, dtype=np.int64)
                    x_w = wx
                    y_w = wy

                    for a, ox in enumerate(offsets):
                        x_idx[a] = min(max(x0 + ox, 0), num_x - 1)
                    for b, oy in enumerate(offsets):
                        y_idx[b] = min(max(y0 + oy, 0), num_y - 1)

                k = 0
                for a in range(len_offsets):
                    for b in range(len_offsets):
                        idxs[i, k] = panel_id*num_x*num_y + x_idx[a]*num_y + y_idx[b]
                        weights[i, k] = x_w[a] * y_w[b]
                        k += 1
                        
            valid_sample_mask = np.sum(valid_sample_mask,axis=0).astype(np.bool_)
            return idxs, weights, valid_sample_mask, valid_sample_ids[1],nearest_data_id,n_weights_per_sample
        return build

    @staticmethod
    @njit()
    def _build_mean_fill_indices(data_shape:tuple)->NDArray:
        """
        Consider the case of a masked pixel (x below) and the
        8 surrounding pixels.
        | | | |
        | |x| |
        | | | |
        This method creates an array that for each data pixel x stores
        the indices of the surrounding pixels.

        At the bountary, e.g.
        | | |
        | |x|
        | | |
        or
        | |x|
        | | |
        
        the indices outsied the data range are mapped to the index of x,
        which is supposed to be masked.
        
        Args:
            gata_shape (tuple): Tuple containing the image_layout shape i.e. the data shape.
        
        Returns
            NDArray: Data Indices of neighboring pixels as described above.
        """
        n_panels,num_x,num_y = data_shape
        N = n_panels*num_x*num_y
        indices= np.empty((N,8),dtype=np.int64)
        for i in range(n_panels):
            for j in range(num_x):
                for k in range(num_y):
                    n = i*num_x*num_y+j*num_y+k                    
                    jmm = j - 1 if 0<j else j
                    jpp = j + 1 if j+1<num_x else j
                    kmm = k - 1 if 0<k else k
                    kpp = k + 1 if k+1<num_y else k
                    indices[n,:]=[
                        i*num_x*num_y+jmm*num_y+kmm,
                        i*num_x*num_y+jmm*num_y+k,
                        i*num_x*num_y+jmm*num_y+kpp,
                        i*num_x*num_y+j  *num_y+kmm,
                        i*num_x*num_y+j  *num_y+kpp,
                        i*num_x*num_y+jpp*num_y+kmm,
                        i*num_x*num_y+jpp*num_y+k,
                        i*num_x*num_y+jpp*num_y+kpp
                    ]
        return indices
    
    def build(self,
              sample_grid:SamplingGrid,
              layout:ImageLayout,
              policy:InterpolationPolicy
              ) -> InterpolationPlan:
        """ Creates InterpolationPlan instances.

        Args:
            sample_grid (SamplingGrid): sampling grid with points in pixel coordinates.
            layout (ImageLayout): Pixel panel layout.
            policy (InterpolationPolicy): Interpolation options.

        Returns:
            InterpolationPlan: Finished interpolation plan.
        """
        sample_points = sample_grid.ravel()
        data_shape = layout.data_shape

        
        error_on_overlap = False
        if policy.overlap_mode == InterpolationPolicy.OverlapMode.error:
            error_on_overlap = True

        xlim = (-0.5,layout.num_x  - 0.5) # xlim[0] <= x < xlim[1] defines valid x
        ylim = (-0.5,layout.num_y - 0.5) # ylim[0] <= y < ylim[1] defines valid y
        if policy.method == InterpolationPolicy.Method.linear:
            if policy.boundary == InterpolationPolicy.Boundary.reject:
                xlim = (0.0,layout.num_x  - 1.0)
                ylim =  (0.0,layout.num_y  - 1.0)
            linear_continuation = policy.boundary == policy.Boundary.extrapolate_linear
            out = self._build_linear(sample_points,data_shape,linear_continuation,error_on_overlap,xlim,ylim)
        elif policy.method == InterpolationPolicy.Method.cubic:
            if policy.boundary == InterpolationPolicy.Boundary.reject:
                xlim = (1.0,layout.num_x  - 2.0)
                ylim =  (1.0,layout.num_y  - 2.0)
            linear_continuation = policy.boundary == policy.Boundary.extrapolate_linear
            out = self._build_cubic(sample_points,data_shape,linear_continuation,error_on_overlap,xlim,ylim)
        else:
            raise ValueError(f'No build method known for interpolation method {policy.method}.')

        

        mean_fill_indices = None
        nearest_data_id = None
        if isinstance(policy.masking,InterpolationPolicy.Masking.MeanFill):
            mean_fill_indices = self._build_mean_fill_indices(layout.data_shape)
            nearest_data_id = out[4]
        plan=InterpolationPlan(
            weight_indices = out[0].flatten(),
            weight_values = out[1].flatten(),
            valid_sample_ids = out[3],
            valid_sample_mask = out[2],
            output_shape = sample_grid.output_shape,
            mean_fill_indices = mean_fill_indices,
            nearest_data_id = nearest_data_id,
            n_weights_per_sample = out[5]
        )
        return plan
