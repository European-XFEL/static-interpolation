from numba import njit
import numpy as np
from numpy.typing import NDArray
from typing import Callable
from dataclasses import dataclass
from .data_structures import SamplingGrid,ImageLayout,SamplingMeshRegular
from .config import InterpolationPolicy
import abc
from math import floor

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
        out_shape (tuple[int,...]): Shape of the fineal interpolated image.
        mean_fill_indices (None|NDArray): 3x3 neighbor indices of each pixel on each data plane.
        nearest_data_id (None|NDArray): (n_valid,) pixel index of the nearest data point for each sampling point.
    """
    weight_indices: NDArray      #(M) | (n_valid, stencil_size)
    weight_values: NDArray      #(M) | (n_valid, stencil_size)
    n_weights_per_sample: NDArray #(n_valid)
    valid_sample_ids: NDArray   # (n_valid,) Ids of samples that lie within the data planes.
    valid_sample_mask: NDArray   # (n_samples,) Mask over all samples that is true if sample lies in one of the data planes.
    out_shape: tuple
    mean_fill_indices: None|NDArray # (n_data_points,8)
    nearest_data_id: None|NDArray # (n_valid)

class InterpolationPlannerBase(abc.ABC):
    @abc.abstractmethod
    def build(self,
              mapped_grid:SamplingGrid,
              layout:ImageLayout,
              policy:InterpolationPolicy
              ) -> InterpolationPlan:
        pass
    
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

class InterpolationPlanner(InterpolationPlannerBase):
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
              mapped_grid:SamplingGrid,
              layout:ImageLayout,
              policy:InterpolationPolicy
              ) -> InterpolationPlan:
        """ Creates InterpolationPlan instances.

        Args:
            mapped_grid (SamplingGrid): sampling grid with points in pixel coordinates.
            layout (ImageLayout): Pixel panel layout.
            policy (InterpolationPolicy): Interpolation options.

        Returns:
            InterpolationPlan: Finished interpolation plan.
        """
        sample_points = mapped_grid.ravel()
        logical_shape = layout.logical_shape

        
        error_on_overlap = False
        if policy.overlap_mode == InterpolationPolicy.OverlapMode.error:
            error_on_overlap = True

        xlim = (-0.5,layout.num_x_logical  - 0.5) # xlim[0] <= x < xlim[1] defines valid x
        ylim = (-0.5,layout.num_y_logical - 0.5) # ylim[0] <= y < ylim[1] defines valid y
        if policy.method == InterpolationPolicy.Method.linear:
            if policy.boundary == InterpolationPolicy.Boundary.reject:
                xlim = (0.0,layout.num_x_logical  - 1.0)
                ylim =  (0.0,layout.num_y_logical  - 1.0)
            linear_continuation = policy.boundary == policy.Boundary.extrapolate_linear
            out = self._build_linear(sample_points,logical_shape,linear_continuation,error_on_overlap,xlim,ylim)
        elif policy.method == InterpolationPolicy.Method.cubic:
            if policy.boundary == InterpolationPolicy.Boundary.reject:
                xlim = (1.0,layout.num_x_logical  - 2.0)
                ylim =  (1.0,layout.num_y_logical  - 2.0)
            linear_continuation = policy.boundary == policy.Boundary.extrapolate_linear
            out = self._build_cubic(sample_points,logical_shape,linear_continuation,error_on_overlap,xlim,ylim)
        else:
            raise ValueError(f'No build method known for interpolation method {policy.method}.')

        

        weight_indices = layout.convert_logical_to_data_ids(out[0].ravel()).ravel()
        mean_fill_indices = None
        nearest_data_id = None
        if isinstance(policy.masking,InterpolationPolicy.Masking.MeanFill):
            mean_fill_indices = self._build_mean_fill_indices(layout.data_shape)
            mean_fill_indices = layout.convert_logical_to_data_ids(mean_fill_indices)
            nearest_data_id = layout.convert_logical_to_daata_ids(out[4])
        plan=InterpolationPlan(
            weight_indices = weight_indices,
            weight_values = out[1].flatten(),
            valid_sample_ids = out[3],
            valid_sample_mask = out[2],
            out_shape = mapped_grid.out_shape,
            mean_fill_indices = mean_fill_indices,
            nearest_data_id = nearest_data_id,
            n_weights_per_sample = out[5]
        )
        return plan




@njit
def pixels_on_line_segment(x0:float, y0:float, x1:float, y1:float) -> NDArray[np.int64]:
    """Return the integer grid cells crossed by a 2D line segment.

    Uses the Amanatides-Woo voxel traversal algorithm in 2D to compute the
    sequence of grid cells intersected by the open line segment from
    `(x0, y0)` to `(x1, y1)`.

    The function applies an open-segment convention by nudging both endpoints
    inward by one ULP using `np.nextafter`. This avoids ambiguous inclusion of
    cells when the segment lies exactly on grid boundaries.

    Args:
        x0: X-coordinate of the segment start point.
        y0: Y-coordinate of the segment start point.
        x1: X-coordinate of the segment end point.
        y1: Y-coordinate of the segment end point.

    Returns:
        A NumPy array of shape `(n, 2)` and dtype `np.int64`, where each row
        contains the `(ix, iy)` indices of a crossed grid cell in traversal
        order.

    Notes:
        - If the segment is degenerate (start and end points are identical),
          an empty array is returned.
        - If the entire segment lies exactly on a vertical or horizontal grid
          boundary, an empty array is returned because no cell interior is hit.
    """    
    # Degenerate segment
    if x0 == x1 and y0 == y1:
        return np.empty((0, 2), np.int64)

    # Entire segment lies on a grid boundary -> no cell interior is hit
    if y0 == y1 and y0 == floor(y0):
        return np.empty((0, 2), np.int64)
    if x0 == x1 and x0 == floor(x0):
        return np.empty((0, 2), np.int64)

    # Open-segment convention: nudge endpoints inward by 1 ulp
    xs = np.nextafter(x0, x1)
    ys = np.nextafter(y0, y1)
    xe = np.nextafter(x1, x0)
    ye = np.nextafter(y1, y0)

    dx = xe - xs
    dy = ye - ys

    ix,iy,ix1,iy1  = int(floor(xs)),int(floor(ys)),int(floor(xe)),int(floor(ye))
    
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # With the line parmetrized as
    #  (x(t),y(t)) = (xs,ys) + t(dx,dy),0≤t≤1
    #
    # tdx: increment in t when crossing each additional vertical grid line
    # tdy: increment in t when crossing each additional horizontal grid line
    # tmx: next t at which the line hits a vertical grid line
    # tmy: next t at which the line hits a horizontal grid line
    if dx != 0.0:
        tdx = abs(1.0 / dx)
        tmx = ((ix + 1 - xs) / dx) if dx > 0 else ((xs - ix) / -dx)
    else:
        tdx = np.inf
        tmx = np.inf

    if dy != 0.0:
        tdy = abs(1.0 / dy)
        tmy = ((iy + 1 - ys) / dy) if dy > 0 else ((ys - iy) / -dy)
    else:
        tdy = np.inf
        tmy = np.inf

    
    # Upper bound on number of crossed cells
    max_crossed_cells = abs(ix1 - ix) + abs(iy1 - iy) + 1
    out = np.empty((max_crossed_cells,2), np.int64)
    n = 0

    out[n, 0] = ix
    out[n, 1] = iy
    
    if max_crossed_cells == 1:
        return out
    else:
        for n in range(1, max_crossed_cells):
            if tmx < tmy:
                ix += step_x
                tmx += tdx
            elif tmy < tmx:
                iy += step_y
                tmy += tdy
            else:
                ix += step_x
                iy += step_y
                tmx += tdx
                tmy += tdy
    
            out[n, 0] = ix
            out[n, 1] = iy
            
            if ix == ix1 and iy == iy1:
                break
        return out[:n+1]
@njit
def poly_area(poly:NDArray[np.float64]) -> float:
    r"""Calculate the area of a polygon using the shoelace formula.

    $$ 2A = \begin{vmatrix}
            x_1 & x_2 \\
            y_1 & y_2
            \end{vmatrix} + \begin{vmatrix}
            x_2 & x_3 \\
            y_2 & y_3
            \end{vmatrix} + \ldots + \begin{vmatrix}
            x_{n} & x_1 \\
            y_{n} & y_1
            \end{vmatrix} $$
    
    The vertices must be ordered consecutively along the polygon boundary.
    
    Args:
        poly: A 2D array of shape (n, 2) where each row
                    represents the (x, y) coordinates of a vertex.
                    

    Returns:
       The area of the polygon.
    """
    n=len(poly)
    if n < 3:
        return 0.0

    area = 0.0
    x_prev = poly[n - 1, 0]
    y_prev = poly[n - 1, 1]

    for i in range(n):
        x = poly[i, 0]
        y = poly[i, 1]
        area += x_prev * y - y_prev * x
        x_prev = x
        y_prev = y
    area = 0.5*abs(area)
    return area    

@njit(inline="always")
def _clip_halfspace(
    in_vertices: NDArray[np.float64],
    n_vertices: int,
    out_vertices: NDArray[np.float64],
    axis: int,
    boundary_value: float,
    keep_direction: float
) -> int:
    """Clip a polygon against an axis-aligned half-space.

    This is a specialized Sutherland-Hodgman-style clipping routine for axis-aligned
    half-spaces i.e. clopping against x,y >= keep_direction*(coord-boundary_value).

    The clipping condition is

        keep_direction * (in_vertices[:,axis] - boundary_value) >= 0

    Args:
        in: In_Verticesut vertex array of shape ``(N, 2)`` containing polygon
            coordinates.
        n_vertices: Number of valid vertices stored in ``in_vertices``. Only the first
             ``n_vertices`` rows are processed.
        out_vertices: Output array of shape ``(M, 2)`` into which the clipped polygon
            vertices are written.
        axis: Coordinate axis used for clipping. Use ``0`` for the
            x-axis and ``1`` for the y-axis.
        boundary_value: Coordinate value of the clipping boundary.
        keep_direction: Side of the boundary to retain. Use ``+1.0`` to keep
            vertices with coordinate greater than or equal to ``boundary_value``,
            and ``-1.0`` to keep vertices with coordinate less than or equal
            to ``boundary_value``.

    Returns:
        The number of output vertices written to ``out_vertices``.
    """
    if n_vertices == 0:
        return 0

    other_axis = 1 - axis

    start_coord = in_vertices[n_vertices - 1, axis]
    start_other = in_vertices[n_vertices - 1, other_axis]
    start_inside = keep_direction * (start_coord - boundary_value) >= 0.0

    n_out = 0

    for vertex_idx in range(n_vertices):
        end_coord = in_vertices[vertex_idx, axis]
        end_other = in_vertices[vertex_idx, other_axis]
        end_inside = keep_direction * (end_coord - boundary_value) >= 0.0

        if start_inside != end_inside:
            delta_coord = end_coord - start_coord
            t = 0.0 if delta_coord == 0.0 else (boundary_value - start_coord) / delta_coord

            out_vertices[n_out, axis] = boundary_value
            out_vertices[n_out, other_axis] = start_other + t * (end_other - start_other)
            n_out += 1

        if end_inside:
            out_vertices[n_out, axis] = end_coord
            out_vertices[n_out, other_axis] = end_other
            n_out += 1

        start_coord = end_coord
        start_other = end_other
        start_inside = end_inside

    return n_out
@njit
def quad_bounding_box(quad):
    """Computes the bounding box of a quad.

    Args:
        quad (ndarray): An array of shape (4, 2) containing quad vertices 
            in clockwise (CW) or counter-clockwise (CCW) order.

    Returns:
        ndarray: An array of shape (4,) storing the bounding box in the 
            format [min_x, max_x, min_y, max_y].
    """
    minx = quad[0, 0]
    maxx = quad[0, 0]
    miny = quad[0, 1]
    maxy = quad[0, 1]

    for i in range(4):
        x = quad[i, 0]
        y = quad[i, 1]
    
        if x < minx:
            minx = x
        elif x > maxx:
            maxx = x
    
        if y < miny:
            miny = y
        elif y > maxy:
            maxy = y
    return minx,maxx,miny,maxy
@njit
def clipped_area_quad_unit_square(quad:NDArray[np.float64],
                                  buf0:NDArray[np.float64],
                                  buf1:NDArray[np.float64]) -> float:
    """ Compute the intersection area of a convex quad with the unit square.

    The input vertices in "quad" must be ordered consecutively along the boundary of the quadrilateral.
    Temporary scratch buffers are used to store intermediate polygon vertices during clipping.

    Args:
        quad: A ``(4, 2)`` array of ``float64`` values containing the
            quadrilateral vertices as ``(x, y)`` coordinates in boundary order.
        buf0: A scratch array of shape ``(8, 2)`` used to store intermediate
            clipped vertices.
        buf1: A scratch array of shape ``(8, 2)`` used as an additional
            temporary buffer during alternating clipping steps.

    Returns:
        The area of the intersection between the quadrilateral and the unit
        square as a non-negative ``float64``.

    Notes:
        - Returns ``0.0`` if the quadrilateral lies entirely outside the unit
          square.
        - Returns the quadrilateral area directly if the quadrilateral lies
          entirely inside the unit square.
        - The scratch buffers are modified in place.
        - Because clipping a convex quadrilateral against the unit square can
          produce up to 8 vertices, the scratch buffers must be large enough to
          hold at least 8 points.
    """

    minx,maxx,miny,maxy = quad_bounding_box(quad)
    inside_all = (minx>0) and (maxx<1) and (miny>0) and (maxy<1)
    buf0[:4] = quad

    # trivial reject
    if maxx < 0.0 or minx > 1.0 or maxy < 0.0 or miny > 1.0:
        return 0.0

    # trivial accept
    #print(inside_all)
    if inside_all:
        return poly_area(buf0[:4])

    n = 4
    n = _clip_halfspace(buf0, n, buf1, 0, 0.0,  1.0)  # clip x >= 0
    if n == 0:
        return 0.0
    n = _clip_halfspace(buf1, n, buf0, 0, 1.0,  -1.0)  # clip x <= 1
    if n == 0:
        return 0.0
    n = _clip_halfspace(buf0, n, buf1, 1, 0.0,  1.0)  #clip  y >= 0
    if n == 0:
        return 0.0
    n = _clip_halfspace(buf1, n, buf0, 1, 1.0,  -1.0)  # clipy <= 1
    if n == 0:
        return 0.0
    buf0[n:,:] = np.nan
    return poly_area(buf0[:n])

@njit
def _get_valid_cell_mask(vertices:NDArray[np.float64],                     
                         xlim:tuple,
                         ylim:tuple,
                         error_on_overlap:bool)->NDArray[np.bool_]:
    """Computes the mask of cells that lie within the data range

    Args:
        vertices (NDArray[np.float64]): (n_panels,num_x,num_y,2) Array of mesh vertices in data index coordinates.
        xlim (tuple): data range x limits
        ylim (tuple): data range y limits
        error_on_overlap (bool): Flag that 

    Returns:
        NDArray[np.bool_]: [description]
    """

    xmin, xmax = xlim
    ymin, ymax = ylim
    n_panels,num_x,num_y = vertices.shape[:-1]
    vertex_claimed = np.zeros((num_x, num_y), dtype=np.bool_)    
    valid_cell_mask =  np.full((n_panels,num_x-1,num_y-1),True,dtype=np.bool_)

    for p in range(n_panels):
        for i in range(num_x):
            for j in range(num_y):
                x = vertices[p, i, j, 0]
                y = vertices[p, i, j, 1]

                vertex_valid = False
                inside_data_range = xmin <= x < xmax and ymin <= y < ymax

                if inside_data_range:
                    if not vertex_claimed[i,j]:
                        vertex_claimed[i,j] = True
                        vertex_valid = True
                    elif error_on_overlap:
                        raise ValueError(
                            "There are sampling points that belong to multiple data panels."
                        )
                    
                if not vertex_valid:
                    if i > 0 and j > 0:
                        valid_cell_mask[p, i - 1, j - 1] = False
                    if i > 0 and j < num_y - 1:
                        valid_cell_mask[p, i - 1, j] = False
                    if i < num_x - 1 and j > 0:
                        valid_cell_mask[p, i, j - 1] = False
                    if i < num_x - 1 and j < num_y - 1:
                        valid_cell_mask[p, i, j] = False
    return valid_cell_mask
@njit
def _compute_bboxes_total_area_and_component_limit(valid_cell_ids,vertices):
    n = len(valid_cell_ids[0])
    cell_areas = np.zeros(n,np.float64)
    bboxes = np.zeros(cell_areas.shape+(4,),np.float64)
    poly = np.zeros((4,2),np.float64)
    n_components_limit = 0
    bb_max_dx = 0
    bb_max_dy = 0
    for i in range(n):
        panel_id = valid_cell_ids[0][i]
        xid =  valid_cell_ids[1][i]
        yid =  valid_cell_ids[2][i]
        
        poly[:2]= vertices[panel_id,xid,yid:yid+2]
        poly[2]= vertices[panel_id,xid+1,yid+1]
        poly[3]= vertices[panel_id,xid+1,yid]
        
        cell_areas[i] = poly_area(poly)
        
        minx,maxx,miny,maxy = quad_bounding_box(poly)
        n_components_limit  += (2+int(maxx-minx))*(2+int(maxy-miny))
        bboxes[i,0] = minx
        bboxes[i,1] = maxx
        bboxes[i,2] = miny
        bboxes[i,3] = maxy
        
        dx = maxx-minx
        dy = maxy-miny 
        if bb_max_dx < dx:
            bb_max_dx = dx
        if bb_max_dy < dy:
            bb_max_dy = dy
    return bboxes,cell_areas,n_components_limit,bb_max_dx,bb_max_dy
@njit(inline="always")
def point_in_convex_quad(px, py, quad):
    """Test whether a point lies inside or on the boundary of a convex quadrilateral.

    Args:
        px: X-coordinate of the point.
        py: Y-coordinate of the point.
        quad: A ``(4, 2)`` array of ``float64`` values containing the
            quadrilateral vertices in boundary order.

    Returns:
        ``True`` if the point lies inside the quadrilateral or on its boundary,
        otherwise ``False``.

    Notes:
        - The quadrilateral is assumed to be convex.
        - The vertex order may be either CW or CCW, but must be consistent.
        - No numerical tolerance is applied; sign checks are performed using
          exact floating-point comparisons.
    """
    sign = 0.0

    for i in range(4):
        x0 = quad[i, 0]
        y0 = quad[i, 1]
        x1 = quad[(i + 1) % 4, 0]
        y1 = quad[(i + 1) % 4, 1]
        
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        
        if cross > 0.0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0.0:
            if sign > 0:
                return False
            sign = -1
            
    return True

@njit(inline="always")
def _flat_weight_index(panel_id: int, pix_x: int, pix_y: int, num_x: int, num_y: int) -> int:
    """Return the flattened output index for one pixel."""
    return panel_id * num_x * num_y + pix_x * num_y + pix_y
@njit(inline="always")
def _load_shifted_quad(vertices, panel_id: int, xid: int, yid: int, poly: NDArray[np.float64]) -> None:
    """Load one quad and shift coordinates by +0.5 so pixels become unit squares."""
    poly[:2] = vertices[panel_id, xid, yid:yid + 2]
    poly[2] = vertices[panel_id, xid + 1, yid + 1]
    poly[3] = vertices[panel_id, xid + 1, yid]

    poly[:, 0] += 0.5
    poly[:, 1] += 0.5
@njit(inline="always")
def _candidate_pixel_range_from_bbox(
    minx: float, maxx: float, miny: float, maxy: float
) -> tuple[int, int, int, int, int, int]:
    """Compute the candidate pixel index range from a shifted-coordinate bbox."""
    ix0 = int(np.floor(minx))
    ix1 = int(np.ceil(maxx)) - 1
    iy0 = int(np.floor(miny))
    iy1 = int(np.ceil(maxy)) - 1

    nx = ix1 - ix0 + 1
    ny = iy1 - iy0 + 1
    return ix0, ix1, iy0, iy1, nx, ny
@njit
def _accumulate_boundary_weights(
    poly: NDArray[np.float64],
    ix0: int,
    iy0: int,
    nx: int,
    ny: int,
    panel_id: int,
    num_x: int,
    num_y: int,
    cell_area: float,
    visited_stamp: NDArray[np.int64],
    stamp: int,
    buf0: NDArray[np.float64],
    buf1: NDArray[np.float64],
    local_poly: NDArray[np.float64],
    weight_indices: NDArray[np.int64],
    weight_values: NDArray[np.float64],
    write_offset: int,
) -> int:
    """Accumulate weights for pixels whose interiors are crossed by the quad boundary.

    Args:
        poly: Quad vertices in shifted coordinates.
        ix0: Minimum candidate pixel x-index.
        iy0: Minimum candidate pixel y-index.
        nx: Number of candidate pixels in x.
        ny: Number of candidate pixels in y.
        panel_id: Panel index.
        num_x: Output x dimension.
        num_y: Output y dimension.
        cell_area: Area of the quadrilateral cell.
        visited_stamp: Stamp array marking boundary-visited pixels in bbox-local coordinates.
        stamp: Stamp value unique to the current cell.
        buf0: Scratch buffer of shape (8, 2) for clipping.
        buf1: Scratch buffer of shape (8, 2) for clipping.
        local_poly: Scratch buffer of shape (4, 2) for translating the quad into pixel-local coordinates.
        weight_indices: Output flat pixel indices.
        weight_values: Output normalized weights.
        write_offset: Start offset into the output arrays.
    
    Returns:
        Number of weights written.
    """
    pixel_count = 0

    prev_x = poly[3, 0]
    prev_y = poly[3, 1]

    for j in range(4):
        curr_x = poly[j, 0]
        curr_y = poly[j, 1]

        pixels = pixels_on_line_segment(prev_x, prev_y, curr_x, curr_y)

        prev_x = curr_x
        prev_y = curr_y

        for pid in range(len(pixels)):
            pix_x = pixels[pid, 0]
            pix_y = pixels[pid, 1]

            lx = pix_x - ix0
            ly = pix_y - iy0

            if lx < 0 or lx >= nx or ly < 0 or ly >= ny:
                continue

            if visited_stamp[lx, ly] == stamp:
                continue
            visited_stamp[lx, ly] = stamp

            local_poly[:, 0] = poly[:, 0] - pix_x
            local_poly[:, 1] = poly[:, 1] - pix_y

            area = clipped_area_quad_unit_square(local_poly, buf0, buf1)
            if area <= 0.0:
                continue

            out_idx = write_offset + pixel_count
            weight_indices[out_idx] = _flat_weight_index(panel_id, pix_x, pix_y, num_x, num_y)
            weight_values[out_idx] = area / cell_area
            pixel_count += 1

    return pixel_count
@njit
def _accumulate_interior_weights(
    poly: NDArray[np.float64],
    ix0: int,
    ix1: int,
    iy0: int,
    iy1: int,
    panel_id: int,
    num_x: int,
    num_y: int,
    cell_area: float,
    visited_stamp: NDArray[np.int64],
    stamp: int,
    weight_indices: NDArray[np.int64],
    weight_values: NDArray[np.float64],
    write_offset: int,
) -> int:
    """Accumulate weights for unvisited pixels whose centers lie inside the quad.

    This assumes that every partially covered pixel was already marked as visited
    during boundary traversal. Therefore, any unvisited pixel is either fully
    inside or fully outside the quadrilateral, so a center test is sufficient.

    Args:
        poly: Quad vertices in shifted coordinates.
        ix0: Minimum candidate pixel x-index.
        ix1: Maximum candidate pixel x-index.
        iy0: Minimum candidate pixel y-index.
        iy1: Maximum candidate pixel y-index.
        panel_id: Panel index.
        num_x: Output x dimension.
        num_y: Output y dimension.
        cell_area: Area of the quadrilateral cell.
        visited_stamp: Stamp array marking boundary-visited pixels.
        stamp: Stamp value unique to the current cell.
        weight_indices: Output flat pixel indices.
        weight_values: Output normalized weights.
        write_offset: Start offset into the output arrays.

    Returns:
        Number of weights written.
    """
    pixel_count = 0

    for pix_x in range(ix0, ix1 + 1):
        lx = pix_x - ix0
        for pix_y in range(iy0, iy1 + 1):
            ly = pix_y - iy0

            if visited_stamp[lx, ly] == stamp:
                continue

            px = pix_x + 0.5
            py = pix_y + 0.5

            if point_in_convex_quad(px, py, poly):
                out_idx = write_offset + pixel_count
                weight_indices[out_idx] = _flat_weight_index(panel_id, pix_x, pix_y, num_x, num_y)
                weight_values[out_idx] = 1.0 / cell_area
                pixel_count += 1

    return pixel_count
@njit
def _compute_weights_area(vertices, valid_cell_ids, cell_areas, bboxes, largest_bbox_shape, num_x, num_y):
    """Compute normalized area weights from mesh cells to image pixels.

    Vertex coordinates are assumed to use the convention that integer values lie
    at pixel centers. Internally, coordinates are shifted by +0.5 so that pixel
    `(ix, iy)` corresponds to the unit square `[ix, ix+1] x [iy, iy+1]`.

    For each valid quadrilateral cell:
      1. Traverse the quad boundary and compute exact clipped overlap areas for
         boundary-intersecting pixels.
      2. For remaining candidate pixels inside the bbox, test whether the pixel
         center lies inside the quad; if yes, assign full overlap area.

    Args:
        vertices: Array of shape `(n_panels, nx, ny, 2)` containing mesh vertices.
        valid_cell_ids: Tuple of arrays `(panel_ids, x_ids, y_ids)` identifying valid cells.
        cell_areas: Array of cell areas with shape `(n_valid_cells,)`.
        bboxes: Array of shape `(n_valid_cells, 4)` storing `[min_x, max_x, min_y, max_y]`
            in the original pixel-center coordinate system.
        largest_bbox_shape: Maximum candidate bbox shape `(max_nx, max_ny)`.
        num_x: Output x dimension.
        num_y: Output y dimension.

    Returns:
        Tuple `(weight_indices, weight_values, n_weights_per_sample)`.
    """
    max_n_weights = largest_bbox_shape[0] * largest_bbox_shape[1]
    n_cells = len(cell_areas)

    weight_indices = np.zeros(n_cells * max_n_weights, np.int64)
    weight_values = np.zeros(n_cells * max_n_weights, np.float64)
    n_weights_per_sample = np.zeros(n_cells, np.int64)

    buf0 = np.zeros((8, 2), dtype=np.float64)
    buf1 = np.zeros((8, 2), dtype=np.float64)
    local_poly = np.zeros((4, 2), dtype=np.float64)
    poly = np.zeros((4, 2), dtype=np.float64)

    visited_stamp = np.zeros(largest_bbox_shape, np.int64)

    n_total = 0

    for i in range(n_cells):
        stamp = i + 1

        panel_id = valid_cell_ids[0][i]
        xid = valid_cell_ids[1][i]
        yid = valid_cell_ids[2][i]

        cell_area = cell_areas[i]
        if cell_area == 0.0:
            n_weights_per_sample[i] = 0
            continue

        _load_shifted_quad(vertices, panel_id, xid, yid, poly)

        # Shift bbox by +0.5 to match shifted quad coordinates
        minx = bboxes[i, 0] + 0.5
        maxx = bboxes[i, 1] + 0.5
        miny = bboxes[i, 2] + 0.5
        maxy = bboxes[i, 3] + 0.5

        ix0, ix1, iy0, iy1, nx, ny = _candidate_pixel_range_from_bbox(minx, maxx, miny, maxy)

        # Entire quad lies in a single pixel
        if nx == 1 and ny == 1:
            weight_indices[n_total] = _flat_weight_index(panel_id, ix0, iy0, num_x, num_y)
            weight_values[n_total] = 1.0
            n_weights_per_sample[i] = 1
            n_total += 1
            continue

        count_boundary = _accumulate_boundary_weights(
            poly,
            ix0,iy0,nx,ny,
            panel_id,num_x,num_y,
            cell_area,visited_stamp,stamp,
            buf0,buf1,local_poly,
            weight_indices,weight_values,
            n_total
        )

        count_interior = _accumulate_interior_weights(
            poly,
            ix0,ix1,iy0,iy1,
            panel_id,num_x, num_y,
            cell_area,visited_stamp,stamp,
            weight_indices, weight_values,
            n_total + count_boundary
        )

        total_count = count_boundary + count_interior
        n_weights_per_sample[i] = total_count
        n_total += total_count

    return weight_indices[:n_total], weight_values[:n_total], n_weights_per_sample

class InterpolationPlannerMeshRegular(InterpolationPlannerBase):
    def build(self,
              mapped_grid:SamplingMeshRegular,
              layout:ImageLayout,
              policy:InterpolationPolicy
              ) -> InterpolationPlan:
        if not isinstance(mapped_grid,SamplingMeshRegular):
            raise ValueError(f"mapped_grid must be of type SamplingMeshRegular but type {type(mapped_grid)} was provided.")

        # get mesh cells that lie in the data range

        xlim = (-0.5,layout.num_x_logical-0.5)
        ylim = (-0.5,layout.num_y_logical-0.5)
        error_on_overlap = policy.overlap_mode == policy.OverlapMode.error
        valid_cell_mask = _get_valid_cell_mask(mapped_grid.points,xlim,ylim,error_on_overlap)
        
        valid_cell_ids = np.nonzero(valid_cell_mask)
        
        # precompute bounding boxes and totat cell area and n_component_limit
        bboxes,cell_areas,n_components_limit,bb_max_dx,bb_max_dy = _compute_bboxes_total_area_and_component_limit(valid_cell_ids,mapped_grid.points)
        largest_bbox_shape = (2+int(bb_max_dx),2+int(bb_max_dy))

        # compute weights
        weight_indices,weight_values,n_weights_per_sample = _compute_weights_area(mapped_grid.points,
                                                                                  valid_cell_ids,
                                                                                  cell_areas,
                                                                                  bboxes,
                                                                                  largest_bbox_shape,
                                                                                  layout.num_x_logical,
                                                                                  layout.num_y_logical)

        weight_indices = layout.convert_logical_to_data_ids(weight_indices.ravel()).ravel()
        valid_cell_ids_flat = valid_cell_ids[1]*(mapped_grid.points.shape[2]-1)+valid_cell_ids[2]
        # instanciate interpolation plan
        plan = InterpolationPlan(weight_indices,
                                 weight_values,
                                 n_weights_per_sample,
                                 valid_cell_ids_flat,
                                 valid_cell_mask,
                                 mapped_grid.out_shape,
                                 None,
                                 None)
        return plan
