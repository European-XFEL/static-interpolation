from numba import njit
import numpy as np
from numpy.typing import NDArray
from functools import partial

from .planning import InterpolationPlan
from .data_structures import ImageLayout
from .config import InterpolationPolicy

#----------------------
#   Plan execution
#  CPU/GPU variants
class InterpolationEngine:
    """ Base class for all Engines execute interpolations based on an InterpolationPlan an ImageLayout and an InterpolationPolicy instance.

    Attributes:
        plan (InterpolationPlan): plan storing all constants for the interpolation
        layout (ImageLayout): image layout defining the input data shape
        policy (InterpolationPolicy): options
    """
    def __init__(self,plan:InterpolationPlan,layout:ImageLayout,policy:InterpolationPolicy):
        self.plan = plan
        self.layout = layout
        self.policy = policy
        
    def __call__(
        self,
        data: np.ndarray,
        masks: np.ndarray | None = None,
        out: np.ndarray | None = None,
        out_masks: np.ndarray | None = None,
    )->None|NDArray|tuple[NDArray,NDArray]:
        """ Do the interpolation for a given data chunk.

        Args:
            data (NDArray): (bunch,n_panels,num_x,num_y) input data bunch.
            masks (NDArray | None): (bunch,n_panels,num_x,num_y) Optional masks for input data (good values = True bad = False)
            out (NDArray | None): self.plan.out_shape Optional output array to store interpolation results to.
            out_masks (NDArray | None): self.plan.out_shape Optional output array to store the mask resulting from the interpolation.

        Returns:
            None|NDArray|tuple(NDArray,NDArray): If only data is given a single NDArray is returned, If in addition out is specified then None is returned.
                                                 If only data and mask are given a tuple of two NDArrays storing the interpolation result for the data and the mask is returned.
                                                 If all inputs are given None is returned.
        """
        
        create_out = out is None
        create_out_masks = out_masks is None
        masks_provided = masks is not None
        
        if masks_provided and (create_out != create_out_masks):
            raise ValueError(f'If Masks is given the output arguments (out, out_masks) have to be both None or both given. User provided type(out) = {type(out)} type(out_masks)={type(out_masks)}')

        if masks_provided and data.shape!=masks.shape:
            raise ValueError(f"Provided data and mask need tha have the same shape but data shape={data.shape} and masks shape = {masks.shape}")

        ravel = self.layout.ravel
        normalize = self.layout.normalize

        
        data = normalize(data)
        data_flat = ravel(data)
        
        n = data_flat.shape[0]
        expected_out_shape = (n,) + self.plan.out_shape

        # check for correct shapes
        if out is not None and out.shape != expected_out_shape:
            raise ValueError(f"out has shape {out.shape}, expected {expected_out_shape}")
        if masks_provided and not create_out_masks and out_masks.shape != expected_out_shape:
            raise ValueError(f"out_masks has shape {out_masks.shape}, expected {expected_out_shape}")            
        
        if create_out:
            out = np.zeros(
                expected_out_shape,
                dtype=np.result_type(data.dtype, np.float64)
            )
        out_flat = out.reshape(n,-1)
        
        
        if masks_provided:
            masks = normalize(masks)
            masks_flat = ravel(masks)
            if create_out_masks:
                out_masks = np.zeros(out.shape,dtype = np.bool_)
            out_masks_flat = out_masks.reshape(n,-1)
                
            self._apply_masked(data_flat,masks_flat,out_flat,out_masks_flat)
            
            if create_out:
                return out,out_masks
        else:            
            self._apply_unmasked(data_flat,out_flat)
            if create_out:
                return out
            
    def _apply_masked(self,data: np.ndarray,masks: np.ndarray,out: np.ndarray,out_masks: np.ndarray):
        pass
    def _apply_unmasked(self,data: np.ndarray,out: np.ndarray):
        pass

@njit(parallel=False)
def _apply_unmasked_njit(imgs_flat:NDArray, out:NDArray,weight_values:NDArray,weight_indices:NDArray,valid_sample_ids:NDArray)->None:
    """
    Kernel that computes the simplest type of interpolation.
    Each point in out is computed by taking the corresponding set of weights,
    multiplying them with their associated data values and summing them up.
    This routine assumes a constant number of weights per output point, i.e. weight_values.shape = (n_valid,stencil_size)
    where stencil_size is the constant number of weights per output point.
    """ 
    M = len(imgs_flat)
    N = len(valid_sample_ids)
    K = weight_indices.shape[1]
    
    for i in range(M):
        img = imgs_flat[i]
        out_i = out[i]
        for j in range(N):
            wj=weight_values[j]
            ij = weight_indices[j]
            s = 0.0
            for k in range(K):
                s += wj[k] * img[ij[k]]
            out_i[valid_sample_ids[j]] = s
@njit(parallel=False)
def _apply_unmasked_flat_njit(imgs_flat, out,weight_values,weight_indices,valid_sample_ids,n_weights_per_sample)->None:
    """
    Same as _apply_unmasked_njit but allowing unequal numbers of weights for each sampling point.
    The number of weights for a given sampling point is provided via n_weights_per_sample.    
    """ 
    M = len(imgs_flat)
    N = len(valid_sample_ids)    
    for i in range(M):
        img = imgs_flat[i]
        out_i = out[i]
        seen=0
        for j in range(N):
            K = n_weights_per_sample[j]
            start = seen
            end = seen + K
            seen = end
            
            s = 0.0
            for k in range(start,end):
                s +=  weight_values[k]*img[weight_indices[k]]
            out_i[valid_sample_ids[j]] = s        
@njit(parallel=False)
def _apply_masked_strict_njit(imgs_flat,masks_flat,out,out_mask,weight_values,weight_indices,valid_sample_ids):
    """
    Computes masked interpolation for uniform number of weights per sample.
    This routine implements strict mask handling.
    This means an output point is masked if any of the necessary data points to compute it is masked.
    """
    M = len(imgs_flat)
    N = len(valid_sample_ids)
    K = weight_indices.shape[1]
    for i in range(M):
        img = imgs_flat[i]
        mask = masks_flat[i]
        out_i = out[i]
        out_mask_i = out_mask[i]
        for j in range(N):#prange(N):
            oid = valid_sample_ids[j]
            wj=weight_values[j]
            ij = weight_indices[j]
            s = 0.0
            valid = True
            for k in range(K):
                pix_id = ij[k]
                if not mask[pix_id]:
                    # computation for j'th point contains a masked data point
                    # abort computation and consider sampling point as masked.
                    valid = False
                    s = 0.0
                    break
                s += wj[k] * img[pix_id]
            out_i[oid]=s
            out_mask_i[oid]=valid
@njit(parallel=False)
def _apply_masked_strict_flat_njit(imgs_flat,
                                   masks_flat,
                                   out,
                                   out_mask,
                                   weight_values,
                                   weight_indices,
                                   valid_sample_ids,
                                   n_weights_per_sample):
    """
    Same as _apply_masked_strict_njit but allowing unequal numbers of weights for each sampling point.
    The number of weights for a given sampling point is provided via n_weights_per_sample.    
    """
    M = len(imgs_flat)
    N = len(valid_sample_ids)
    for i in range(M):
        img = imgs_flat[i]
        mask = masks_flat[i]
        out_i = out[i]
        out_mask_i = out_mask[i]
        
        seen = 0
        for j in range(N):
            K = n_weights_per_sample[j]
            start = seen
            end = seen + K
            seen = end
            
            oid = valid_sample_ids[j]
            s = 0.0
            valid = True
            for k in range(start,end):
                pix_id = weight_indices[k]
                if not mask[pix_id]:
                    valid = False
                    s = 0.0
                    break
                s += weight_values[k] * img[pix_id]
            out_i[oid]=s
            out_mask_i[oid]=valid            
@njit()
def _mean_fill_all_njit(img,mask,tmp_img,tmp_mask,mean_fill_indices,max_masked)->None:
    """
    Takes an input image and tries to fill in masked values by computing the mean of its surrounding 8 data values marked by o:
    |o|o|o|
    |o|x|o|
    |o|o|o|
    if more than max_masked of these 8 values are masked the filling fails and the values stays masked.
    """
    N = len(img)
    for i in range(N):
        if mask[i]:
            tmp_img[i]= img[i]
            tmp_mask[i] = True
            continue
        neighbourhood_ids = mean_fill_indices[i]
        
        n_valid = 0
        mean = 0.0
        for j in range(8):
            pixid = neighbourhood_ids[j]
            valid = mask[pixid]
            n_valid += int(valid)
            if valid:
                mean += img[pixid]
        tmp_mask[i] = n_valid>=(8-max_masked)
        tmp_img[i] = mean/float(n_valid) if tmp_mask[i] else 0
@njit()
def _apply_masked_mean_fill_njit(imgs_flat,
                                 masks_flat,
                                 out,
                                 out_mask,
                                 weight_values,
                                 weight_indices,
                                 valid_sample_ids,
                                 mean_fill_indices,
                                 nearest_data_id,
                                 mask_nearest,
                                 max_masked):
    """
    Computes masked interpolation for uniform number of weights per sample.
    This routine implements mean_fill mask handling.
    Masked data values are first tried to fill using _mean_fill_all_njit.
    If the option mask_nearest is set, then sampling points whose nearest data point is a masked data value will always be masked
    independent on whether the masked data value was meaan_filled or not.
    
    Remaining masked data values are treated by the strict rule.
    """
    tmp_img = np.empty(imgs_flat.shape[1:],dtype = imgs_flat.dtype)
    tmp_mask = np.empty(masks_flat.shape[1:],dtype=masks_flat.dtype)
    M = len(imgs_flat)
    N = len(valid_sample_ids)
    K = weight_indices.shape[1]
    for i in range(M):
        img = imgs_flat[i]
        mask = masks_flat[i]
        out_i = out[i]
        out_mask_i = out_mask[i]
        
        _mean_fill_all_njit(img,mask,tmp_img,tmp_mask,mean_fill_indices,max_masked)
        for j in range(N):
            oid = valid_sample_ids[j]            
            if mask_nearest and (not mask[nearest_data_id[j]]):
                # if mask_nearest is enabled and the nearest data value is masked.
                # mask this output and continue to the next sample.
                out_i[oid]=0
                out_mask_i[oid]=False
                continue
            
            s = 0.0
            valid = True
            for k in range(K):
                img_id = weight_indices[j, k]
                if not tmp_mask[img_id]:
                    valid = False
                    s = 0.0
                    break
                s += weight_values[j, k] * tmp_img[img_id]
            out_i[oid]=s
            out_mask_i[oid]=valid
@njit()
def _apply_masked_mean_fill_flat_njit(imgs_flat,
                                      masks_flat,
                                      out,
                                      out_mask,
                                      weight_values,
                                      weight_indices,
                                      valid_sample_ids,
                                      n_weights_per_sample,
                                      mean_fill_indices,
                                      nearest_data_id,
                                      mask_nearest,
                                      max_masked):
    """
    Same as _apply_masked_mean_fill_njit but allowing unequal number of weights for each sampling point.
    The number of weights for a given sampling point is provided via n_weights_per_sample.    
    """
    tmp_img = np.empty(imgs_flat.shape[1:],dtype = imgs_flat.dtype)
    tmp_mask = np.empty(masks_flat.shape[1:],dtype=masks_flat.dtype)
    M = len(imgs_flat)
    N = len(valid_sample_ids)
    for i in range(M):
        img = imgs_flat[i]
        mask = masks_flat[i]
        out_i = out[i]
        out_mask_i = out_mask[i]
        
        seen = 0
        _mean_fill_all_njit(img,mask,tmp_img,tmp_mask,mean_fill_indices,max_masked)
        for j in range(N):
            K = n_weights_per_sample[j]
            start = seen
            end = seen + K
            seen = end
            
            oid = valid_sample_ids[j]
            if mask_nearest and (not mask[nearest_data_id[j]]):
                # if mask_nearest is enabled and the nearest data value is masked.
                # mask this output and continue to the next sample.
                out_i[oid]=0
                out_mask_i[oid]=False
                continue
            
            s = 0.0
            valid = True
            for k in range(start,end):
                pix_id = weight_indices[k]
                if not tmp_mask[pix_id]:
                    valid = False
                    s = 0.0
                    break
                s += weight_values[k] * tmp_img[pix_id]
            out_i[oid]=s
            out_mask_i[oid]=valid
            
class NumbaEngine(InterpolationEngine):
    """ Numba interpolation engine
    That ties toghether the above defined numba kernels.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        plan = self.plan
        opt = self.policy.masking

        unique_n_weights_per_sample = np.unique(self.plan.n_weights_per_sample)
        self.uniform_number_of_wights = len(unique_n_weights_per_sample)==1
        if self.uniform_number_of_wights:
            weight_values = plan.weight_values.reshape(-1,unique_n_weights_per_sample[0])
            weight_indices = plan.weight_indices.reshape(-1,unique_n_weights_per_sample[0])
        else:
            weight_values = plan.weight_values
            weight_indices = plan.weight_indices
        
        if self.uniform_number_of_wights:
            self.unmasked_kernel = partial(_apply_unmasked_njit,
                                           weight_values = weight_values,
                                           weight_indices = weight_indices,
                                           valid_sample_ids = plan.valid_sample_ids
                                           )
        else:
            self.unmasked_kernel = partial(_apply_unmasked_flat_njit,
                                           weight_values = weight_values,
                                           weight_indices = weight_indices,
                                           valid_sample_ids = plan.valid_sample_ids,
                                           n_weights_per_sample = plan.n_weights_per_sample
                                           )
        if isinstance(self.policy.masking,InterpolationPolicy.Masking.Strict):
            if self.uniform_number_of_wights:                
                self.masked_kernel = partial(_apply_masked_strict_njit,
                                             weight_values = weight_values,
                                             weight_indices = weight_indices,
                                             valid_sample_ids = plan.valid_sample_ids
                                             )
            else:
                self.masked_kernel = partial(_apply_masked_strict_flat_njit,
                                             weight_values = weight_values,
                                             weight_indices = weight_indices,
                                             valid_sample_ids = plan.valid_sample_ids,
                                             n_weights_per_sample = plan.n_weights_per_sample
                                             )
        elif isinstance(self.policy.masking,InterpolationPolicy.Masking.MeanFill):
            if self.uniform_number_of_wights:
                self.masked_kernel = partial(_apply_masked_mean_fill_njit,
                                             weight_values = weight_values,
                                             weight_indices = weight_indices,
                                             valid_sample_ids = plan.valid_sample_ids,
                                             mean_fill_indices = plan.mean_fill_indices,
                                             nearest_data_id = plan.nearest_data_id,
                                             mask_nearest = opt.mask_nearest,
                                             max_masked = opt.max_masked
                                             )
            else:
                self.masked_kernel = partial(_apply_masked_mean_fill_flat_njit,
                                             weight_values = weight_values,
                                             weight_indices = weight_indices,
                                             valid_sample_ids = plan.valid_sample_ids,
                                             n_weights_per_sample = plan.n_weights_per_sample,
                                             mean_fill_indices = plan.mean_fill_indices,
                                             nearest_data_id = plan.nearest_data_id,
                                             mask_nearest = opt.mask_nearest,
                                             max_masked = opt.max_masked
                                             )
        else:
            raise ValueError(f'{self.policy.masking} is an unknown masking type for the numba interpolation engine.')
        
    def _apply_unmasked(self,data,out):
        self.unmasked_kernel(data,out)
    def _apply_masked(self,data,masks,out,out_masks):
        self.masked_kernel(data,masks,out,out_masks)
