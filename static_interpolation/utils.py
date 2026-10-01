from math import ceil
import numpy as np
from numpy.typing import NDArray
import scipy.constants as constants
from extra_geom.base import DetectorGeometryBase
import h5py as h5
import traceback
def balanced_slices(n: int, max_chunk_size: int, start: int = 0) -> list[slice]:
    """
    Split a range of length n into the minimal number of contiguous chunks,
    each of size <= max_chunk_size, with chunk sizes differing by at most 1.

    Returns Python slice objects [slice(start0, stop0), slice(start1, stop1), ...].

    Parameters
    ----------
    n : int
        Number of items.
    max_chunk_size : int
        Upper bound S for each chunk size.
    start : int, default 0
        Start index for the slices. Use start=0 for normal Python indexing.
        Use start=1 if you literally want slices over labels 1..N.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be >= 1")
    if n == 0:
        return []

    # Minimal number of chunks
    k = ceil(n / max_chunk_size)

    # Balanced chunk sizes: r chunks of size q+1, the rest size q
    q, r = divmod(n, k)

    out = []
    i = start
    for j in range(k):
        size = q + (j < r)
        out.append(slice(i, i + size))
        i += size
    return out

def polar_scattering_coordinates_to_pixel_coordinates(polar_scatter_grid:NDArray,
                                                      p_origin:NDArray,
                                                      p_xdir:NDArray,
                                                      p_ydir:NDArray,
                                                      energy:float)->NDArray:
    r""" Converts Elwald's sphere coodinates to detector pixel coordinates

    Points on the Ewald's sphere $(q,\theta(q),\phi)$ given by $(q,\phi)$ are mapped to 
    pixel coordinates on a detector plane in real space whose origin is given by p_origin, whose x-direction
    is given by p_xdir and whose y-direction is given by p_ydir. Forthermore the length of p_ydir and p_xdir
    specify the pixel dimensions.
    
    Args:
        polar_scatter_grid (NDArray): (n_panels,...,2) Array containing (q,\phi) points in its last axis. q unit is [2*pi/m]
        p_origin (NDArray): (n_panels,3) Array containing Panel origin vectors in cartesian coodinates, unit is [m]
        p_xdir (NDArray): (n_panels,3)  Array containing pixel x-direction vectors in cartesian coodinates, unit is [m]
        p_ydir (NDArray): (n_panels,3) Array containing pixel y-direction vectors in cartesian coodinates, unit is [m]
        energy (float): X-ray energy in electron Volts [eV]

    Returns:
        NDArray: (n_panels,...,2) pixel coordinates.
    """
    
    # phi in spherical coords of pixel and scattering grid are the same   
    phi_pix = polar_scatter_grid[...,-1]

    # get scattering angles from energy and q of scattering grid
    h=constants.physical_constants['Planck constant in eV s'][0]
    c = constants.c
    scattering_angles = np.arcsin(polar_scatter_grid[...,0]*(h*c/(4*np.pi*energy)))*2

    # compute the scattering ray direction vector from the scattering angle and the phi angle
    direction_z = np.cos(scattering_angles)
    dir_xy = np.sin(scattering_angles)

    direction_x = np.cos(phi_pix)*dir_xy
    direction_y = np.sin(phi_pix)*dir_xy

    direction = np.stack([direction_x,direction_y,direction_z],axis = -1)
    direction /= np.linalg.norm(direction,axis=-1)[...,None]

    # compute normal of detector plane
    p_normal = np.cross(p_xdir,p_ydir)
    p_normal /= np.linalg.norm(p_normal)
    
    # compute intersection of scattering ray with pixel plane
    length = p_origin.dot(p_normal)/direction.dot(p_normal)
    
    # get pixel coordinate of sphericical scattering grid points.
    points = (length[...,None]*direction)
    
    # set origin to plane origin
    points -= p_origin
    
    # go to pixel coordinates
    x_size = np.linalg.norm(p_xdir)
    y_size = np.linalg.norm(p_ydir)
    points = (points @ (np.array([p_xdir/x_size**2,p_ydir/y_size**2]).T))
    return points

def get_max_q(geom:DetectorGeometryBase,detector_origin:NDArray,xray_energy:float,pad:bool = False)->float:
    """Computes the maximum momentum transfer in 2pi/meter for the given inputs.

    Args:
        geom (DetectorGeometryBase): Geometry describing the detector pixel positions.
        detector_origin (NDArray): Detector origin (x,y,z) relative to the sample in meters [m]
        xray_energy (float): Energy of the used X-rays in electron Volts[eV]
        pad (bool): Whether or not to pad the pixel center coordinates so that max_q coverse the entire pixel area not only the centers. 
    Returns:
        float: maximal momentum transfer covered by the geometry.
    """

    
    pixels=geom.get_pixel_positions(centre=True)
    pixels[...]+=detector_origin
    r = np.linalg.norm(pixels,axis = -1)
    
    if pad:
        pixel_size = geom.pixel_size
        diagonal = np.sqrt(2)*pixel_size
        r += 5*diagonal #(padding to cover the edge of the outer pixels)
    z = pixels[...,2]

    scatteringAngles = np.zeros(z.shape)
    neg_z = z<0
    zr = np.where(r==0,0,z/r)
    scatteringAngles[~neg_z] = np.arccos(zr[~neg_z])
    scatteringAngles[neg_z] = np.pi - np.arccos(-zr[neg_z])

    h=constants.physical_constants['Planck constant in eV s'][0]
    c = constants.c
    wavelength = (c*h)/xray_energy
    qs = 4*np.pi*np.sin(scatteringAngles/2)/wavelength
    #print(f'max_q = {np.max(qs)},wavelength = {wavelength},c = {c} h={h} energy = {xray_energy}')
    return np.max(qs)

# Ploting for examples in documentation
def plot_geom_native(data,geom,ax=None,figsize=None,**kwargs):
    ax = geom.plot_data(data,ax=ax,figsize=figsize,**kwargs)
    ax.invert_xaxis()
    return ax

def plot_polar(polar_data,figsize=None,**kwargs):
    from matplotlib import pyplot as plt
    # Plotting
    shape=polar_data.shape
    rVal=np.arange(0,shape[0])
    phiVal= np.linspace(0,2*np.pi,shape[1],endpoint=False)
    r,phi=np.meshgrid(rVal,phiVal)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
        
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    im = ax.pcolormesh(phi,r,np.swapaxes(polar_data,0,1),**kwargs)
    fig.colorbar(im)
    return fig,ax

def _generate_test_data(geom,n_images=100,seed=123456):
    rng = np.random.RandomState(123456)
    
    shape = (n_images,)+geom.expected_data_shape
    data = rng.random(shape)
    data*=np.arange(1,shape[1]+1)[None,:,None,None]
    pixpos = geom.get_pixel_positions()
    px, py, pz = np.moveaxis(pixpos, -1, 0)  # Separate x, y, z coordinates
    angle = np.arctan2(py, px)
    wedge_mask = (np.pi * 5/8 < angle) & (angle < np.pi * 7/8)
    masks = np.zeros(data.shape,bool)
    masks[:] = ~wedge_mask[None,...]
    return data,masks

def _plot_detector_test(data,masks,interpolation_result,interpolation_mask,geom,figsize=None):
    from matplotlib import pyplot as plt

    n_modules = geom.expected_data_shape[0]
    
    pltdat1 = data.copy()
    pltdat1[~masks]=np.nan
    pltdat2 = interpolation_result.copy()
    pltdat2[~interpolation_mask]=np.nan
    
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='polar')
    ax1.set_title('Data',fontsize=30)
    ax2.set_title("Polar samples on Ewald's sphere (cubic interp)",fontsize=30)
    plot_geom_native(pltdat1,geom,ax=ax1,cmap=None,vmin=0,vmax=n_modules)


    shape= interpolation_result.shape
    rVal=np.arange(0,shape[0])
    phiVal= np.linspace(0,2*np.pi,shape[1],endpoint=False)
    r,phi=np.meshgrid(rVal,phiVal)
    
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    
    im = ax2.pcolormesh(phi,r,np.swapaxes(pltdat2,0,1),vmin=0,vmax=n_modules)
    fig.colorbar(im)

    return fig



class HDF5_DB:
    h5=h5
    debug = False
    load_custom_types={}
    save_custom_types={}
    def __init__(self):
        save_custom_types={}
        HDF5_DB.save_custom_types = save_custom_types

        load_custom_types={}
        HDF5_DB.load_custom_types = load_custom_types
        
    @staticmethod
    def save(path,value_dict={},as_h5_object = False,**kwargs):     
        if isinstance(path,str):
            if as_h5_object:
                return h5.File(path, 'w', libver='latest')
            with h5.File(path, 'w') as h5file:
                HDF5_DB.recursively_save_dict_to_group(h5file,'/',value_dict)
        else:
            raise FileNotFoundError('Path Error, Abort loading "{}".'.format(path))
            
    @staticmethod
    def load(path,as_h5_object=False,h5_path=None,write_mode='r',**kwargs):
        try:
            if isinstance(path,str):
                if not as_h5_object:
                    if isinstance(h5_path,str):
                        entry_point = h5_path
                        if entry_point[-1] != '/':
                            entry_point+='/'
                    else:
                        entry_point = '/'
                    with h5.File(path, 'r') as h5file:
                        if h5_path is not None:
                            h5_obj = h5file[entry_point]
                            obj_type = h5_obj.attrs.get('type',False)
                            if isinstance(h5_obj,h5._hl.dataset.Dataset):
                                data = HDF5_DB.load_single_dataset(h5_obj)
                            elif obj_type=='list':
                                data = HDF5_DB._load_list(h5_obj,'','.')
                            elif obj_type=='tuple':
                                data = HDF5_DB._load_tuple(h5_obj,'','.')
                            else:
                                data=HDF5_DB.recursively_load_dict_from_group(h5file, entry_point)
                        else:
                            data=HDF5_DB.recursively_load_dict_from_group(h5file, entry_point)
                else:
                    data = h5.File(path, write_mode)
            else:
                raise FileNotFoundError('Path Error, Abort loading "{}".'.format(path))
        except Exception as e:            
            raise e
        return data
    
    @staticmethod
    def recursively_save_dict_to_group(h5_file, path, dic):
        custom_save_routines = HDF5_DB.save_custom_types
        for key,item in dic.items():
            key=str(key)
            try:                
                if isinstance(item,(complex,float,int,bytes,bool,np.number,np.bool_)):
                    h5_file[path].create_dataset(key,data=item)
                elif isinstance(item,(str,np.character)):
                    h5_file[path].create_dataset(key,data=item.encode('utf-8'))
                    h5_file[path+'/'+key].attrs['type']='str'
                elif isinstance(item,np.ndarray):
                    HDF5_DB.save_numpy_array(h5_file,path,key,item)            
                elif isinstance(item,h5.VirtualLayout):
                    h5_file[path].create_virtual_dataset(key, item)
                elif isinstance(item,(list,tuple)):
                    h5_file[path].create_group(key)
                    if isinstance(item,list):
                        h5_file[path+'/'+key].attrs['type']='list'
                    else:
                        h5_file[path+'/'+key].attrs['type']='tuple'                        
                    HDF5_DB.recursively_save_dict_to_group(h5_file, path + key + '/', {str(i):elem for i,elem in enumerate(item)})     
                elif isinstance(item, dict):
                    HDF5_DB._save_dict(h5_file,path,key,item)
                elif type(item).__name__ in custom_save_routines:
                    custom_save_routines[type(item).__name__](h5_file,path,key,item)
                elif item is None:
                    h5_file[path].create_dataset(key,data='None'.encode('utf-8'))
                    h5_file[path+'/'+key].attrs['type']='none'
                else:
                    raise ValueError('Cannot save {} type for key {}'.format(type(item),key))
            except Exception as e:
                print("Failed to save key {} and item type {} with error: \n {}".format(key,type(item),e))
                if self.debug:
                    traceback.print_exc()
    @staticmethod
    def load_single_dataset(item):
        custom_load_routines=HDF5_DB.load_custom_types
        item_type = item.attrs.get('type',False)
        if item_type == 'str':
            val = item[()].decode('utf-8')
        elif item_type == 'none':
            val = None
        elif item_type in custom_load_routines:
            val = custom_load_routines[item_type](item)
        else:                   
            val = item[()]
        return val
                
    @staticmethod
    def recursively_load_dict_from_group(h5_file, path):
        ans = {}
        for key, item in h5_file[path].items():
            item_type = item.attrs.get('type',False)
            if isinstance(item, h5._hl.dataset.Dataset):                  
                ans[key] = HDF5_DB.load_single_dataset(item)
            elif isinstance(item, h5._hl.group.Group):
                if item_type=='list':
                    ans[key] = HDF5_DB._load_list(h5_file,path,key)
                elif item_type=='tuple':
                    ans[key] = HDF5_DB._load_tuple(h5_file,path,key)
                else:
                    ans[key] = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        return ans
    
    @staticmethod
    def save_numpy_array(h5_file,path,key,array,meta_dict={}):
        dtype=array.dtype
        if dtype==np.dtype('complex'):
            h5_file[path].create_dataset(key,data=array.astype('<c16'))
        elif dtype == np.dtype('bool'):
            h5_file[path].create_dataset(key,data=array)
        elif 'str' in dtype.name:
            h5_file[path].create_dataset(key,data=array.astype('S'))
        else:
            h5_file[path].create_dataset(key,data=array)
            for attrib in meta_dict:
                h5_file[path+key+'/'].attrs.create(attrib,meta_dict[attrib])


    @staticmethod
    def _save_dict(h5_file,path,key,item):        
        h5_file[path].create_group(key)
        HDF5_DB.recursively_save_dict_to_group(h5_file, path + key + '/', item)
    @staticmethod
    def _load_tuple(h5_file,path,key):
        data = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        n_elements = len(data)
        return tuple(data[str(i)] for i in range(n_elements))
    @staticmethod
    def _load_list(h5_file,path,key):
        data = HDF5_DB.recursively_load_dict_from_group(h5_file, path + key + '/')
        n_elements = len(data)        
        return list(data[str(i)] for i in range(n_elements))
