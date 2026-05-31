import pytest
import numpy as np
from scipy import constants
from scipy.interpolate import RegularGridInterpolator,CubicHermiteSpline

from static_interpolation import (
    data_structures,utils,coordinate_mappers,engines,planning,interpolators,config
)

class TestImageLayout:
    @pytest.mark.parametrize("shape", [(10,7,4,6),(4,3,6),(23,6)])
    def test_normalize_raises_on_not_c_contiguous(self,shape):
        # important to check since this could fail silently in InterpolationEngines
        with pytest.raises(ValueError, match="C_CONTIGUOUS"):
            layout = data_structures.ImageLayout.from_shape(shape[-3:])
            # make uncontiguouse data
            data = np.arange(np.prod(shape[:-1])*12).reshape(shape[:-1]+(12,))[...,:6]
            normalized = layout.normalize(data)

class TestSamplingGrid:
    def test_init_raises_on_not_c_contiguous(self):
        # important to check since this could silently cause errors in InterpolationPlanner
        with pytest.raises(ValueError, match="C_CONTIGUOUS"):
            points = np.random.rand(13,8,7,2)[:,::2]
            sampling_grid = data_structures.SamplingGrid(points = points ,n_panels=13)

class TestUtils:
    '''Check Momentum space to pixel grid conversion.'''
    def wavelength_from_energy(self,energy_eV: float) -> float:
        h = constants.physical_constants["Planck constant in eV s"][0]
        return constants.c * h / energy_eV
    
    def q_from_scattering_angle(self,angle_rad: float, energy_eV: float) -> float:
        wavelength = wavelength_from_energy(energy_eV)
        return 4 * np.pi * np.sin(angle_rad / 2) / wavelength

    def polar_to_pix_for_axis_aligned_panels_matches_expectation(self):
        energy = 9000.0
        distance = 0.5
        angle = 0.2  # scattering angle in radians
        q = self.q_from_scattering_angle(angle, energy)

        polar_grid = np.array([
            [q, 0.0],
            [q, np.pi / 2],
            [q, np.pi],
        ])

        p_origin = np.array([0.0, 0.0, distance])
        p_xdir = np.array([1.0, 0.0, 0.0])
        p_ydir = np.array([0.0, 1.0, 0.0])

        out = utils.polar_scattering_coordinates_to_pixel_coordinates(
            polar_grid, p_origin, p_xdir, p_ydir, energy
        )

        radius_on_plane = distance * np.tan(angle)
        expected = np.array([
            [radius_on_plane, 0.0], 
            [0.0, radius_on_plane],
            [-radius_on_plane, 0.0],
        ])

        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)

class TestStaticInterpolatorAgainstScipy:
    '''
    Low level integration test.
    This is also the main test for correctnes of the interpolation procedure.
    I trust that the scipy implementation is correct. 
    '''
    def test_linear_interpolation(self):
        # test that StaticInterpolator gives same result scipy linear interpolation
        rng = np.random.RandomState(123456)
        n_data = 10
        n_samples = 24
        data = rng.random((n_data,n_data))
        
        x_points = np.arange(n_data)
        y_points = np.arange(n_data)
        reg = RegularGridInterpolator([x_points,y_points],data)


        sample_points = np.stack((rng.random(n_samples)*9,rng.random(n_samples)*9),axis=-1)
        ref_out = reg(sample_points)
        
        layout = data_structures.ImageLayout.from_shape((n_data,n_data))
        samples = data_structures.SamplingGrid(points = sample_points[None,...],n_panels=1)
        opt = config.InterpolationPolicy()
        opt.method = opt.Method.linear
        reg2 = interpolators.StaticInterpolator(samples,layout=layout,policy=opt)
        assert np.allclose(ref_out,reg2(data)),"Mismatch between linear interpolation of scipy and StaticInterpolator output."

    def catmull_rom_tangents_1d(self,v):
        """
        Catmull-Rom-like tangents for 1D samples v at coordinates t.
        Uses centered differences inside, one-sided at the ends.
        (equivalent to linear_extension at boundary)
        assumes unit square pixels. 
        """
        m = np.empty_like(v, dtype=float)
        # inside
        m[1:-1] = (v[2:] - v[:-2]) / 2 #(t[2:] - t[:-2])
        # starting edge
        m[0] = (v[1] - v[0]) #/ (t[1] - t[0])
        # closing edge
        m[-1] = (v[-1] - v[-2]) #/ (t[-1] - t[-2])
        return m
    def cubic_catmull_rom_scipy_version(self,img, xqs, yqs):
        """
        Separable 2D Catmull-Rom interpolation on a regular grid.
        First interpolate along x in each row, then along y.
        Using CubicHermiteSpline from scipy.
        """
        out = []
        for xq,yq in zip(xqs,yqs):
            # Step 1: interpolate each row at xq
            row_values = np.empty(img.shape[1], dtype=float)
            for j in range(img.shape[1]):
                row = img[j, :]
                dx = self.catmull_rom_tangents_1d(row)
                spline_x = CubicHermiteSpline(np.arange(img.shape[0]), row, dx)
                row_values[j] = spline_x(xq)

            # Step 2: interpolate those intermediate values along y at yq
            dy = self.catmull_rom_tangents_1d(row_values)
            spline_y = CubicHermiteSpline(np.arange(img.shape[1]), row_values, dy)
            out.append(float(spline_y(yq)))
        return np.array(out)
    def test_cubic_interpolation(self):
        # test that StaticInterpolator gives same result scipy based cubic Catmull-Rom interpolation
        rng = np.random.RandomState(123456)
        n_data = 10
        n_samples = 24
        data = rng.random((n_data,n_data))
        
        sample_points = np.stack((rng.random(n_samples)*(n_data-1),rng.random(n_samples)*(n_data-1)),axis=-1)
        ref_out = self.cubic_catmull_rom_scipy_version(data,sample_points[...,1],sample_points[...,0])
        
        layout = data_structures.ImageLayout.from_shape((n_data,n_data))
        samples = data_structures.SamplingGrid(points = sample_points[None,...],n_panels=1)
        opt = config.InterpolationPolicy()
        opt.method = opt.Method.cubic
        opt.boundary = opt.Boundary.extrapolate_linear
        reg2 = interpolators.StaticInterpolator(samples,layout=layout,policy=opt)

        assert np.allclose(ref_out,reg2(data)),"Mismatch between linear interpolation of scipy and StaticInterpolator output."

class TestNumbaEngine:
    '''
    Just checking that flat and nonflat numba kernels produce the same result.
    '''
    def _make_mean_fill_indices_for_9_pixels(self):
        """
        Build a (9, 8) neighbour table for _mean_fill_all_njit.
        
        For each pixel i, the "neighbourhood" is simply all other 8 pixels.
        This is enough for unit testing the kernel logic.
        """
        ids = np.arange(9, dtype=np.int64)
        return np.array([ids[ids != i] for i in ids], dtype=np.int64)

    @pytest.fixture
    def kernel_inputs(self):
        rng = np.random.RandomState(123456)
        imgs_flat  = rng.random((2,9))
        masks_flat = rng.random((2,9)) > 0.3

        valid_sample_ids = np.array([1, 3, 5], dtype=np.int64)

        # Uniform weights: 3 output samples, each with 3 contributing pixels
        weight_indices = np.arange(9).reshape(3,3)
        weight_values = rng.random((3,3))

        mean_fill_indices = self._make_mean_fill_indices_for_9_pixels()
        nearest_data_id = np.array([1, 3, 6], dtype=np.int64)

        return {
            "imgs_flat": imgs_flat,
            "masks_flat": masks_flat,
            "valid_sample_ids": valid_sample_ids,
            "weight_indices": weight_indices,
            "weight_values": weight_values,
            "n_weights_per_sample":np.full(*weight_values.shape, dtype=np.int64),
            "mean_fill_indices": mean_fill_indices,
            "nearest_data_id": nearest_data_id,
            "out_width": 6,
        }
    
    def test_apply_unmasked_flat_matches_nonflat(self,kernel_inputs):
        imgs_flat = kernel_inputs["imgs_flat"]
        weight_values = kernel_inputs["weight_values"]
        weight_indices = kernel_inputs["weight_indices"]
        valid_sample_ids = kernel_inputs["valid_sample_ids"]
        n_weights_per_sample = kernel_inputs["n_weights_per_sample"]
    
        out_nonflat = np.zeros((imgs_flat.shape[0], kernel_inputs["out_width"]), dtype=np.float64)
        out_flat = np.zeros_like(out_nonflat)
    
        engines._apply_unmasked_njit(
            imgs_flat,
            out_nonflat,
            weight_values,
            weight_indices,
            valid_sample_ids,
        )
        engines._apply_unmasked_flat_njit(
            imgs_flat,
            out_flat,
            weight_values.ravel(),
            weight_indices.ravel(),
            valid_sample_ids,
            n_weights_per_sample,
        )
    
        assert np.allclose(out_flat, out_nonflat),"Mismatch between flat and nonflat unmasked njit kernel."
    def test_apply_masked_strict_flat_matches_nonflat(self,kernel_inputs):
        imgs_flat = kernel_inputs["imgs_flat"]
        masks_flat = kernel_inputs["masks_flat"]
        weight_values = kernel_inputs["weight_values"]
        weight_indices = kernel_inputs["weight_indices"]
        valid_sample_ids = kernel_inputs["valid_sample_ids"]
    
        n_weights_per_sample = kernel_inputs["n_weights_per_sample"]
    
        out_nonflat = np.zeros((imgs_flat.shape[0], kernel_inputs["out_width"]), dtype=np.float64)
        out_mask_nonflat = np.zeros_like(out_nonflat, dtype=np.bool_)
    
        out_flat = np.zeros_like(out_nonflat)
        out_mask_flat = np.zeros_like(out_mask_nonflat)
    
        engines._apply_masked_strict_njit(
            imgs_flat,
            masks_flat,
            out_nonflat,
            out_mask_nonflat,
            weight_values,
            weight_indices,
            valid_sample_ids,
        )
        engines._apply_masked_strict_flat_njit(
            imgs_flat,
            masks_flat,
            out_flat,
            out_mask_flat,
            weight_values.ravel(),
            weight_indices.ravel(),
            valid_sample_ids,
            n_weights_per_sample,
        )
        
        assert np.allclose(out_flat, out_nonflat),"Result value mismatch between flat and nonflat masked strict njit routines."
        assert np.array_equal(out_mask_flat, out_mask_nonflat),"Result mask mismatch between flat and nonflat masked strict njit routines."
        
    def test_mean_fill_all_njit_fills_single_masked_pixel(self):
        img = np.arange(1,10,dtype = np.float64)
        img[4] = -999
        mask = np.array([True, True, True, True, False, True, True, True, True], dtype=np.bool_)
    
        tmp_img = np.empty_like(img)
        tmp_mask = np.empty_like(mask)
        mean_fill_indices = self._make_mean_fill_indices_for_9_pixels()
    
        engines._mean_fill_all_njit(
            img,
            mask,
            tmp_img,
            tmp_mask,
            mean_fill_indices,
            max_masked=0,
        )
    
        # Pixel 4 should be filled with the mean of the other 8 pixels
        expected_mean = np.mean([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0])
        assert tmp_mask[4]
        assert tmp_img[4] == pytest.approx(expected_mean)
    
        # Unmasked pixels should remain unchanged
        assert np.allclose(tmp_img[mask], img[mask])
        assert np.array_equal(tmp_mask[mask], np.full(8,True))
        
    @pytest.mark.parametrize("mask_nearest", [False, True])
    def test_apply_masked_mean_fill_flat_matches_nonflat(self,kernel_inputs, mask_nearest):
        imgs_flat = kernel_inputs["imgs_flat"]
        masks_flat = kernel_inputs["masks_flat"]
        weight_values = kernel_inputs["weight_values"]
        weight_indices = kernel_inputs["weight_indices"]
        valid_sample_ids = kernel_inputs["valid_sample_ids"]
    
        n_weights_per_sample = kernel_inputs["n_weights_per_sample"]
    
        mean_fill_indices = kernel_inputs["mean_fill_indices"]
        nearest_data_id = kernel_inputs["nearest_data_id"]
    
        out_nonflat = np.zeros((imgs_flat.shape[0], kernel_inputs["out_width"]), dtype=np.float64)
        out_mask_nonflat = np.zeros_like(out_nonflat, dtype=np.bool_)
    
        out_flat = np.zeros_like(out_nonflat)
        out_mask_flat = np.zeros_like(out_mask_nonflat)
    
        engines._apply_masked_mean_fill_njit(
            imgs_flat,
            masks_flat,
            out_nonflat,
            out_mask_nonflat,
            weight_values,
            weight_indices,
            valid_sample_ids,
            mean_fill_indices,
            nearest_data_id,
            mask_nearest,
            max_masked=1,
        )
        engines._apply_masked_mean_fill_flat_njit(
            imgs_flat,
            masks_flat,
            out_flat,
            out_mask_flat,
            weight_values.ravel(),
            weight_indices.ravel(),
            valid_sample_ids,
            n_weights_per_sample,
            mean_fill_indices,
            nearest_data_id,
            mask_nearest,
            max_masked=1,
        )
    
        assert np.allclose(out_flat, out_nonflat)
        assert np.array_equal(out_mask_flat, out_mask_nonflat)
