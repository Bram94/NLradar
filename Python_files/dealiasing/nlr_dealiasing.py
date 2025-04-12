# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np
import time as pytime
import matplotlib.pyplot as plt

from nlr_functions import get_window_sum, get_window_indices



class DualPRFDealiasing():
    """Apply dealiasing using the cmean method for both detection of outliers and their correction
    """
    def __init__(self):
        pass
    
    
    
    def __call__(self, v_array, data_mask, azimuthal_res, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth = None, window_detection = [2, 2, 2, 2, 2], window_correction = [2, 2, 2, 2, 2], deviation_factor = 1.0, 
    n_it = 50, z_array=None, c_array = None, mask_all_nearzero_velocities = False, min_gates_correct = 0, plot_velocity_deviations = False):
        # t = pytime.time()
        """Parameters:
        - v_array: Array with velocities that need to be dealiasd
        - data_mask: Boolean array with shape of v_array that masks empty velocity bins.
        - vn_e: Extended Nyquist velocity
        - vn_l: Low Nyquist velocity
        - vn_h: High Nyquist velocity. Some radars switch halfway a radial from PRF, in which case every radial uses the same Nyquist velocity in the dual PRF procedure. 
        In this case should vn_l and vn_h be equal.
        - vn_first_azimuth: Should be 'l', 'h' or None, depending on whether the first radial is scanned with the low PRF, the high PRF, or whether it is unknown what PRF is 
        used for the first radial. If unknown, then it gets determined in the function 'determine_vn_even_and_odd_radials'.
        - window_detection: Window that is used for detection of aliased velocities, specified as a list of numbers. The length of the list specifies the number of radials 
        that the window comprises, and the elements in the list specify for each radial (centered around the central radial in which the radar bin is located) the number
        of radial bins in the window. 
        For each radial do 2*x+1 radial bins belong to the window, where x is the element in 'window'. The elements in window thus specify the number of radial bins
        above and below the central radius of the window. 
        The window should be symmetric around the central azimuth. This implies that the window should include an odd number of elements, and that the sequence of
        elements should be symmetric about the central value. Finally, the central element should not be smaller than the others.
        - window_correction: As window_detection, but now the window that is used for correction of velocities that were deemed to be aliased. 
        - deviation_factor: Velocities are deemed to be aliased when they differ by more than deviation_factor * (vn_l or vn_h) from the window-averaged reference velocity.
        - n_it: Maximum number of iterations of the algorithm. If convergence is reached sooner, then the algorithm is stopped. 
        - c_array: If an array with correlation coefficients is available for the radar, then it can be provided here. If given, then it is used for an attempt to mask clutter 
        bins with near-zero velocity (where in addition to a near-zero velocity also the correlation coefficient is low).
        - mask_all_nearzero_velocities: If no c_array is given, then it is still possible to mask all bins with near-zero velocity by setting this parameter to True. By not having the
        possibility to check the correlation coefficient, it is however possible that non-clutter bins are masked in this way.
        If mask_all_nearzero_velocities is False and if also no c_array is provided, then still near-zero velocities within a range of 25 km from the radar are
        masked. This is done since most clutter is located near the radar, so it is quite likely that near-zero velocities here represent in fact clutter.
        - min_gates_correct: A velocity that is deemed to be aliased is only corrected if there are at least min_gates_correct radar bins in window_correction that were not
        deemed to be aliased.
        - plot_velocity_deviations: By setting this to True, a plot with velocity deviations is created during calling of the function 'determine_vn_even_and_odd_radials'. 
        """
        
        self.data = v_array
        # Integer dtype instead of unsigned integer, since in calculations some operations can lead to negative values
        self.data_indices = np.array(np.meshgrid(np.arange(self.data.shape[0], dtype = 'int16'), np.arange(self.data.shape[1], dtype = 'int16'))).T
        
        if (c_array is None and mask_all_nearzero_velocities):
            data_mask = np.logical_or(data_mask, np.abs(v_array) < 1.5) #Also mask bins with wind speeds of less than 1.5 m/s, because these are often associated with clutter.
        elif not c_array is None:
            data_mask = np.logical_or(data_mask, (np.abs(v_array) < 1.5) & (c_array < 85.)) #Mask bins with wind speeds of less than 1.5 m/s only if the correlation 
            #coefficient (percentage) is low, which usually implies the presence of clutter.
        else:
            #Mask near-zero velocities within a range of 25 km from the radar.
            data_slantrange = np.tile(1+np.arange(self.data.shape[1]) * radial_res, (self.data.shape[0], 1))
            data_mask = np.logical_or(data_mask, (data_slantrange <= 25.) & (np.abs(v_array) < 1.5))
            
        before = data_mask.copy()
        if not z_array is None:
            data_mask |= z_array < 10
                
        self.data_mask = data_mask
        self.data_nonmask = ~data_mask
        self.azimuthal_res = azimuthal_res
        self.radial_res = radial_res
        self.vn_e, self.vn_l, self.vn_h = vn_e, vn_l, vn_h
        self.vn_first_azimuth = vn_first_azimuth
        self.window_detection = window_detection
        self.window_correction = window_correction
        self.deviation_factor = deviation_factor
        self.min_gates_correct = min_gates_correct
        self.n_it = n_it
        self.plot_velocity_deviations = plot_velocity_deviations
        
        self.dealias()
        
        if not z_array is None:
            self.data_mask = before.copy() | (z_array >= 10)
            self.data_nonmask = ~self.data_mask
            step = window_detection[1]-window_detection[0]
            self.window_detection = [window_detection[0]+step*j for j in range(len(window_detection))]
            self.window_detection += self.window_detection[:-1][::-1]
            step = window_correction[1]-window_correction[0]
            self.window_correction = [window_correction[0]+step*j for j in range(len(window_correction))]
            self.window_correction += self.window_correction[:-1][::-1]

            # self.window_detection = window_detection[:n]+[max(j*2, 1) for j in window_detection]+window_detection[-n:]
            # n = (len(window_correction)+1)//2
            # self.window_correction = window_correction[:n]+[max(j*2, 1) for j in window_correction]+window_correction[-n:]
            self.dealias()
            
        # print(self.n+1, pytime.time() - t,'t_dealias')
        return self.data
        
    
    def dealias(self):
        self.n_azi, self.n_rad = self.data.shape
        if self.vn_first_azimuth:
            vn_order = [self.vn_h, self.vn_l] if self.vn_first_azimuth == 'h' else [self.vn_l, self.vn_h]
            self.vn_radials = np.tile(vn_order, (self.n_azi+1)//2)[:self.n_azi].astype(self.data.dtype)
            self.vn_array = np.transpose(np.tile(self.vn_radials, (self.n_rad, 1))).astype(self.data.dtype)
            
        self.n_outliers_before = 1e8; self.n_outliers = 1e6
        # t = pytime.time()
        for self.n in range(self.n_it):
            if self.n == 0:                 
                self.phi = np.pi * self.data / self.vn_e
                self.sin_phi = np.sin(self.phi); self.cos_phi = np.cos(self.phi)
                self.sin_phi[self.data_mask] = self.cos_phi[self.data_mask] = 0.
            else:                
                self.phi[self.outliers] = np.pi * self.data[self.outliers] / self.vn_e
                phi_outliers = self.phi[self.outliers]
                self.sin_phi[self.outliers] = np.sin(phi_outliers); self.cos_phi[self.outliers] = np.cos(phi_outliers)
                              
            self.detect_outliers()
            if self.n_outliers_before <= self.n_outliers:
                break
            self.correct_outliers()
            
            # print(self.n, self.n_outliers, pytime.time() - t)

                                                    
    def detect_outliers(self):
        """Outlier detection step: First convert the velocities to a phase in [-pi, pi], by multiplying them by pi/vn_e. 
        Next, for each radar bin the circular mean of all phases in a certain window is calculated.
        Next, for each radar bin the minimum angular difference between the phase at that bin and the circular mean phase is calculated, and
        converted to a velocity difference. If this velocity difference is greater than the high/low Nyquist velocity corresponding to that bin, then
        that velocity is classified as an outlier.
        The reason for using phases instead of velocities is that it circumvents problems with extended Nyquist velocity aliasing."""
        if self.n == 0:
            self.update = self.data_nonmask
        else:
            # Keep in mind that the window sum cannot become larger than 1 for a boolean array. But that is not a problem here.
            self.update = self.data_nonmask & get_window_sum(self.outliers, self.window_detection)
        self.n_update = np.count_nonzero(self.update)
        
        phi_ref = self.calculate_reference_phase(self.update, self.n_update, self.window_detection)
        
        phi_diff = np.abs(self.phi[self.update] - phi_ref)
        corr = phi_diff > np.pi
        phi_diff[corr] = 2*np.pi - phi_diff[corr]
        v_diff = phi_diff * self.vn_e/np.pi
                    
        if self.vn_first_azimuth is None:
            self.v_diff_detection = np.zeros((self.n_azi, self.n_rad), dtype = self.data.dtype)
            self.v_diff_detection[self.update] = v_diff
            
            self.determine_vn_even_and_odd_radials()
        
        if self.n > 0: 
            self.outliers[:] = 0
        else:
            self.outliers = np.zeros((self.n_azi, self.n_rad), dtype = 'bool')
        outliers = (np.abs(v_diff) > self.deviation_factor * self.vn_array[self.update])
        self.outliers[self.update] = outliers 
        
        self.n_outliers_before = self.n_outliers
        self.n_outliers = np.count_nonzero(self.outliers)
        
    def correct_outliers(self):
        """Correction step: Add 2n*v_h or 2n*v_l to the velocity of the outliers, with n such that the difference between the resulting phase and 
        phi_ref is minimised."""
        self.sin_phi[self.outliers] = self.cos_phi[self.outliers] = 0.
        
        if self.min_gates_correct > 0:
            """Setting self.min_gates_correct to 0 is equal to stating that correction of velocities should always occur.
            If min_gates_correct = 0, then there might be some cases where the reference velocity is set to zero because there is no
            non-outlier velocity. This will likely however only occur in areas with very few velocities, in which case the data
            is usually not reliable/useful. For this reason min_gates_correct is set to zero by default, because this increases
            the speed of the algorithm."""
            n_nonoutliers = get_window_sum(np.logical_or(self.data_nonmask, ~self.outliers).astype('uint8'), self.window_correction)
            self.outliers[n_nonoutliers < self.min_gates_correct] = False
        
        #Calculate again the mean phase, but exclude the outliers from the averaging        
        phi_ref = self.calculate_reference_phase(self.outliers, self.n_outliers, self.window_correction)
        
        v_diff = (self.phi[self.outliers] - phi_ref)*self.vn_e/np.pi
        select = np.abs(v_diff) > self.vn_e
        v_diff[select] -= np.sign(v_diff[select])*2*self.vn_e
        
        vn_outliers = self.vn_array[self.outliers]
        correction_ints = np.round(v_diff/(2*vn_outliers))
        data_outliers = self.data[self.outliers] - 2*correction_ints*vn_outliers
        outside_extended_nyquist_interval = np.abs(data_outliers) > self.vn_e
        data_outliers_outside_nyq_int = data_outliers[outside_extended_nyquist_interval]
        data_outliers[outside_extended_nyquist_interval] = -np.sign(data_outliers_outside_nyq_int)*2*self.vn_e\
        + data_outliers_outside_nyq_int
        self.data[self.outliers] = data_outliers
        
    def calculate_reference_phase(self, select, n_select, window):
        if n_select < 12000 * (self.n_rad * self.n_azi) / (838 * 360):
            #Around this value of n_select becomes this method faster
            rows, cols = get_window_indices(self.data_indices[select], window, self.data.shape, periodic = 'rows')
            window_size = sum([j*2+1 for j in window])
            sin_phi_sum_update = np.sum(self.sin_phi[rows, cols].reshape((window_size, n_select)), axis = 0)
            cos_phi_sum_update = np.sum(self.cos_phi[rows, cols].reshape((window_size, n_select)), axis = 0)
            return np.arctan2(sin_phi_sum_update, cos_phi_sum_update)
        else:
            sin_phi_sum_detection, cos_phi_sum_detection = self.calculate_sin_cos_phi_sum(window) 
            return np.arctan2(sin_phi_sum_detection[select], cos_phi_sum_detection[select])
        
    def calculate_sin_cos_phi_sum(self, window):
        return get_window_sum(self.sin_phi, window), get_window_sum(self.cos_phi, window)
    
    def determine_vn_even_and_odd_radials(self):
        counts1, v = np.histogram(self.v_diff_detection[0:self.n_azi:2], bins = int(self.vn_e), range = (0, self.vn_e))
        counts2, v = np.histogram(self.v_diff_detection[1:self.n_azi:2], bins = int(self.vn_e), range = (0, self.vn_e))
        v_avg = (v[1:] + v[:-1])/2.
        
        # Exclude velocity differences close to 0, focus on differences that are twice the low/high Nyquist velocity
        select = v_avg > 1.5*self.vn_l
        self.vn_first_azimuth = 'h' if np.argmax(counts1[select]) > np.argmax(counts2[select]) else 'l'
        vn_order = [self.vn_h, self.vn_l] if self.vn_first_azimuth == 'h' else [self.vn_l, self.vn_h]
        self.vn_radials = np.tile(vn_order, (self.n_azi+1)//2)[:self.n_azi]
        self.vn_array = np.tile(self.vn_radials, (self.n_rad, 1)).T
        
        f = 1.9426
        v_avg *= f
        # for i, v in enumerate(v_avg):
        #     if 2/3*self.vn_l*f < v < 3*self.vn_l*f:
        #         print(int(round(v)), counts1[i], counts2[i])
        if self.plot_velocity_deviations:
            plt.figure()
            plt.plot(v_avg, counts1)
            plt.plot(v_avg, counts2)
            plt.xlim([0, 3*self.vn_l*f])
            select = v_avg > 2/3*self.vn_l*f
            plt.ylim([0, max([counts1[select].max(), counts2[select].max()])])
            plt.show()




        
da = DualPRFDealiasing()

def apply_dual_prf_dealiasing(v_array, data_mask, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth = None, window_detection = None, window_correction = None, deviation_factor = 1.0, n_it = 50, z_array=None, c_array = None, mask_all_nearzero_velocities = False):
    """If window_detection and window_correction are not specified, then they are given below. Those window sizes are based upon the assumption that there can be many outliers,
    such that the window should not be too small. If the number of outliers is always small for a particular radar, then the window size should be smaller, to prevent undesired
    smoothing of the velocity field.
    Another assumption is that the low and high Nyquist velocity are about 30/40 kts resp. If the true Nyquist velocities are lower/higher, then the window size can be chosen
    smaller/larger. 
    """
    azimuthal_res = 360/v_array.shape[0]
    ratio = azimuthal_res / radial_res
    if window_detection is None:
        if ratio <= 1: window_detection = [1,2,2,2,1]
        elif ratio <= 2: window_detection = [1,2,3,2,1]
        elif ratio < 5: window_detection = [1,3,5,3,1]
        else: window_detection = [2,4,6,4,2]
    if window_correction is None:
        if ratio <= 1: window_correction = [1,2,1]
        elif ratio <= 2: window_correction = [2,3,2]
        elif ratio < 5: window_correction = [3,5,3]
        else: window_correction = [4,6,4]
                
    return da(v_array, data_mask, azimuthal_res, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth, window_detection, window_correction, deviation_factor, n_it, z_array, c_array, mask_all_nearzero_velocities)