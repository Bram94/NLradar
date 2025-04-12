# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:02:06 2018

@author: bramv

"""

import numpy as np
import time as pytime
#import matplotlib.pyplot as plt

from nlr_functions import get_window_sum



class DualPRFDealiasing():
    """Apply dealiasing using the cmean method for both detection of outliers and their correction
    """
    def __init__(self):
        pass
    
    
    
    def __call__(self, v_array, data_mask, azimuthal_res, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth = None,\
    window_detection = [2, 2, 2, 2, 2], window_correction = [2, 2, 2, 2, 2], deviation_factor = 1.0, n_it = 5, c_array = None, mask_nearzero_velocities = False, min_gates_correct = 0, plot_velocity_deviations = False):
        t = pytime.time()
        """Parameters:
        - v_array: Array with velocities that need to be dealiased
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
        - n_it: Number of iterations of the algorithm
        - c_array: If an array with correlation coefficients is available for the radar, then it can be provided here. If given, then it is used for an attempt to mask clutter 
        bins with near-zero velocity (where in addition to a near-zero velocity also the correlation coefficient is low).
        - mask_nearzero_velocities: If no c_array is given, then it is still possible to mask bins with near-zero velocity by setting this parameter to True. By not having the
        possibility to check the correlation coefficient, it is however possible that non-clutter bins are masked in this way.
        - min_gates_correct: A velocity that is deemed to be aliased is only corrected if there are at least min_gates_correct radar bins in window_correction that were not
        deemed to be aliased.
        - plot_velocity_deviations: By setting this to True, a plot with velocity deviations is created during calling of the function 'determine_vn_even_and_odd_radials'. 
        """
        
        self.data = v_array
        
        if c_array is None and mask_nearzero_velocities:
            data_mask = np.logical_or(data_mask, np.abs(v_array) < 1.5) #Also mask bins with wind speeds of less than 1.5 m/s, because these are often associated with clutter.
        elif not c_array is None:
            data_mask = np.logical_or(data_mask, (np.abs(v_array) < 1.5) & (c_array < 85)) #Mask bins with wind speeds of less than 1.5 m/s only if the correlation 
            #coefficient (percentage) is low, which usually implies the presence of clutter.
                
        self.data_mask = data_mask
        self.data_nonmask = data_mask == False
        self.azimuthal_res = azimuthal_res
        self.radial_res = radial_res
        self.vn_e = vn_e; self.vn_l = vn_l; self.vn_h = vn_h
        self.vn_first_azimuth = vn_first_azimuth
        self.window_detection = window_detection
        self.window_correction = window_correction
        self.deviation_factor = deviation_factor
        self.min_gates_correct = min_gates_correct
        self.n_it = n_it
        self.plot_velocity_deviations = plot_velocity_deviations
        
        self.dealiase()
        print(pytime.time() - t,'t_dealias')
        return self.data
        
    
    def dealiase(self):
        self.n_azi, self.n_rad = self.data.shape
        
        if not self.vn_first_azimuth is None:
            self.vn_radials = np.tile([self.vn_h, self.vn_l] if self.vn_first_azimuth == 'h' else [self.vn_l, self.vn_h],int(self.n_azi/2))
            self.vn_array = np.transpose(np.tile(self.vn_radials,(self.n_rad,1)))
                    
            
        self.sin_phi_sum = np.zeros((self.n_azi, self.n_rad)); self.cos_phi_sum = self.sin_phi_sum.copy()
        for n in range(self.n_it):    
            
            if n == 0:                 
                self.phi = np.pi * self.data / self.vn_e
                self.sin_phi = np.sin(self.phi); self.cos_phi = np.cos(self.phi)
                self.sin_phi[self.data_mask] = 0.; self.cos_phi[self.data_mask] = 0.
            else:                
                self.phi[self.outliers] = np.pi * self.data[self.outliers] / self.vn_e
                self.sin_phi[self.outliers] = np.sin(self.phi[self.outliers]); self.cos_phi[self.outliers] = np.cos(self.phi[self.outliers])
                                                                    
            self.detect_outliers()
            self.correct_outliers()


                                                    
    def detect_outliers(self):
        """Outlier detection step: First convert the velocities to a phase in [-pi, pi], by multiplying them by pi/vn_e. 
        Next, for each radar bin the circular mean of all phases in a certain window is calculated.
        Next, for each radar bin the minimum angular difference between the phase at that bin and the circular mean phase is calculated, and
        converted to a velocity difference. If this velocity difference is greater than the high/low Nyquist velocity corresponding to that bin, then
        that velocity is classified as an outlier.
        The reason for using phases instead of velocities is that it circumvents problems with extended Nyquist velocity aliasing."""
        self.calculate_sin_cos_phi_sum(self.window_detection)        
        phi_ref = np.arctan2(self.sin_phi_sum[self.data_nonmask], self.cos_phi_sum[self.data_nonmask])
        
        phi_diff = np.abs(self.phi[self.data_nonmask] - phi_ref)
        corr = phi_diff > np.pi
        phi_diff[corr] = 2*np.pi - phi_diff[corr]
        v_diff = phi_diff * self.vn_e/np.pi
                    
        if self.vn_first_azimuth is None:
            self.v_diff_detection = np.zeros((self.n_azi, self.n_rad))
            self.v_diff_detection[self.data_nonmask] = v_diff
            
            self.determine_vn_even_and_odd_radials()
        
        self.outliers = np.zeros((self.n_azi, self.n_rad), dtype = 'bool')
        self.outliers[self.data_nonmask] = np.abs(v_diff) > self.deviation_factor * self.vn_array[self.data_nonmask]     
        
    def correct_outliers(self):
        """Correction step: Add 2n*v_h or 2n*v_l to the velocity of the outliers, with n such that the difference between the resulting phase and 
        avg_a is minimised."""
        
        #Calculate again the mean phase, but exclude the outliers from the averaging
        self.sin_phi[self.outliers] = 0.0; self.cos_phi[self.outliers] = 0.0
        sin_phi_sum, cos_phi_sum = self.calculate_sin_cos_phi_sum(self.window_correction)        
        phi_ref = np.arctan2(sin_phi_sum[self.outliers],cos_phi_sum[self.outliers])
        
        v_diff = (self.phi[self.outliers] - phi_ref)*self.vn_e/np.pi
        
        if self.min_gates_correct > 0: 
            """Setting self.min_gates_correct to 0 is equal to stating that correction of velocities should always occur.
            If min_gates_correct = 0, then there might be some cases where the reference velocity is set to zero because there is no
            non-outlier velocity. This will likely however only occur in areas with very few velocities, in which case the data
            is usually not reliable/useful. For this reason min_gates_correct is set to zero by default, because this increases
            the speed of the algorithm."""
            n_nonoutliers = get_window_sum(np.logical_or(self.data_nonmask, self.outliers==False).astype('float'), self.window_correction)
            self.outliers[n_nonoutliers < self.min_gates_correct] = False
        
        correction_ints = np.round(v_diff/(2*self.vn_array[self.outliers]))
        data_outliers = self.data[self.outliers] - 2*correction_ints*self.vn_array[self.outliers]
        outside_extended_nyquist_interval = np.abs(data_outliers) > self.vn_e
        data_outliers_outside_nyq_int = data_outliers[outside_extended_nyquist_interval]
        data_outliers[outside_extended_nyquist_interval] = -np.sign(data_outliers_outside_nyq_int)*2*self.vn_e\
        + data_outliers_outside_nyq_int
        self.data[self.outliers] = data_outliers
        
    def calculate_sin_cos_phi_sum(self, window):
        self.sin_phi_sum, self.cos_phi_sum = get_window_sum(self.sin_phi, window), get_window_sum(self.cos_phi, window)
    
    def determine_vn_even_and_odd_radials(self):
        counts1, v = np.histogram(self.v_diff_detection[0:360:2], bins = 50, range = (-self.vn_e, self.vn_e))
        counts2, v = np.histogram(self.v_diff_detection[1:360:2], bins = 50, range = (-self.vn_e, self.vn_e))
        v_avg = (v[1:] + v[:-1])/2.
        
        self.vn_first_azimuth = 'h' if np.argmax(counts1[v_avg>2*self.vn_l]) > np.argmax(counts2[v_avg>2*self.vn_l]) else 'l'
        self.vn_radials = np.tile([self.vn_h, self.vn_l] if self.vn_first_azimuth == 'h' else [self.vn_l, self.vn_h],int(self.n_azi/2))
        self.vn_array = np.transpose(np.tile(self.vn_radials,(self.n_rad,1)))
        
        if self.plot_velocity_deviations:
            plt.figure()
            plt.plot((v[:-1]+v[1:])/2., counts1)
            plt.plot((v[:-1]+v[1:])/2., counts2)
            plt.ylim([0,1000])
            plt.show()
        



        
da = DualPRFDealiasing()

def apply_dual_prf_dealiasing(v_array, data_mask, azimuthal_res, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth = None, window_detection = None, window_correction = None, deviation_factor = 1.0, n_it = 5, c_array = None, mask_nearzero_velocities = False):
    """If window_detection and window_correction are not specified, then they are given below. Those window sizes are based upon the assumption that there can be many outliers,
    such that the window should not be too small. If the number of outliers is always small for a particular radar, then the window size should be smaller, to prevent undesired
    smoothing of the velocity field.
    Another assumption is that the low and high Nyquist velocity are about 30/40 kts resp. If the true Nyquist velocities are lower/higher, then the window size can be chosen
    smaller/larger. 
    """
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
                
    return da(v_array, data_mask, azimuthal_res, radial_res, vn_e, vn_l, vn_h, vn_first_azimuth, window_detection, window_correction, deviation_factor, n_it, c_array, mask_nearzero_velocities)