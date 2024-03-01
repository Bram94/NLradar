# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np

import nlr_globalvars as gv
import nlr_functions as ft



def calculate_srv_array(v_arr, stormmotion, data_startazimuth = 0.):
    direction, speed = stormmotion
    azimuthal_bins = len(v_arr)
    srv_term = speed*np.cos(np.deg2rad(direction-data_startazimuth-np.linspace(0, 360, azimuthal_bins, endpoint=False)))
    # SRV is calculated from uint velocity data since this prevents convert back and forth between dtypes uint and float. 
    # Hence, the srv_term above, that is added to v_arr below, is first converted to a corresponding term in uint dtype.
    n_bits = gv.products_data_nbits['s']
    pm_lim = gv.products_maxrange_masked['s']
    v_arr += ft.convert_float_to_uint(srv_term+pm_lim[0], n_bits, pm_lim)[:, np.newaxis]
    return v_arr
                                                    
def calculate_zdr_array(data_Zh, data_Zv):
    return data_Zh-data_Zv