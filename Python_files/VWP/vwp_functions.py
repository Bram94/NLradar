# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np

import nlr_globalvars as gv


def interpolate(x, y, x_int):
    """Interpolate y to x=x_int.
    x must be a 1D array of values, and y can be a multidimensional array, of which the first dimension must match that of x.
    Returns the interpolated y value and x_int_true, which is the x coordinate at which the interpolated value is valid. 
    This differs from x_int when x_int lies outside the range of values given in x.
    """
    xi_closest = np.abs(x-x_int).argmin()
    
    if x[xi_closest] < x_int:
        if xi_closest == len(x)-1:
            y_int = y[xi_closest]
            x_int_true = x[xi_closest]
        else:
            y_int = ((x[xi_closest+1]-x_int)*y[xi_closest]+(x_int-x[xi_closest])*y[xi_closest+1])/(x[xi_closest+1]-x[xi_closest])
            x_int_true = x_int
    else:
        if xi_closest == 0: 
            y_int = y[xi_closest]
            x_int_true = x[xi_closest]
        else:
            y_int = ((x[xi_closest]-x_int)*y[xi_closest-1]+(x_int-x[xi_closest-1])*y[xi_closest])/(x[xi_closest]-x[xi_closest-1])
            x_int_true = x_int
    return x_int_true, y_int

def get_bunkers_stormmotions(V, h, units):
    if h[0] > 1. or h[-1] - h[0] < 3.:
        sm = None
        return sm, sm, sm
    
    dh = 0.1
    heights_interp = np.arange(0, 6+1e-3, dh)
    V_interp = np.array([interpolate(h, V, j)[-1] for j in heights_interp])
    Vmean_0_6 = np.mean(V_interp, axis=0)
    Vmean_0_0p5 = np.mean(V_interp[:int(round(0.5/dh))], axis=0)
    Vmean_5p5_6 = np.mean(V_interp[-int(round(0.5/dh)):], axis=0)
    V_shear = Vmean_5p5_6-Vmean_0_0p5
    #0-0.5 - 5.5-6km wind shear vector
    dv = 7.5*gv.scale_factors_velocities[units]
    BK_left = Vmean_0_6 + dv*np.cross(np.array([0,0,1]), V_shear/np.linalg.norm(V_shear))[:2]
    BK_right = Vmean_0_6 - dv*np.cross(np.array([0,0,1]), V_shear/np.linalg.norm(V_shear))[:2]
    
    return Vmean_0_6, BK_left, BK_right

def get_deviant_tornado_motion(V, h, sm):
    if h[0] > 0.3:
        return None
    dh = 0.1
    heights_interp = np.arange(0, 0.5+1e-3, dh)
    V_interp = np.array([interpolate(h, V, j)[-1] for j in heights_interp])
    Vmean_0_0p5 = np.mean(V_interp, axis=0)
    return 0.5*(sm+Vmean_0_0p5)

    
def calculate_bulk_shear(V, h, h_min, h_max):
    h_min_true, V_min = interpolate(h, V, h_min)
    h_max_true, V_max = interpolate(h, V, h_max)

    return h_min_true, h_max_true, np.linalg.norm(V_max-V_min)

def calculate_SRH(V, h, h_min, h_max, sm, units): #sm is storm motion in format [sm_x, sm_y]
    """Important: input arrays V and sm should have the units that are specified by the variable
    units. The velocities are here converted to m/s, to get the unit m^2/s^2 for SRH.
    """
    V = V/gv.scale_factors_velocities[units] #Should not be done in-place, because then the array
    #gets modified multiple times in subsequent calls of this function.
    sm = sm/gv.scale_factors_velocities[units]
    
    h_min_true, V_min = interpolate(h, V, h_min)
    h_max_true, V_max = interpolate(h, V, h_max)
    
    hi_min = np.argmin(np.abs(h-h_min))
    if h[hi_min] <= h_min:
        hi_min += 1
    hi_max = np.argmin(np.abs(h-h_max))
    if h[hi_max] >= h_max:
        hi_max -= 1
    
    V_SRH = np.concatenate([[V_min],V[hi_min:hi_max+1],[V_max]], axis=0)
    u = V_SRH[:,0]; v = V_SRH[:,1]
    SRH = np.sum([(u[i+1]-sm[0])*(v[i]-sm[1])-(u[i]-sm[0])*(v[i+1]-sm[1]) for i in range(len(V_SRH)-1)])
    return h_min_true, h_max_true, SRH


def calculate_shear_profile(V, h, units):
    scale = 1./gv.scale_factors_velocities[units]
    # Should not be done in-place, because then the array gets modified multiple times in subsequent calls of this function.
    V = scale*V
    h = h*1e3
    shear = np.diff(V, axis=0)/np.diff(h)[:, np.newaxis]
    shear_magnitude = np.linalg.norm(shear, axis=1)
    return shear_magnitude

def interpolate_velocity_profile(V, h, units, h_max=None):
    if h_max and h.max() > h_max:
        _, V3km = interpolate(h, V, 3.)
        select = h < 3.
        V = np.append(V[select], [V3km], axis=0)
        h = np.append(h[
            select], 3.)
        
    V_split = 1.*gv.scale_factors_velocities[units]
    velocity, height = [], []
    for i in range(len(V)-1):
        V1, V2 = V[i], V[i+1]
        h1, h2 = h[i], h[i+1]
        velocity += [V1]
        height += [h1]
        
        delta_V = np.linalg.norm(V2-V1)
        if delta_V > V_split:
            n_sublayers = int(np.ceil(delta_V/V_split))
            # Exclude the last sublayer, since its endpoint will be included in the iteration over the next layer
            for j in range(1, n_sublayers):
                Vj = V1+j/n_sublayers*(V2-V1)
                hj = h1+j/n_sublayers*(h2-h1)
                velocity += [Vj]
                height += [hj]
    velocity = np.array(velocity+([V2] if len(height) else []))
    height = np.array(height+([h2] if len(height) else []))
    return velocity, height

def calculate_vorticity_components(V, h, sm, units):
    V = V/gv.scale_factors_velocities[units]
    sm = sm/gv.scale_factors_velocities[units]
    h = h*1e3
    
    shear = np.diff(V, axis=0)/np.diff(h)[:, np.newaxis]
    vorticity = np.cross(np.array([[0,0,1]]), shear)[:, :2]
    avg_V = 0.5*(V[1:]+V[:-1])
    srV = avg_V-sm
    srV_unitvector = srV/np.linalg.norm(srV, axis=1)[:, np.newaxis]
    sw_vorticity = np.sum(vorticity*srV_unitvector, axis=1)
    cw_vorticity = -np.sum(shear*srV_unitvector, axis=1)
    return sw_vorticity, cw_vorticity

def calculate_SR_windspeed(V, sm):
    avg_V = 0.5*(V[1:]+V[:-1])
    srV = avg_V-sm
    return np.linalg.norm(srV, axis=1)

def calculate_avg_quantity_halflevels(quant, h, V, h_min, h_max):
    # Assumes quant to be defined at half-levels (in between levels of h)
    h_min_true, _ = interpolate(h, V, h_min)
    h_max_true, _ = interpolate(h, V, h_max)
        
    hi_min = np.argmin(np.abs(h-h_min_true))
    if h[hi_min] <= h_min_true:
        hi_min += 1
    hi_max = np.argmin(np.abs(h-h_max_true))
    if h[hi_max] >= h_max_true:
        hi_max -= 1
    
    quant = quant[hi_min-1:hi_max+1]
    h = np.concatenate([[h_min_true], h[hi_min:hi_max+1], [h_max_true]])
    avg_quant = np.sum([quant[i]*(h[i+1]-h[i]) for i in range(len(quant))]) / (h_max_true-h_min_true)
    return h_min_true, h_max_true, avg_quant


def get_direction_magnitude_vector(vector):
    direction = np.mod(180+180/np.pi*np.arctan2(vector[0], vector[1]),360)
    #direction in degrees, measured relative to North.
    speed = np.linalg.norm(vector)
    
    return direction, speed


def get_hlabel_pos(V, h, V_label, h_label, plot_range, rel_pos='left'):
    #rel_pos can be 'left' or 'right', and specifies whether the label is placed to the left or to the right of
    #the vector that connects 2 subsequent velocities around the height of h_label.
    hi_closest = np.argmin(np.abs(h-h_label))
    h_closest = h[hi_closest]
    if h_closest<=h_label:
        if hi_closest==len(h)-1:
            v_below = V[hi_closest-1]
            v_above = V[hi_closest]
        else:
            v_below = V[hi_closest]
            v_above = V[hi_closest+1]
    else:
        if hi_closest==0:
            v_below = V[hi_closest]
            v_above = V[hi_closest+1]
        else:
            v_below = V[hi_closest-1]
            v_above = V[hi_closest]
            
    sign = 1 if rel_pos=='left' else -1
    if np.linalg.norm(v_above-v_below)>0.:
        return V_label + sign*plot_range/70*np.cross(np.array([0,0,1]),(v_above-v_below)/np.linalg.norm(v_above-v_below))[:2]
    else:
        return V_label + sign*plot_range/70*np.array([1,1])/np.sqrt(2)
    
    
if __name__ == "__main__":    
    V = np.array([[0, 10], [0, 20], [0, 30]])
    h = np.array([0, 1, 2])
    sm = np.array([20, 10])
    units = 'm/s'
    calculate_streamwise_vorticity_profile(V, h, sm, units)