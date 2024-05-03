# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np
from vispy.visuals.transforms import BaseTransform

import nlr_functions as ft
import nlr_globalvars as gv



class To_Mercator(BaseTransform):
    """This class if for mercator transformation of the Base map."""
    glsl_map = """
                vec4 mercator_map(vec4 pos)
                {   
		    float k0 = 0.75;
                    float a  = 1.00;
                    float lambda = atan(sinh(pos.x/(k0*a)),cos(pos.y/(k0*a)));
                    float phi    = asin(sin(pos.y/(k0*a))/cosh(pos.x/(k0*a)));
                    return vec4(lambda,phi,pos.z,1);
                }
                """

    glsl_imap = """
                vec4 mercator_imap(vec4 pos)
                {   
		    //pos.x is the longitude (lambda), pos.y the latitude (phi)
		    float k0 = 0.75;
                    float a  = 1.00;
                    float x = 0.5*k0*log((1.0+sin(pos.x)*cos(pos.y)) / (1.0 - sin(pos.x)*cos(pos.y)));
                    float y = k0*a*atan(tan(pos.y), cos(pos.x));
                    return vec4(x,y,pos.z,1);
                }
                """
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    
    

class PolarTransform(BaseTransform):
    """Polar transform
    Maps (theta, r, z) to (x, y, z), where `x = r*cos(theta)`
    and `y = r*sin(theta)`.
    """
    glsl_map = """
        vec4 polar_transform_map(vec4 pos) {
            if (pos.y > 400) {
                pos.y = pos.y+180;
            }
            return vec4(pos.x*sin(pos.y), pos.x*cos(pos.y), pos.z, 1);
            }
        """

    glsl_imap = """
        vec4 polar_transform_imap(vec4 pos) {
            // TODO: need some modulo math to handle larger theta values..?
            float theta = -atan(pos.x, pos.y); 
            theta = degrees(theta+3.14159265358979323846);
            float r = length(pos.xy);
            return vec4(r, theta, pos.z, 1);
        }
        """

    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    
    
    
# Suggestion: use elliposidal form https://neacsu.net/docs/geodesy/snyder/5-azimuthal/sect_25/
class LatLon_to_Azimuthal_Equidistant_Transform(BaseTransform):
    glsl_map = """        
    vec4 slantrange_to_groundrange_imap(vec4 pos) {
        // lat, lon to x, y
        float Re = 6371.0; 
        pos.x = radians(pos.x);
        pos.y = radians(pos.y);
        
        float cos_lat = cos(pos.x);
        float sin_lat = sin(pos.x);
        
        float cos_c = sin($lat_0)*sin_lat+cos($lat_0)*cos_lat*cos(pos.y-$lon_0);
        float sin_c = sin(acos(cos_c));
        float k = asin(sin_c)/sin_c;
                
        float x = Re*k*cos_lat*sin(pos.y-$lon_0);
        float y = Re*k*(cos($lat_0)*sin_lat-sin($lat_0)*cos_lat*cos(pos.y-$lon_0));
        return vec4(x, y, pos.z, 1);
        }
    """

    glsl_imap = """
    vec4 slantrange_to_groundrange_imap(vec4 pos) {
        // x, y to lat, lon
        float Re = 6371.0;
        
        pos.x /= Re;
        pos.y /= Re;
        float c = length(pos.xy);
        float cos_c = cos(c);
        float sin_c = sin(c);

        float lat = asin(cos_c*sin($lat_0)+pos.y*sin_c*cos($lat_0)/c);
        float lon = $lon_0+atan(pos.x*sin_c/(c*cos($lat_0)*cos_c-pos.y*sin($lat_0)*sin_c));
        return vec4(degrees(lon), degrees(lat), pos.z, 1);
        }
    """    
        
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    
    def __init__(self, radar=None, latlon_0=None):
        # Either radar or latlon_0 should be specified
        super(LatLon_to_Azimuthal_Equidistant_Transform, self).__init__()
        if radar:
            self.radar = radar
        else:
            self.latlon_0 = latlon_0
        
        
    @property
    def radar(self):
        return self._radar

    @radar.setter
    def radar(self, radar):
        self._radar = radar
        self.latlon_0 = gv.radarcoords[radar]

    @property
    def latlon_0(self):
        return self._latlon_0

    @latlon_0.setter
    def latlon_0(self, latlon_0):
        self._latlon_0 = np.array(latlon_0)
        self._update_shaders()
                
    def map(self, latlon):
        return ft.aeqd(self.latlon_0, latlon)

    def imap(self, xy):
        return ft.aeqd(self.latlon_0, xy, inverse=True)
    
    def _update_shaders(self):
        # Convert to radians
        self._shader_map['lat_0'] = self.latlon_0[0]*np.pi/180.
        self._shader_map['lon_0'] = self.latlon_0[1]*np.pi/180.
        self._shader_imap['lat_0'] = self.latlon_0[0]*np.pi/180.
        self._shader_imap['lon_0'] = self.latlon_0[1]*np.pi/180.


        
class Slantrange_to_Groundrange_Transform(BaseTransform):  
    glsl_map = """        
    vec4 slantrange_to_groundrange_map(vec4 pos) {
        float Re=6371.0; 
        float ke=4.0/3.0;
        float theta=radians($scanangle);
        float h = 0.0;
        float gr = pos.x;
        if(theta>0.0)
            h = sqrt(pow(pos.x,2)+pow(Re*ke,2)+2*Re*ke*pos.x*sin(theta));
            gr = ke*Re*asin(pos.x*cos(theta)/h);
        return vec4(gr, pos.y, pos.z, 1);
        }

    """

    glsl_imap = """
    vec4 slantrange_to_groundrange_imap(vec4 pos) {
        // Ground range to slant range
        float Re=6371.0; 
        float ke=4.0/3.0;
        float theta=radians($scanangle);
        float sr=pos.x;
        if(theta>0.0)
            sr=Re*ke/cos(pos.x/(Re*ke)+theta)*sin(pos.x/(Re*ke));
        return vec4(sr, pos.y, pos.z, 1);
        }
    """    
        
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False
    
    def __init__(self, scanangle=None):
        super(Slantrange_to_Groundrange_Transform, self).__init__()
        self.scanangle = 0. if scanangle in (90.,None) else scanangle
        
        

    @property
    def scanangle(self):
        return self._scanangle

    @scanangle.setter
    def scanangle(self, s):
        self._scanangle = 0. if s==90. else float(s)
        self._update_shaders()
        
    def map(self, coords):
        Re=6371; ke=4./3.
        sr=coords[0]; theta=coords[1]*np.pi/180.
        h1 = np.sqrt(np.power(sr,2)+np.power(Re*ke,2)+2*Re*ke*sr*np.sin(theta))
        gr = ke*Re*np.arcsin(sr*np.cos(theta)/h1)
        return gr

    def imap(self, coords):
        Re=6371; ke=4./3.
        gr=coords[0]; theta=coords[1]*np.pi/180.
        sr=Re*ke/np.cos(gr/(Re*ke)+theta)*np.sin(gr/(Re*ke))
        return sr
    
    def _update_shaders(self):
        self._shader_map['scanangle']=self.scanangle
        self._shader_imap['scanangle']=self.scanangle
        
        
        
class Slantrange_to_Groundrange_Transform_Cosine(BaseTransform):  
    glsl_map = """        
        vec4 polar_transform_map(vec4 pos) {
            return vec4(pos.y * cos(pos.x), pos.y * sin(pos.x), pos.z, 1);
        }
        """
        
    glsl_imap = """
        vec4 test_imap(vec4 pos) {
            // Ground range to slant range
            return vec4(pos.x/cos(radians($scanangle)), pos.y, pos.z, 1);
            }
        """    

    Linear = True
    Orthogonal = True
    NonScaling = False
    Isometric = False
    
    def __init__(self, scanangle=None):
        super(Slantrange_to_Groundrange_Transform_Cosine, self).__init__()
        self._scanangle = 0.0
        self.scanangle = scanangle

    @property
    def scanangle(self):
        return self._scanangle

    @scanangle.setter
    def scanangle(self, s):
        self._scanangle = s
        
    def map(self, coords):
        return coords[0]*np.cos(coords[1]*np.pi/180.)

    def imap(self, coords):
        return coords[0]/np.cos(coords[1]*np.pi/180.)
        
    def shader_map(self):
        fn = super(Slantrange_to_Groundrange_Transform_Cosine, self).shader_map()
        fn['scanangle'] = self.scanangle
        return fn

    def shader_imap(self):
        fn = super(Slantrange_to_Groundrange_Transform_Cosine, self).shader_imap()
        fn['scanangle'] = self.scanangle
        return fn
        
    
        
def generate_vertices_circle(center, radius, start_angle, span_angle, num_segments):
    if isinstance(radius, (list, tuple)):
        if len(radius) == 2:
            xr, yr = radius
        else:
            raise ValueError("radius must be float or 2 value tuple/list"
                             " (got %s of length %d)" % (type(radius),
                                                         len(radius)))
    else:
        xr = yr = radius

    start_angle = np.deg2rad(start_angle)

    vertices = np.empty([num_segments + 2, 2], dtype=np.float32)

    # split the total angle into num_segments intances
    theta = np.linspace(start_angle,
                        start_angle + np.deg2rad(span_angle),
                        num_segments + 1)

    # PolarProjection
    vertices[:-1, 0] = center[0] + xr * np.cos(theta)
    vertices[:-1, 1] = center[1] + yr * np.sin(theta)

    # close the curve
    vertices[num_segments + 1] = np.float32([center[0], center[1]])

    return vertices