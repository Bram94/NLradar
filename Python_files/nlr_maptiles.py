# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import imageio
import numpy as np
import os
opa = os.path.abspath
import time as pytime

import nlr_functions as ft
import nlr_globalvars as gv

from PyQt5.QtCore import QThread,QObject,pyqtSignal    


class MapTiles(QThread):
    finished_signal=pyqtSignal(np.ndarray, dict)
    
    def __init__(self, pb_class):
        QThread.__init__(self)
        self.pb = pb_class
                
        self.radar = None
        self.x_min = None; self.x_max = None; self.y_min = None; self.y_max = None 
        #x_min, x_max, y_min and y_max specify the bounds of the current view
        #Should be set by calling self.set_radar_and_mapbounds
        
        self.basedir = opa(gv.programdir+'/Input_files')
        self.n_layers = int(np.sort([j for j in os.listdir(self.basedir) if os.path.isdir(opa(self.basedir+'/'+j)) and j[:5]=='Layer'])[-1][-1])
        self.layer = None #The ID of the tile layer that is currently used. Starts at 1.
        self.starting = True #Is used to prevent considering previous map tiles (which is done to increase speed) when reading map tiles for the first time.
        #self.starting is used instead of self.start, because 'start' is a method of QThread.
        self.get_info_tiles()
        
        self.finished_signal.connect(self.pb.draw_map_tiles)



    def set_radar_and_mapbounds(self, radar, lat_min, lat_max, lon_min, lon_max):
        self.radar = radar
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = lat_min, lat_max, lon_min, lon_max
        
    def run(self):
        map_bounds, tiles_changed = self.get_tiles()
        if tiles_changed:
            self.finished_signal.emit(self.map, map_bounds)
        
    def run_outside_thread(self):
        """Function that also collects map tiles, but that is not run within a separate thread. Can be used when the map must be obtained immediately.
        """
        map_bounds, tiles_changed = self.get_tiles()
        return self.map, map_bounds
    
    def get_tiles(self):
        self.get_filenames_tilegrid()
        tiles_changed = self.get_map_from_tiles()
        
        map_bounds = {'lat':[self.tilebounds[0], self.tilebounds[1]], 'lon':[self.tilebounds[2], self.tilebounds[3]]}
        return map_bounds, tiles_changed
        
        
        
    def get_info_tiles(self):      
        """This function ordens the files that contain the tiles into a grid, in which the tile in the first row and first column is the tile that is
        centered at the lowest latitude and longitude.
        - self.latslons_grid contains arrays with the latitudes and longitudes of the edges of the grid cells. These arrays have therefore length n(m)+1,
        where n(m) is the number of grid cells in the latitudinal(longitudinal) direction.
        - self.filenames_grid contains for each grid cell the filename that corresponds to the tile that resides in that cell.
        - self.tile_sizes contains for each layer the size of the images that represent the tiles.
        """
        self.latslons_grid = {}
        self.filenames_grid = {}
        self.tile_sizes = {}
        for i in range(1,self.n_layers+1):
            self.latslons_grid[i] = {}
            
            directory = opa(self.basedir+'/Layer_'+str(i))
            filenames = np.array(os.listdir(directory))
            
            extents = np.zeros((len(filenames),4))
            for j in range(len(filenames)):
                f = filenames[j]
                for k in range(3):
                    index = f.index('_')
                    extents[j,k] = float(f[:index])
                    f = f[index+1:]
                extents[j,3] = float(f[:f.index('.jpg')])
                
            self.latslons_grid[i]['lats'] = np.append(np.unique(extents[:,0]),np.max(extents[:,1]))
            self.latslons_grid[i]['lons'] = np.append(np.unique(extents[:,2]),np.max(extents[:,3]))
                
            latlon_spacing_layer = extents[0,1]-extents[0,0]
            ntiles_lat = int(round((self.latslons_grid[i]['lats'][-1]-self.latslons_grid[i]['lats'][0])/latlon_spacing_layer))
            ntiles_lon = int(round((self.latslons_grid[i]['lons'][-1]-self.latslons_grid[i]['lons'][0])/latlon_spacing_layer))
                
            sorted_fileindices_latmin = np.argsort(extents[:,0])
            
            self.filenames_grid[i] = []
            #First sort files based on the lat_min
            for j in range(ntiles_lat):
                #Then sort each row in the grid with the same latitude based on the longitude.
                extents_row = extents[sorted_fileindices_latmin[j*ntiles_lon:(j+1)*ntiles_lon]]
                filenames_row = filenames[sorted_fileindices_latmin[j*ntiles_lon:(j+1)*ntiles_lon]]
                sorted_fileindices_lonmin = np.argsort(extents_row[:,2])
                self.filenames_grid[i].append(filenames_row[sorted_fileindices_lonmin])
            self.filenames_grid[i] = np.array(self.filenames_grid[i])
            
            #Get the size of the images that represent the tiles
            self.tile_sizes[i] = imageio.imread(opa(directory+'/'+self.filenames_grid[i][0,0])).shape
    
    def get_filenames_tilegrid(self):
        npixels_main_layers = {}
        tilebounds_layers = {}
        
        for j in self.latslons_grid:
            latlon_spacing_layer = self.latslons_grid[j]['lons'][1]-self.latslons_grid[j]['lons'][0]
            
            lats_diff = self.latslons_grid[j]['lats']-self.lat_min
            outside_view = lats_diff<0
            #All elements in outside_view will be False if the view extents beyond the bounds of the map layer, and in this case we choose simply
            #the outer tile.
            if np.count_nonzero(outside_view)==0: outside_view = np.ones(len(lats_diff), dtype=bool)
            tile_lat_min = self.latslons_grid[j]['lats'][outside_view][np.argmin(np.abs(lats_diff[outside_view]))]
            lats_diff = self.latslons_grid[j]['lats']-self.lat_max
            outside_view = lats_diff>0 
            if np.count_nonzero(outside_view)==0: outside_view = np.ones(len(lats_diff), dtype=bool)
            tile_lat_max = self.latslons_grid[j]['lats'][outside_view][np.argmin(np.abs(lats_diff[outside_view]))]
            lons_diff = self.latslons_grid[j]['lons']-self.lon_min
            outside_view = lons_diff<0
            if np.count_nonzero(outside_view)==0: outside_view = np.ones(len(lons_diff), dtype=bool)
            tile_lon_min = self.latslons_grid[j]['lons'][outside_view][np.argmin(np.abs(lons_diff[outside_view]))]
            lons_diff = self.latslons_grid[j]['lons']-self.lon_max
            outside_view = lons_diff>0
            if np.count_nonzero(outside_view)==0: outside_view = np.ones(len(lons_diff), dtype=bool)
            tile_lon_max = self.latslons_grid[j]['lons'][outside_view][np.argmin(np.abs(lons_diff[outside_view]))]
            
            if tile_lat_min == tile_lat_max or tile_lon_min == tile_lon_max:
                # Prevents errors due to division by 0
                tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max = 1, -1, 1, -1
                
            #Determine the bounds of the tiles that would be used when layer j would be shown, and
            #the number of tiles that is required when showing layer j
            tilebounds_layers[j] = np.array([tile_lat_min, tile_lat_max, tile_lon_min, tile_lon_max])
            ntiles_layer = int(round((tile_lat_max-tile_lat_min)*(tile_lon_max-tile_lon_min)/latlon_spacing_layer**2))
            npixels_layer = int(ntiles_layer*self.tile_sizes[j][0]*self.tile_sizes[j][1])
            #npixels_main_layers gives the number of pixels of the tile layer that are used to span the latlon range shown at the screen.
            npixels_main_layers[j] = npixels_layer*(self.lat_max-self.lat_min)/(tilebounds_layers[j][1]-tilebounds_layers[j][0]) * \
            (self.lon_max-self.lon_min)/(tilebounds_layers[j][3]-tilebounds_layers[j][2])
            
        #Determine the desired layers from which to take the files.
        npixels_main = int(self.pb.wsize['main'][0]*self.pb.wsize['main'][1])
        layers_enough_pixels = [j for j in npixels_main_layers if npixels_main_layers[j]>=npixels_main]
        self.layer = layers_enough_pixels[0] if len(layers_enough_pixels)>0 else self.n_layers
        #Select the first layer for which the total number of pixels of all tiles is high enough, and if there is not such a layer, than the layer
        #with the highest resolution is selected.
        
        self.tilebounds = tilebounds_layers[self.layer]
        tilebounds_gridindices = []
        for j in range(4):
            coord_type = 'lats' if j<2 else 'lons'
            tilebounds_gridindices.append(np.argmin(np.abs(self.latslons_grid[self.layer][coord_type]-self.tilebounds[j])))
    
        self.filenames_tilegrid = self.filenames_grid[self.layer][
                tilebounds_gridindices[0]:tilebounds_gridindices[1], tilebounds_gridindices[2]:tilebounds_gridindices[3]]  
        
    def get_map_from_tiles(self):
        if not self.starting and self.filenames_tilegrid.shape==self.filenames_tilegrid_before.shape and \
        np.all(self.filenames_tilegrid==self.filenames_tilegrid_before):
            #self.map does not need to be updated
            return False #tiles_changed = False
        
        ni, nj = self.filenames_tilegrid.shape
        if not self.starting:
            nib, njb = self.filenames_tilegrid_before.shape
        
        tile_size = self.tile_sizes[self.layer]
        # At least 1 element per dimension, to prevent errors elsewhere in code due to division by dimension
        shape = max(ni*tile_size[0], 1), max(nj*tile_size[1], 1), 3
        self.map = np.empty(shape, dtype='uint8')
        i_list, j_list = [], []
        ib_list, jb_list = [], []
        for i in range(ni):
            for j in range(nj):
                if not self.starting and self.filenames_tilegrid[i,j] in self.filenames_tilegrid_before:
                    ib, jb = [k[0] for k in np.where(self.filenames_tilegrid_before==self.filenames_tilegrid[i,j])]
                    i_list.append(i); j_list.append(j)
                    ib_list.append(ib); jb_list.append(jb)
                else:
                    self.map[(ni-(i+1))*tile_size[0]:(ni-i)*tile_size[0], j*tile_size[1]:(j+1)*tile_size[1]] = \
                    imageio.imread(opa(self.basedir+'/Layer_'+str(self.layer)+'/'+self.filenames_tilegrid[i,j]))[:,:,::-1] #Convert from BGR to RGB
        if len(i_list) > 0:
            i_min, i_max = min(i_list), max(i_list)
            j_min, j_max = min(j_list), max(j_list)
            ib_min, ib_max = min(ib_list), max(ib_list)
            jb_min, jb_max = min(jb_list), max(jb_list)            
            self.map[(ni-i_max-1)*tile_size[0]:(ni-i_min)*tile_size[0], j_min*tile_size[1]:(j_max+1)*tile_size[1]] =\
            self.map_before[(nib-ib_max-1)*tile_size[0]:(nib-ib_min)*tile_size[0], jb_min*tile_size[1]:(jb_max+1)*tile_size[1]]
                
        self.filenames_tilegrid_before = self.filenames_tilegrid
        self.map_before = self.map
        self.starting = False
        return True #tiles_changed = True
