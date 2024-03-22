# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import nlr_background as bg
import nlr_customvispy as cv
import nlr_functions as ft
import nlr_globalvars as gv
import nlr_maptiles as mt
from VWP.nlr_plottingvwp import PlottingVWP

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

import vispy
vispy.set_log_level(verbose='warning')
from vispy import app
from vispy import color
from vispy import visuals
from vispy import gloo
from vispy.visuals.transforms import STTransform, ChainTransform
from vispy.visuals.filters import Clipper, ColorFilter

from OpenGL import GL

import numpy as np
from numpy import genfromtxt
import os
opa=os.path.abspath
import time as pytime
import copy
import traceback



"""This class handles plotting of the radar panels and associated color bars and titles. 
Plotting of other things such as the VWP hodographs happens in separate classes, although some necessary actions such as those required to make room for these other
things happen also in this class. The same holds for initiating the plotting calls for these other things.
"""



class Plotting(QObject,app.Canvas):
    setback_gridheightrings_signal=pyqtSignal()
    update_map_tiles_signal=pyqtSignal()
    def __init__(self, gui_class, empty_init=False):
        if empty_init:
            # Is used to initialise this class without getting issues below with references to other classes that at this point don't exist yet.
            return
        super(Plotting, self).__init__()
        self.start_dpi = gui_class.screen_DPI()
        self.start_screen_size = gui_class.screen_size()
        app.Canvas.__init__(self, dpi=self.start_dpi) #It is important to set DPI manually, since at Linux vispy
        #is not able to determine it correctly automatically!
        self.gui=gui_class
        
        self.setback_gridheightrings_signal.connect(self.setback_gridheightrings)
        self.update_map_tiles_signal.connect(self.update_map_tiles)
        
        self.gui=gui_class
        self.crd=self.gui.crd
        self.ani=self.gui.ani
        self.dsg=self.crd.dsg
        self.dp=self.dsg.dp
        self.vwp = PlottingVWP(gui_class=self.gui, pb_class = self)
        self.mt = mt.MapTiles(self)
        
        #Startup settings
        self.wdims = np.array([self.gui.dimensions_main['width'], self.gui.dimensions_main['height']])
        #First value gives x dimension of the 'left' and 'right' widgets. Second value gives y dimension of the 'top' and 'bottom' widgets.
        #In cm.
        
        self.panels=1; self.panellist=(0,)
        self.max_panels=10
        self.nrows=1; self.ncolumns=1
        self.panel=0       
        
        self.data_empty={j:True for j in range(self.max_panels)} #True when self.dsg.data[j] contains no radar data for panel j
        self.data_isold={j:True for j in range(self.max_panels)} #True when self.dsg.data[j] contains data that is not for the current time
        # Parameters that are valid for the data that is currently shown in the panels. Since a panel might not get updated always, 
        # it is necessary to store these parameters and to not use current values.
        # 'xy_bins', 'res' replace 'radial_bins','radial_res','azimuthal_bins','azimuthal_res' when a plain product is displayed in 
        # cartesian coordinates
        self.data_attr = {j:{} for j in ('product','scanangle','radial_bins','radial_res','azimuthal_bins','azimuthal_res',
                                          'xy_bins', 'res', 'proj', 'scantime', 'scandatetime')}
        self.data_attr_before = copy.deepcopy(self.data_attr)
        
        self.heightrings_scanangles = {j:-1 for j in range(self.max_panels)}
        self.products_before=self.crd.products.copy(); self.scans_before=self.crd.scans.copy()  
        self.productunits=gv.productunits_default.copy()
        self.scale_factors={j: 1. for j in gv.products_all}
        self.firstdraw_map=True #It is important to draw the map for all panels during the first draw, because otherwise the scale gets screwed up when changing the
        #number of panels.
        self.changing_panels=False
        self.datareadout_text=''
        self.use_interpolation=False
        self.firstplot_performed=False
        self.max_delta_time_move_view_with_storm=12*3600
        self.mouse_hold=False
        self.mouse_hold_left=False
        self.mouse_hold_right=False
        self.last_mouse_pos_px=None
        self.gridheightrings_removed=False
        self.radar_mouse_selected=None
        self.marker_mouse_selected_index = None
        self.in_view_mask_specs=None
        self.timer_setback_gridheightrings_running=False
        self.timer_update_map_tiles_running = False
        self.postpone_plotting_gridheightrings=False
        self.panels_horizontal_ghtext=list(range(self.max_panels)); self.panels_vertical_ghtext=[0,5]
        self.lines_order=['provinces','countries','rivers','grid','heightrings'] #Order in which lines are drawn, from bottom to top
        self.radarcoords_xy = np.array(ft.aeqd(gv.radarcoords[self.crd.radar], np.array([gv.radarcoords[j] for j in gv.radars_all])))
        self.zoomfactor_vispy=0.007
                
        self.cm1={}; self.cm2={}; 
        self.data_values_colors={}; self.data_values_colors_int={}
        self.data_values_ticks={}; self.tick_map = {}
        self.masked_values={}; self.masked_values_int={}; self.clim_int={}
        self.cmap_lastmodification_time={}
        self.cmaps_minvalues_before=self.gui.cmaps_minvalues.copy(); self.cmaps_maxvalues_before=self.gui.cmaps_maxvalues.copy()
        self.cbars_products_before=[]
        
        self.starting=True
        self.set_draw_action('updating_cbars')
        self.update_map_tiles_ondraw = False
        self.start_scissor_test=True
        
        self.base_range = 250. #The base/default panel y range (distance panel center to top of panel), is used on start-up and when resetting panel view
        #The fractional width of the VWP plot relative to that of the whole canvas. Gets updated on resizing in self.calculate_vwp_relxdim
        self.vwp_relxdim = 0.255
        
        self.unitcircle_vertices=cv.generate_vertices_circle([0,0],1,0,360,100)[:-1]


        #Information about the positions of panels. plotnumber increases from 1 to the total number of panels when going from the upperleft corner to the lowerright corner (zigzagging). 
        #For panelnumber the presence of 8 panels is assumed, and the number corresponds to the position in a 2x4 grid. An exception is the presence of 2 panels, where the second panel (panel 1, it starts at 0)
        #is assigned a panelnumber of 4 (meaning 5th panel), although it should have panelnumber 2 according to the above rule. The reason for this exception is that it works nicer when changing the number of panels.
        self.rows_panels={1:{0:0},2:{0:0,1:0},3:{0:0,1:0,2:0},4:{0:0,1:0,2:1,3:1},6:{0:0,1:0,2:0,3:1,4:1,5:1},8:{0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1},10:{0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:1}}
        self.columns_panels={1:{0:0},2:{0:0,1:1},3:{0:0,1:1,2:2},4:{0:0,1:1,2:0,3:1},6:{0:0,1:1,2:2,3:0,4:1,5:2},8:{0:0,1:1,2:2,3:3,4:0,5:1,6:2,7:3},10:{0:0,1:1,2:2,3:3,4:4,5:0,6:1,7:2,8:3,9:4}}
        self.plotnumber_to_panelnumber={1:{0:0},2:{0:0,1:5},3:{0:0,1:1,2:2},4:{0:0,1:1,2:5,3:6},6:{0:0,1:1,2:2,3:5,4:6,5:7},8:{0:0,1:1,2:2,3:3,4:5,5:6,6:7,7:8},10:{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}}
        self.panelnumber_to_plotnumber={}
        for j in self.plotnumber_to_panelnumber: 
            self.panelnumber_to_plotnumber[j]={v: k for k, v in self.plotnumber_to_panelnumber[j].items()}
            
        self.determine_panellist_nrows_ncolumns()
        self.set_widget_sizes_and_bounds()
        
        self.set_cmaps(gv.products_all)
        

        #Visuals are drawn in the order in which they are defined here
        self.visuals_order=['background','background_map','map','radar_polar','radar_cartesian','map_lines','gh_lines','text_hor1','text_hor2','text_vert1','text_vert2']
        self.visuals_order+=['sm_pos_markers','radar_markers','panel_borders','titles']
        self.visuals_order+=['cbar'+str(j) for j in range(10)]+['cbars_ticks','cbars_reflines','cbars_labels']
        self.visuals_panels=['map','radar_polar','radar_cartesian','map_lines','gh_lines','text_hor1','text_hor2','text_vert1','text_vert2','sm_pos_markers','radar_markers'] 
        #Visuals that are created for each panel separately
        self.visuals_global=[j for j in self.visuals_order if not j in self.visuals_panels] #Visuals that are not created for each panel separately
        
        self.visuals={j:({} if j in self.visuals_panels else None) for j in self.visuals_order}
        
        #self.visuals_widgets lists for each widget the visuals that are located in it. One visual could be located in multiple widgets, but this should only
        #be done when these widgets are always updated simultaneously, or when the widgets number of objects that is displayed is small.
        self.widgets=['left','right','bottom','top','main']
        self.visuals_widgets={}
        self.visuals_widgets['left']=['cbar'+str(j) for j in range(5)]+['cbars_ticks','cbars_reflines','cbars_labels']
        self.visuals_widgets['right']=['cbar'+str(j) for j in range(5,10)]+self.visuals_widgets['left'][-3:]
        self.visuals_widgets['bottom']=self.visuals_widgets['top']=['titles']
        self.visuals_widgets['main']=['background_map']+self.visuals_panels+['panel_borders']
        
        self.font_sizes = {'text_hor1':'self.gui.gridheightrings_fontsize', 'text_vert1':'self.gui.gridheightrings_fontsize', 
                           'titles':"self.gui.fontsizes_main['titles']", 'cbars_ticks':"self.gui.fontsizes_main['cbars_ticks']", 
                           'cbars_labels':"self.gui.fontsizes_main['cbars_labels']"}
                
        
        # set self.map_data and self.map_bounds
        self.update_map_tiles(xy_bounds = [-300, 300, -300, 300], separate_thread=False)
        self.shapefiles_latlon_combined, self.shapefiles_connect_combined = bg.import_shapefiles()
                
        self.lines_pos_combined={}; self.lines_connect_combined={}; self.lines_colors_combined={}
        for i in range(self.max_panels):
            self.lines_pos_combined[i]={}; self.lines_connect_combined[i]={}; self.lines_colors_combined[i]={}
            for j in self.gui.lines_names:
                self.lines_pos_combined[i][j]=[]
                self.lines_connect_combined[i][j]=[] #The last number of a sub-array of connect should always be zero, to prevent different lines from getting connected.
                self.lines_colors_combined[i][j]=[]
        self.update_combined_lineproperties(range(self.max_panels),changing_radar=True,start=True)
        
        self.ghtext_hor_pos_combined={}; self.ghtext_hor_strings_combined={}; self.ghtext_vert_pos_combined={}; self.ghtext_vert_strings_combined={}
        self.heights_text={}; self.heights_text_pos={}
        for i in range(self.max_panels):
            self.ghtext_hor_pos_combined[i]={}; self.ghtext_hor_strings_combined[i]={}; self.ghtext_vert_pos_combined[i]={}; self.ghtext_vert_strings_combined[i]={}
            for j in self.gui.ghtext_names:
                self.ghtext_hor_pos_combined[i][j]={}
                self.ghtext_hor_strings_combined[i][j]={}
                self.ghtext_vert_pos_combined[i][j]={}
                self.ghtext_vert_strings_combined[i][j]={}      
                
                
            
        self.visuals['background']=visuals.RectangleVisual(center=[0,0],color=self.gui.bgcolor/255)
        self.visuals['background'].transform=STTransform()
        
        self.visuals['background_map']=visuals.RectangleVisual(center=[0,0],color=self.gui.bgmapcolor/255.)
        self.visuals['background_map'].transform=STTransform(scale=self.wsize['main'],translate=self.wcenter['main'])
        
        self.map_transforms = {}
        self.map_initial_bounds = None; self.map_initial_scale = None
        self.ref_radial_bins = {}; self.ref_azimuthal_bins = {}
        self.polar_transforms = {}
        self.polar_transforms_individual = {'scanangle':{}, 'scale':{}, 'polar':{}}
        self.cartesian_transforms = {}
        self.cartesian_transforms_individual = {'scale_translate':{}}
        
        startup_string=''.join([str(j) for j in range(10)]) #Initialize all TextVisuals with startup_string, to shorten the time that is needed to perform the
        #first plot.
        
        self.map_colorfilter = ColorFilter(self.gui.mapcolorfilter)
        self.text_hor_top_colorfilter = ColorFilter(np.append(self.gui.gridheightrings_fontcolor['top']/255., 1.))
        self.text_hor_bottom_colorfilter = ColorFilter(np.append(self.gui.gridheightrings_fontcolor['bottom']/255., 1.))
        
        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'] = STTransform(scale=(scale_x,scale_y), translate=(t_x, t_y))
        self.map_transforms['aeqd'] = cv.LatLon_to_Azimuthal_Equidistant_Transform(gv.radarcoords[self.crd.radar])
        for j in range(self.max_panels):
            if j==0:
                self.visuals['map'][j]=visuals.ImageVisual(self.map_data,interpolation='bilinear',method='impostor')
            else:
                #Create a view of the map in the first panel, such that the map data is stored only once in memory.
                #Using views of visuals is possible when all panels show the same visual.
                self.visuals['map'][j] = self.visuals['map'][0].view()
            self.visuals['map'][j].transform=STTransform(scale=(1,-1))*self.map_transforms['aeqd']*self.map_transforms['st']
            self.visuals['map'][j].attach(self.map_colorfilter)
            self.visuals['map'][j].visible = self.gui.mapvisibility
            
            self.visuals['radar_polar'][j] = visuals.ImageVisual(method='auto', cmap=self.cm1[self.crd.products[j]], clim=self.clim_int[self.crd.products[j]])
            self.visuals['radar_cartesian'][j] = visuals.ImageVisual(method='auto', cmap=self.cm1[self.crd.products[j]], clim=self.clim_int[self.crd.products[j]])
            self.polar_transforms_individual['scanangle'][j]=cv.Slantrange_to_Groundrange_Transform()
            self.polar_transforms_individual['scale'][j]=STTransform()
            self.polar_transforms_individual['polar'][j]=cv.PolarTransform()
            self.cartesian_transforms_individual['scale_translate'][j] = STTransform()
            self.visuals['radar_polar'][j].transform = self.polar_transforms_individual['polar'][j] * self.polar_transforms_individual['scanangle'][j] * self.polar_transforms_individual['scale'][j]
            self.visuals['radar_cartesian'][j].transform = self.cartesian_transforms_individual['scale_translate'][j]
            
            if j==0:
                self.visuals['map_lines'][j]=visuals.LineVisual(pos=None,connect=None,color=None,method='gl',antialias=self.gui.lines_antialias)
            else:
                self.visuals['map_lines'][j] = self.visuals['map_lines'][0].view()
            self.visuals['map_lines'][j].transform = STTransform(scale=(1,-1))*self.map_transforms['aeqd']
            self.visuals['gh_lines'][j] = visuals.LineVisual(pos=None,connect=None,color=None,method='gl',antialias=self.gui.lines_antialias)
            
            self.visuals['text_hor1'][j] = visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color=np.ones(4),font_size=eval(self.font_sizes['text_hor1']),face='OpenSans-Bold',anchor_x='center',anchor_y='center')
            self.visuals['text_hor2'][j] = self.visuals['text_hor1'][j].view()
            self.visuals['text_hor1'][j].attach(self.text_hor_bottom_colorfilter, view=self.visuals['text_hor1'][j])
            self.visuals['text_hor1'][j].attach(self.text_hor_top_colorfilter, view=self.visuals['text_hor2'][j])
            if j in self.panels_vertical_ghtext:
                #For now only created for panel 0 and 5, because others don't need vertically oriented text atm.
                self.visuals['text_vert1'][j] = visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color=np.ones(4),font_size=eval(self.font_sizes['text_vert1']),face='OpenSans-Bold',rotation=90,anchor_x='center',anchor_y='center')
                self.visuals['text_vert2'][j] = self.visuals['text_vert1'][j].view()
                self.visuals['text_vert1'][j].attach(self.text_hor_bottom_colorfilter, view=self.visuals['text_vert1'][j])
                self.visuals['text_vert1'][j].attach(self.text_hor_top_colorfilter, view=self.visuals['text_vert2'][j])
            
            if j==0: 
                self.visuals['sm_pos_markers'][j]=visuals.MarkersVisual(pos=np.array([[0,0]]))
                self.visuals['radar_markers'][j]=visuals.MarkersVisual(pos=np.array([[0,0]]))
            else: 
                self.visuals['sm_pos_markers'][j] = self.visuals['sm_pos_markers'][0].view()
                self.visuals['radar_markers'][j] = self.visuals['radar_markers'][0].view()
            self.visuals['sm_pos_markers'][j].visible=False
            
        self.panel_borders_width = 1
        self.visuals['panel_borders']=visuals.LineVisual(color=self.gui.panelbdscolor/255.,method='gl',width=self.scale_pixelsize(self.panel_borders_width))
        self.visuals['titles']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',bold=True,font_size=eval(self.font_sizes['titles']),face='OpenSans',anchor_x='center',anchor_y='top')
        
        for j in range(self.max_panels):
            self.visuals['cbar'+str(j)]=visuals.ColorBarVisual(pos=[0,0],size=[1,1],cmap=self.cm2[self.crd.products[j]],orientation='right',clim=[-1,1],label_color=(0,0,0,0),border_width=self.scale_pixelsize(1))
            for i in self.visuals['cbar'+str(j)]._ticks:
                i.visible=False 
            #Set the visibility of the TextVisuals for the ticks and the label to False, to prevent that they are drawn. This is because I don't use them.
            self.visuals['cbar'+str(j)]._label.visible=False
                    
        self.visuals['cbars_ticks']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',font_size=eval(self.font_sizes['cbars_ticks']),face='OpenSans',anchor_x='center',anchor_y='center')
        self.visuals['cbars_reflines']=visuals.LineVisual(pos=None,color='black',method='gl',connect=None,width=self.scale_pixelsize(1))
        self.visuals['cbars_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',font_size=eval(self.font_sizes['cbars_labels']),bold=True,face='OpenSans',anchor_x='center',anchor_y='top')
                
        self.panel_bounds={}
        self.panel_corners={}
        self.panel_centers={j:np.array([0,0]) for j in range(self.max_panels)} #In screen coordinates
        self.clippers={j:Clipper() for j in range(self.max_panels)}
        #STTransforms for handling positioning of widgets in panels, and panning and zooming.
        # Use (approximately) the base panel y range on start-up. It is approximately, because 0.9*self.gui.screen_size() is used instead of
        # self.wcenter['main'], because the latter isn't up-to-date yet at this point in the initialisation process
        self.panels_sttransforms={j:STTransform(scale=1/self.base_range*0.5*0.9*self.gui.screen_size()[1]*np.array([1, 1])) for j in range(self.max_panels)}
        
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()  
                
        for i in range(self.max_panels):
            self.panels_sttransforms[i].dynamic=True
            for j in self.visuals_panels:
                if not i in self.visuals[j]: continue
                self.visuals[j][i].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                self.visuals[j][i].attach(self.clippers[i], view = self.visuals[j][i]) 
                #Important!!!: view = self.visuals[j][i] is required when some of the objects in self.visuals are views.
                #If view is not specified, then the clipper is always applied to the visual to which the view belongs, meaning that
                #due to nonoverlapping clippers this visual is not shown at all.

                if type(self.visuals[j][i].transform)==STTransform:
                    #This is done to prevent that the current STTransform and self.panels_sttransforms[i] are combined into one, which would cause the transform
                    #of that visual not to update when self.panels_sttransforms[i] is updated.
                    self.visuals[j][i].transform=ChainTransform(self.visuals[j][i].transform)
                self.visuals[j][i].transform=self.panels_sttransforms[i]*self.visuals[j][i].transform
                
            d = 1.5
            self.visuals['text_hor1'][i].transform = STTransform(translate=(d,d))*ChainTransform(self.visuals['text_hor1'][i].transform)
            if i in self.panels_vertical_ghtext:
                self.visuals['text_vert1'][i].transform = STTransform(translate=(d,d))*ChainTransform(self.visuals['text_vert1'][i].transform)
                                
        for j in self.visuals_global:
            #These do not need a pan-zoom transform or clipper
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    visual.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            else:
                self.visuals[j].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            
        self.coordmaps={j:self.panels_sttransforms[j].map for j in range(self.max_panels)}
        self.coordimaps={j:self.panels_sttransforms[j].imap for j in range(self.max_panels)}

        self.set_cbars()  
        self.set_maplineproperties(self.panellist)
        self.set_ghlineproperties(self.panellist)
        self.set_radarmarkers_data()
        self.set_sm_pos_markers()
        
        
        
    def physical_size_cm(self):
        #Physical size of the vispy canvas
        return np.array(self.physical_size)/(self.gui.screen_DPI())*2.54
    
    def scale_physicalsize(self,size):
        return size * self.gui.screen_physicalsize()[1]/self.gui.ref_screen_physicalsize[1]
        
    def scale_pointsize(self,size):
        return self.scale_physicalsize(size)
        
    def scale_pixelsize(self,size):
        return size*self.gui.screen_size()[1]/self.gui.ref_screen_size[1]

    def scale(self,r,array):
        array=np.asarray(array)
        array[array<0.5]=r*array[array<0.5]
        array[array>=0.5]=1-r*(1-array[array>=0.5])
        return array
    def calc_bounds_and_size(self,widget,xmin,xmax,ymin,ymax):
        print('size=',self.size)
        bounds=np.round(np.array([[xmin,xmax],[ymin,ymax]])*np.reshape(self.size,(2,1))).astype('int')
        pos=np.array([bounds[:,0],[bounds[0,0],bounds[1,1]],bounds[:,1],[bounds[0,1],bounds[1,0]]])
        center=np.mean(pos,axis=0); center[1]=self.size[1]-center[1]
        size=bounds[:,1]-bounds[:,0]
        
        self.wbounds[widget]=bounds.copy(); self.wpos[widget]=pos.copy(); self.wcenter[widget]=center.copy(); self.wsize[widget]=size.copy()
    def set_widget_sizes_and_bounds(self):
        """Sets the bounds, corner positions center positions and sizes of the 5 parts in which the canvas is divided, i.e. a left and right part for the 
        color bars, a lower and upper part for the titles, and the main part for the panels.
        
        When viewing the window in maximized state (but not full screen), then the relative bounds of the widgets are given by self.ref_bounds.
        When the window size is different, then the widths/heights of the left and right/bottom and top part of the canvas are kept constant (in pixels), 
        to keep enough space for the titles and cbars. This means that the relative bounds are different for different window sizes.
        
        The unit of the parameters is pixels.
        self.wbounds, self.wpos and self.wcenter are determined relative to an origin at the top left corner, with the y axis pointing downwards
        (as Vispy usually does).
        # self.wbounds_yu, self.wpos_yu and self.wcenter_yu are defined in a similar way, but with the origin at the bottom left corner, 
        # and the y axis pointing upwards.
        
        Bounds are given in the format [[xmin,xmax],[ymin,ymax]]
        Corner positions start with the top left corner, and are listed in counterclockwise order.
        """
        wdims = self.scale_physicalsize(self.wdims)
        panels_width = self.physical_size_cm()[0] - 2*wdims[0]
        if self.gui.show_vwp: 
            panels_width = self.physical_size_cm()[0]*(1-self.vwp_relxdim) - 2*wdims[0]
        rel_main_bounds = []
        rel_main_bounds += [[wdims[0], panels_width+wdims[0]]]
        rel_main_bounds += [[wdims[1], self.physical_size_cm()[1]-wdims[1]]]
        rel_main_bounds = np.array(rel_main_bounds) / np.reshape(self.physical_size_cm(),(2,1))
        b=rel_main_bounds
        
        if b[0,0]<b[0,1] and b[1,0]<b[1,1]:
            if not hasattr(self, 'wbounds'):
                # Do this only on initialisation, since otherwise it might happen while updating the map tiles (which runs in another thread),
                # that no specs for the main widget are available. This way at least they are, although they might be outdated.
                self.wbounds={}; self.wpos={}; self.wcenter={}; self.wsize={}
            
            self.calc_bounds_and_size('main',b[0,0],b[0,1],b[1,0],b[1,1])
            self.calc_bounds_and_size('left',0,b[0,0],0,1)
            if self.gui.show_vwp:
                self.calc_bounds_and_size('right',b[0,1],1-self.vwp_relxdim,0,1)
            else:
                self.calc_bounds_and_size('right',b[0,1],1,0,1)
            self.calc_bounds_and_size('top',b[0,0],b[0,1],0,b[1,0])   
            self.calc_bounds_and_size('bottom',b[0,0],b[0,1],b[1,1],1)
            if self.gui.show_vwp:
                self.calc_bounds_and_size('vwp',1-self.vwp_relxdim,1,0,1)     
            print(self.wsize)
        
    def determine_panellist_nrows_ncolumns(self):  
        self.panellist=tuple(self.plotnumber_to_panelnumber[self.panels][j] for j in range(0,self.panels))
        self.nrows=self.rows_panels[self.panels][self.panels-1]+1
        self.ncolumns=self.columns_panels[self.panels][self.panels-1]+1

    def get_row_col_panel(self,panel):
        p=self.panelnumber_to_plotnumber[self.panels][panel]
        return self.rows_panels[self.panels][p],self.columns_panels[self.panels][p]
        
    def get_panel_for_position(self,pos):
        rel_pos=np.array(pos)-self.wpos['main'][0] #Position relative to top left corner of the main part of the canvas
        size_panel=np.array(self.wsize['main'])/np.array([self.ncolumns,self.nrows])
        
        floor=np.floor(rel_pos/size_panel)
        if floor[0]+1>=self.ncolumns: floor[0]=self.ncolumns-1
        if floor[1]+1>=self.nrows: floor[1]=self.nrows-1
        plot=int(self.ncolumns*floor[1]+floor[0])
        
        return self.plotnumber_to_panelnumber[self.panels][plot]
    
    def set_panel_info(self):
        self.panel_centers_before=self.panel_centers.copy() #These values are needed in the function self.set_panel_sttransforms_and_clippers
        
        for j in self.panellist:
            size_panel=np.array(self.wsize['main'])/np.array([self.ncolumns,self.nrows])
            row, col=self.get_row_col_panel(j)
            topleft=self.wpos['main'][0]+size_panel*np.array([col,row]) #In screen coordinates
            bottomright=topleft+size_panel
            
            # b=np.array([topleft[0],self.size[1]-bottomright[1],size_panel[0],size_panel[1]])
            # (x, y, w, h), with y measured with upward pointing y axis, because that is necessary for the clipper! Screen coordinates are measured with 
            #the y axis pointing downwards.
            self.panel_bounds[j]=np.array([topleft[0],self.size[1]-bottomright[1],size_panel[0],size_panel[1]])
            #First corner is the top left one, and the other 3 are listed in counterclockwise order. These are given for y axis pointing downward!
            self.panel_corners[j]=topleft+np.array([[0,0],[0,size_panel[1]],size_panel,[size_panel[0],0]])
            self.panel_centers[j]=0.5*(topleft+bottomright)
    
    def set_panel_borders(self):
        #Call this function after self.set_panel_info
        panel_borders_pos=np.zeros((self.panels*5,2),dtype='float32')
        panel_borders_connect=np.ones(self.panels*5,dtype='bool')
        i=0
        a=self.scale_pixelsize(0.0)
        for j in self.panellist:
            c=self.panel_corners[j]
            #Shift the vertices of the lines that represent the edges of the panel a pixels inwards, such that the lines are
            #plotted at the edges of the panel (first/lasts pixel row/column encompassed by panel).
            panel_borders_pos[i:i+5]=np.array([c[0]+a*np.array([1,-1]),c[1]+a*np.array([1,1]),c[2]+a*np.array([-1,1]),c[3]+a*np.array([-1,-1]),c[0]+a*np.array([1,-1])])
            panel_borders_connect[i+4]=0
            i+=5
            
        self.visuals['panel_borders'].set_data(pos=panel_borders_pos,connect=panel_borders_connect)
            
    def set_panel_sttransforms_and_clippers(self):
        self.set_panel_info()

        panel_center_shift=np.array(self.panels_sttransforms[0].translate[:2])-self.panel_centers_before[0] #Always use the first panel for calculating the
        #shift of the panel center, that is required to ensure that the center of each panel shows the same geographical location as before. 
        #Using the first panel is necessary, because this is the only panel that always gets updated by this function.
        for j in self.panellist:
            self.clippers[j].bounds = tuple(self.panel_bounds[j])
            #Shift the geographical location of the center of the panels
            self.panels_sttransforms[j].translate=self.panel_centers[j]+panel_center_shift
      
        
    def calculate_vwp_relxdim(self):
        f = self.size[1]/self.size[0] / 0.5136825645035183 # Reference value
        self.vwp_relxdim = 0.255*f #The fractional width of the VWP plot relative to that of the whole canvas
            
    def on_resize(self, event=None): 
        self.dpi = self.gui.screen_DPI()
        
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        
        for j in self.visuals_order:
            if j in self.visuals_global:
                self.visuals[j].transforms.configure(canvas=self, viewport=vp)
                if hasattr(self.visuals[j], 'font_size'):
                    self.visuals[j].font_size = self.scale_pointsize(eval(self.font_sizes[j]))
            else:
                for visual in self.visuals[j].values():
                    visual.transforms.configure(canvas=self, viewport=vp)
                    if hasattr(visual, 'font_size'):
                        visual.font_size = self.scale_pointsize(eval(self.font_sizes[j]))
        
        # First resize the 5 widgets in which the canvas is divided, and then resize the visuals that reside in them.
        self.calculate_vwp_relxdim()
        self.set_widget_sizes_and_bounds()
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()
        
        # Now update all visuals with line widths, font sizes etc in order to use the new canvas dimensions and/or resolution.
        self.set_cbars(resize=True, set_cmaps=False)
        self.set_maplineproperties(self.panellist)
        self.set_radarmarkers_data()
        if self.firstplot_performed:
            if 'grid' in self.gui.lines_show: self.set_grid()
            if 'heightrings' in self.gui.lines_show: self.set_heightrings()
            if any([j in self.gui.lines_show for j in ('grid','heightrings')]):
                self.set_ghlineproperties(self.panellist); self.set_ghtextproperties(self.panellist)
            self.set_titles()
            
        self.visuals['background_map'].transform.scale=self.wsize['main']
        self.visuals['background_map'].transform.translate=self.wcenter['main']
                    
        if self.gui.show_vwp:
            self.vwp.on_resize()
                    
        self.set_draw_action('resizing')
        self.update_map_tiles_ondraw = True

    def set_draw_action(self, new_action):
        # Change self.draw_action in such a way that it combines the actions of new_action with those of any possibly existing self.draw_action 
        # Assumes that from left to right each draw action includes at least the widgets contained in the previous action.
        priority_order = {'no vwp':['panning_zooming', 'plotting', 'changing_panels', 'updating_cbars', 'resizing'],
                          'vwp':['vwp_only', 'plotting_vwp']}
        if not hasattr(self, 'draw_action') or self.draw_action is None:
            self.draw_action = new_action
        elif ('vwp' in new_action) != ('vwp' in self.draw_action):
            self.draw_action = 'plotting_vwp'
        elif ('vwp' in new_action) == ('vwp' in self.draw_action):
            vwp_mode = 'vwp' if 'vwp' in new_action else 'no vwp'
            i1, i2 = [priority_order[vwp_mode].index(j) for j in (self.draw_action, new_action)]
            self.draw_action = priority_order[vwp_mode][i1 if i1 > i2 else i2]

    def on_draw(self, ev):    
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable()
        if self.update_map_tiles_ondraw:
            if self.draw_action in ('panning_zooming', 'resizing'):
                self.set_timer_update_map_tiles()
            else:
                self.update_map_tiles(separate_thread=False, draw_map=True)
        self.update_map_tiles_ondraw = False
                
        if self.draw_action=='panning_zooming':
            self.draw_widgets=['main']
        elif self.draw_action in ('plotting','changing_panels'): #When plotting a VWP it is desired to also plot
            #the left and right widgets, since the title might extend a bit into those widgets.
            self.draw_widgets=['main','bottom','top']
        elif self.draw_action == 'vwp_only':
            self.draw_widgets = ['vwp']
        else: 
            self.draw_widgets=self.widgets.copy() #Draw all widgets
            if self.gui.show_vwp:
                self.draw_widgets += ['vwp']
            
        if not self.gui.use_scissor: self.draw_widgets=self.widgets
                             
        
        if self.starting or self.draw_action in ('resizing',None) or self.draw_widgets!=self.draw_widgets_before:
            topleft_corner=np.min([self.wpos[j][0] for j in self.draw_widgets],axis=0)
            bottomright_corner=np.max([self.wpos[j][2] for j in self.draw_widgets],axis=0)
            self.visuals['background'].transform.translate=np.mean([topleft_corner, bottomright_corner], axis = 0)
            self.visuals['background'].transform.scale=bottomright_corner-topleft_corner
        self.visuals['background'].draw()
                 
        if self.starting or self.draw_action in ('resizing',None) or self.draw_widgets!=self.draw_widgets_before:
            if self.starting: gloo.clear('white')
            GL.glDrawBuffer(GL.GL_FRONT_AND_BACK)
                        
            
        #Only plot visuals in the widgets for the main radar window here, not those in other windows such as the vwp window.
        #These are plotted in their own classes.
        visuals_to_plot = [self.visuals_widgets[j] for j in self.draw_widgets if j in self.widgets]
        if len(visuals_to_plot) > 0:
            visuals_to_plot=np.concatenate(visuals_to_plot)
        for j in self.visuals_order:
            if not j in visuals_to_plot: continue
            
            if j in self.visuals_global:
                if isinstance(self.visuals[j], list):
                    for visual in self.visuals[j]:
                        if visual.visible:
                            visual.draw()
                elif self.visuals[j].visible:
                    self.visuals[j].draw()
            else:    
                for i in self.panellist:
                    if i in self.visuals[j] and self.visuals[j][i].visible:
                        self.visuals[j][i].draw()
                        
        if self.firstdraw_map and self.gui.mapvisibility:
            #It is important to draw the map for all panels during the first draw, because otherwise the scale gets screwed up when changing the
            #number of panels.
            for j in range(self.max_panels):
                if not j in self.panellist:
                    self.visuals['map'][j].draw()
            self.firstdraw_map = False
            
        for j in self.panellist:
            if self.data_attr['proj'].get(j, None) == 'pol' and j not in self.ref_radial_bins:
                # These reference values are needed in self.set_newdata when changing the scale of the polar image transform
                self.ref_azimuthal_bins[j], self.ref_radial_bins[j] = self.dsg.data[j].shape
            
        if 'vwp' in self.draw_widgets:
            #Plot the vwp visuals
            self.vwp.on_draw()
        
        if self.draw_action in ('plotting', 'plotting_vwp', 'changing_panels') and self.gui.continue_savefig:
            self.swap_buffers() #swapping the buffers causes the offscreen buffer to become visible, which must be the case before saving the figure.
            pytime.sleep(0.01)
            self.gui.savefig(select_filename=False)
          
        self.draw_action=None
        self.starting=False
        self.draw_widgets_before=self.draw_widgets.copy()
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(30)  
        
            
    def set_panels_sttransforms_manually(self, scale=None, translate=None, panzoom_action=True, draw=True):
        for j in range(self.max_panels):
            if not scale is None: self.panels_sttransforms[j].scale = scale
            if not translate is None: self.panels_sttransforms[j].translate = translate
        self.set_panel_sttransforms_and_clippers()
            
        if panzoom_action:
            self.dsg.time_last_panzoom=pytime.time()
        
        if draw:
            if any([j in self.gui.lines_show for j in ('grid','heightrings')]) and self.firstplot_performed: 
                self.update_gridheightrings()
                
            self.set_draw_action('panning_zooming')
            self.update_map_tiles(separate_thread = False, draw_map = True)
            self.update()         
        
    def reset_panel_view(self, reset_zoom=True, reset_center=True):
        pc = self.panel_centers[0]
        ps, pt = self.panels_sttransforms[0].scale[:2], self.panels_sttransforms[0].translate[:2]
        if reset_zoom:
            new_scale = 1/self.base_range*self.wcenter['main'][1]*np.array([1, 1])
            if reset_center:
                self.set_panels_sttransforms_manually(new_scale, pc)
            else:
                zoom = new_scale/ps
                new_translate = pc-(pc-pt)*zoom
                self.set_panels_sttransforms_manually(new_scale, new_translate)
        else:
            self.set_panels_sttransforms_manually(translate=pc)
            
    def translation_dist_km(self, delta_time):
        # Is also used in self.dsg.check_need_scans_change
        SM = self.gui.stormmotion
        SM = -SM[1]*np.array([np.sin(np.deg2rad(SM[0])), np.cos(np.deg2rad(SM[0]))])
        return SM*delta_time/1e3
    def move_view_with_storm(self, panellist):
        # print('Move_view_with_storm')
        delta_times = {}
        for j in panellist:
            if self.gui.switch_to_case_running:
                try:
                    dt = ft.datetimediff_s(self.gui.current_case['scandatetime'], self.data_attr['scandatetime'][0])
                except Exception:
                    dt = ft.datetimediff_s(self.gui.current_case['datetime'], self.crd.date+self.crd.time)
            else:
                try:
                    dt = ft.datetimediff_s(self.data_attr_before['scandatetime'][j], self.data_attr['scandatetime'][j])
                except Exception: 
                    dt = 0.
            delta_times[j] = dt

        max_dt = max(map(abs, delta_times.values()))
        if max_dt == 0. or (self.ani.continue_type[:3] != 'ani' and max_dt > self.max_delta_time_move_view_with_storm) or (
        self.ani.continue_type[:3] == 'ani' and max_dt > 60*(self.ani.duration+60)): # Add some margin, since actual duration might differ somewhat
            # Prevent movement in case of very big timesteps, that e.g. occur when switching to the next day with available data
            return False
            
        scale = np.array(self.panels_sttransforms[0].scale[:2])
        for j in panellist:
            distance_move = -self.translation_dist_km(delta_times[j])*np.array([1, -1])*scale
            self.panels_sttransforms[j].move(distance_move) # Moving needs to occur in opposite direction

        # self.dsg.time_last_panzoom=pytime.time()
        self.update_map_tiles_ondraw = True
        return True
        
    def on_mouse_wheel(self, ev):
        self.t = pytime.time()
        if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0]: return
        
        zoomfactor=(1 + self.zoomfactor_vispy) ** (ev.delta[1] * 30)
        
        pos=np.array(ev.pos)
        selected_panel=self.get_panel_for_position(pos)
        rel_pos=pos-self.panel_centers[selected_panel]
        for j in range(self.max_panels): #Zooming the transforms for all panels is necessary, to ensure that all panels keep showing the same area
            self.panels_sttransforms[j].zoom((zoomfactor,zoomfactor),center=self.panel_centers[j]+rel_pos,mapped=True)
            
        self.dsg.time_last_panzoom=pytime.time()
        
        if any([j in self.gui.lines_show for j in ('grid','heightrings')]) and self.firstplot_performed: 
            if self.gui.showgridheightrings_panzoom: 
                #This is needed to ensure that updating the heightrings occurs after the zooming has been performed
                self.update_gridheightrings(ev)
            else:
                if not self.gridheightrings_removed:
                    self.remove_gridheightrings()
                self.set_timer_setback_gridheightrings()
                
        self.update_data_readout()
            
        self.set_draw_action('panning_zooming')
        self.update_map_tiles_ondraw = True
        self.update()
        
    def on_mouse_press(self, ev):
        if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0] and not (
        self.gui.show_vwp and ft.point_inside_rectangle(ev.pos,self.wpos['vwp'])[0]):
            return
        self.last_mousepress_pos=np.array(ev.pos)
        self.mouse_moved_after_press=False
        
        if self.gui.show_vwp and ft.point_inside_rectangle(ev.pos,self.wpos['vwp'])[0]:
            #No more steps needed in this case
            return
        
        if ev.button==1: self.mouse_hold_left=True; self.gridheightrings_removed=False
        else: self.mouse_hold_right=True
        self.mouse_hold=True
        
        self.check_presence_near_radars(ev.pos)
        self.check_presence_near_pos_markers(ev.pos)
        
        self.panel=self.get_panel_for_position(ev.pos)
        
    def on_mouse_release(self, ev):     
        self.mouse_hold_left=False; self.mouse_hold_right=False; self.mouse_hold=False
        
        if self.firstplot_performed and any([j in self.gui.lines_show for j in ('grid','heightrings')]) and\
        self.mouse_moved_after_press:
            if self.gui.showgridheightrings_panzoom: 
                self.update_gridheightrings(ev)
            else:
                self.set_timer_setback_gridheightrings()
                
        self.update_data_readout()
        
        #button=1 refers to the left mouse button, 2 to the right one
        if ev.button==1:
            self.mouse_hold_left=False
            if self.radar_mouse_selected not in (None, self.crd.selected_radar) and not self.mouse_moved_after_press:
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:
                    self.crd.change_selected_radar(self.radar_mouse_selected)
                else:
                    self.crd.change_radar(self.radar_mouse_selected)
        elif ev.button==2:
            self.mouse_hold_right=False
            if self.gui.need_rightclickmenu: 
                self.gui.showrightclickMenu(self.gui.rightmouseclick_Qpos)
 
    def on_mouse_move(self, ev): 
        if hasattr(self, 'previous_mouse_position') and (ev.pos == self.previous_mouse_position).all():
            # With my voice dictation software it sometimes happens that mouse move signals are emitted while the mouse is in fact not moving
            return
        self.previous_mouse_position=np.array(ev.pos)
        self.mouse_moved_after_press=True
        
        if self.mouse_hold_left:
            for j in self.panellist:
                if j == 0:
                    self.panels_sttransforms[0].move(np.array(ev.pos)-ev.last_event.pos)
                else:
                    self.panels_sttransforms[j].translate = self.panels_sttransforms[0].translate[:2]+(self.panel_centers[j]-self.panel_centers[0])

        elif self.mouse_hold_right:
            p1c = np.array(ev.last_event.pos)[:2]
            p2c = np.array(ev.pos)[:2]
            zoomfactor = (1 + self.zoomfactor_vispy) ** ((p2c-p1c) * np.array([1, -1]))
            zoomfactor=zoomfactor[0]*zoomfactor[1]
            
            pos=ev.press_event.pos
            selected_panel=self.get_panel_for_position(pos)
            rel_pos=pos-self.panel_centers[selected_panel]
            for j in range(self.max_panels): #Zooming the transforms for all panels is necessary, to ensure that all panels keep showing the same area
                self.panels_sttransforms[j].zoom((zoomfactor,zoomfactor),center=self.panel_centers[j]+rel_pos,mapped=True)
                                
        if self.mouse_hold:
            #Used in the function self.dsg.check_need_scans_change.
            self.dsg.time_last_panzoom=pytime.time()
            
            if self.timer_setback_gridheightrings_running:
                #It is possible that a timer is running when moving the mouse just after using the mouse wheel (where the timer is set), which
                #is stopped here.
                self.timer_setback_gridheightrings.stop()
            if self.firstplot_performed and not self.gui.showgridheightrings_panzoom and (
            any([j in self.gui.lines_show for j in ('grid','heightrings')])) and not self.gridheightrings_removed:
                self.remove_gridheightrings()
                      
            self.set_draw_action('panning_zooming')
            self.update_map_tiles_ondraw = True
            self.update()
        else:
            if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0]:
                self.gui.plotwidget.setCursor(Qt.ArrowCursor)
                return
            self.last_mouse_pos_px = ev.pos
            self.check_presence_near_radars(ev.pos)
            self.check_presence_near_pos_markers(ev.pos)
            self.update_data_readout()
            
    def update_data_readout(self):
        pos = self.last_mouse_pos_px
        if pos is None:
            pos = self.wcenter['main']
        
        if not ft.point_inside_rectangle(pos,self.wpos['main'])[0]: return
        
        panel=self.get_panel_for_position(pos)
        xy_coord=self.screencoord_to_xy(pos)
        
        radius=np.linalg.norm(xy_coord)
        
        if self.firstplot_performed:
            try:
                product=self.data_attr['product'][panel]
                if not product in gv.plain_products: 
                    scanangle=self.data_attr['scanangle'][panel]
                    
                if self.data_attr['proj'][panel] == 'pol':
                    radial_res=self.data_attr['radial_res'][panel]
                    azimuthal_res=self.data_attr['azimuthal_res'][panel]
                    azimuth=ft.azimuthal_angle(xy_coord, deg=True)
                    row=int(np.floor(np.mod(azimuth-self.dsg.data_azimuth_offset[panel], 360)/azimuthal_res))
                    if self.dsg.data[panel].shape[0] > self.data_attr['azimuthal_bins'][panel]:
                        row += 1 #+1 for the added radials for interpolation
                    
                    if product in gv.plain_products or scanangle == 90.: 
                        col=int(np.floor((radius-self.dsg.data_radius_offset[panel])/radial_res))
                        self.cursor_elevation='--' if product in gv.plain_products else radius
                    else:
                        sr, self.cursor_elevation = ft.var1_to_var2(radius, scanangle, 'gr+theta->sr+h')
                        col = int(np.floor((sr-self.dsg.data_radius_offset[panel])/radial_res))
                else:
                    res = self.data_attr['res'][panel]
                    xy_bins = self.data_attr['xy_bins'][panel]
                    row = int(0.5*xy_bins-np.ceil(xy_coord[1]/res))
                    col = int(0.5*xy_bins+np.floor(xy_coord[0]/res))
                    self.cursor_elevation = '--'
            except Exception:
                #Occurs e.g. when panel not in self.data_attr['scanangle'], which is the case when the panel is empty.
                self.cursor_elevation='--'
                
            try:
                n_bits=gv.products_data_nbits[product]
                pm_lim=gv.products_maxrange_masked[product]
                cursor_datavalue_int=self.dsg.data[panel][row,col]
                if cursor_datavalue_int==self.masked_values_int[product]:
                    self.cursor_datavalue='--'
                else:
                    self.cursor_datavalue=ft.convert_uint_to_float(cursor_datavalue_int,n_bits,pm_lim)
                    self.cursor_datavalue*=self.scale_factors[product]
                    if product=='r':
                        self.cursor_datavalue=10**self.cursor_datavalue
            except Exception: self.cursor_datavalue='--'   
            try:
                min_value, max_value = [ft.convert_uint_to_float(j, n_bits, pm_lim)*self.scale_factors[product] for j in self.get_min_max_in_view(panel)]
            except Exception:
                min_value, max_value = '--', '--'
        else:
            self.cursor_elevation='--'; self.cursor_datavalue='--'
            min_value, max_value = '--', '--'
                                   
        self.cursor_latlon=ft.aeqd(gv.radarcoords[self.crd.radar],xy_coord,inverse=True)
        self.cursor_radius=radius if not self.gui.sm_marker_present else \
        ft.calculate_great_circle_distance_from_xy(gv.radarcoords[self.crd.radar], xy_coord, self.gui.sm_marker_position)
        
        lat_text='%6.6s' % str(np.around(self.cursor_latlon[0]*1000)/1000)
        lon_text='%6.6s' % str(np.around(self.cursor_latlon[1]*1000)/1000)
        x_text, y_text = '%5.5s' % xy_coord[0], '%5.5s' % xy_coord[1]
        r_text='%5.5s' % str(np.around(self.cursor_radius*1000)/1000)
        h_text='%5.5s' % str(np.around(self.cursor_elevation*1000)/1000) if self.cursor_elevation!='--' else '%5.5s' % self.cursor_elevation
        factor=1000 if self.cursor_datavalue!='--' and np.abs(self.cursor_datavalue)<0.01 else 100
        data_text='%5.5s' % (str(np.around(self.cursor_datavalue*factor)/factor) if self.cursor_datavalue!='--' else self.cursor_datavalue)+'/'+\
                  '%5.5s' % (str(np.around(min_value*factor)/factor) if min_value!='--' else min_value)+'/'+\
                  '%5.5s' % (str(np.around(max_value*factor)/factor) if max_value!='--' else max_value)
        try: #An error occurs when product is not defined, as is the case when self.data_attr['product'][panel] does not exist.
            product_unit = self.productunits[product]
        except: product_unit = ''
        self.datareadout_text='('+lat_text+', '+lon_text+'), ('+x_text+', '+y_text+'), r='+r_text+' km, h='+h_text+' km, '+data_text+' '+product_unit
        self.gui.set_textbar()
            
    def screencoord_to_xy(self, pos, panel=None):
        if len(pos.shape)>1:
            xy_coord=[]
            for j in pos:
                panel = self.get_panel_for_position(j) if panel is None else panel
                xy_coord.append(self.coordimaps[panel](j)[:2])
            return np.array(xy_coord)*np.array([1,-1]) # y-coordinate was reversed.
        else:
            panel = self.get_panel_for_position(pos) if panel is None else panel
            return np.array(self.coordimaps[panel](pos)[:2])*np.array([1,-1]) # y-coordinate was reversed.
        
    def xycoord_to_screen(self, panel, pos):
        if len(pos.shape)>1:
            screen_coord=[]
            for j in pos:
                screen_coord.append(self.coordmaps[panel](j*np.array([1,-1]))[:2]) # reverse y-coordinate.
            return np.array(screen_coord)
        else:
            return np.array(self.coordimaps[panel](pos*np.array([1,-1]))[:2]) # reverse y-coordinate.
        
    def get_in_view_mask_specs(self, data, panel):
        specs = str(self.corners[panel])+str(data.shape)+str(self.data_attr['radial_res'][panel] if self.data_attr['proj'][panel] == 'pol' else
                                                      self.data_attr['res'][panel])
        if self.data_attr['proj'][panel] == 'pol':
            specs += str(self.data_attr['scanangle'][panel])+str(self.dsg.data_radius_offset[panel])+str(self.dsg.data_azimuth_offset[panel])
        return specs
    def get_min_max_in_view(self, panel):
        data = self.dsg.data[panel]
        if self.data_attr['proj'][panel] == 'pol' and data.shape[0] > self.data_attr['azimuthal_bins'][panel]:
            data = data[1:-1] # Exclude the 2 extra rows for interpolation
        
        self.get_corners()
        if self.get_in_view_mask_specs(data, panel) != self.in_view_mask_specs:
            if self.data_attr['proj'][panel] == 'pol':
                dr, theta = self.data_attr['radial_res'][panel], self.data_attr['scanangle'][panel]
                r = ft.var1_to_var2(self.dsg.data_radius_offset[panel] + dr*(0.5+np.arange(data.shape[1], dtype='float32')), 
                                    theta, 'sr+theta->gr')
                da = self.data_attr['azimuthal_res'][panel]
                azi = np.deg2rad(self.dsg.data_azimuth_offset[panel] + da*(0.5+np.arange(data.shape[0], dtype='float32')))
                x = r[np.newaxis,:]*np.sin(azi[:,np.newaxis])
                y = r[np.newaxis,:]*np.cos(azi[:,np.newaxis])
            else:
                xy_bins = self.data_attr['xy_bins'][panel]
                res = self.data_attr['res'][panel]
                coords = res*(0.5+np.arange(-0.5*xy_bins, 0.5*xy_bins, dtype='float32'))
                x, y = np.meshgrid(coords, coords[::-1], copy=False)
            
            corners = self.corners[panel]
            self.in_view_mask = (x >= corners[0][0]) & (x  <= corners[-1][0]) & (y >= corners[1][1]) & (y <= corners[0][1])
            self.in_view_mask_specs = self.get_in_view_mask_specs(data, panel)
            
        in_view = data[self.in_view_mask & (data != self.masked_values_int[self.crd.products[panel]])]
        return in_view.min(), in_view.max()
        
    def get_max_dist_mouse_to_marker(self, f=1):
        self.get_corners()
        return f*15*self.ydim*self.nrows/1e3
    
    def check_presence_near_radars(self, pos):
        xy_mouse = self.screencoord_to_xy(pos)
        distances_to_radars = np.linalg.norm(self.radarcoords_xy-xy_mouse, axis=1)
        min_distance_index = np.argmin(distances_to_radars)
        max_distance = self.get_max_dist_mouse_to_marker()
        if min(distances_to_radars) < max_distance:
            self.radar_mouse_selected = gv.radars_all[min_distance_index]
            self.gui.set_textbar()
            if self.radar_mouse_selected != self.crd.selected_radar:
                self.gui.plotwidget.setCursor(Qt.PointingHandCursor)
        else: 
            self.radar_mouse_selected = None
            self.gui.plotwidget.setCursor(Qt.ArrowCursor)
            
    def check_presence_near_pos_markers(self, pos):
        if len(self.gui.pos_markers_positions):
            xy_mouse = self.screencoord_to_xy(pos)
            xy_markers = np.array(self.gui.pos_markers_positions)
            distances_to_markers = np.linalg.norm(xy_markers-xy_mouse, axis=1)
            min_distance_index = np.argmin(distances_to_markers)
            max_distance = self.get_max_dist_mouse_to_marker()
            if distances_to_markers[min_distance_index] < max_distance:
                self.marker_mouse_selected_index = min_distance_index
            else:
                self.marker_mouse_selected_index = None
            
    def update_gridheightrings(self,event=None):
        if 'grid' in self.gui.lines_show: self.set_grid()
        if 'heightrings' in self.gui.lines_show: self.set_heightrings()
        self.set_ghlineproperties(self.panellist)
        self.set_ghtextproperties(self.panellist)
        self.set_draw_action('panning_zooming')
        self.update()
        
        if self.gui.view_nearest_radar:
            self.crd.switch_to_nearby_radar(1)
    
    def remove_gridheightrings(self):
        self.postpone_plotting_gridheightrings=True
        self.gridheightrings_removed=True
        self.set_ghlineproperties(self.panellist)
        for j in range(self.max_panels):
            if j in self.visuals['text_hor1']:
                self.visuals['text_hor1'][j].visible=False
                self.visuals['text_hor2'][j].visible=False
            if j in self.visuals['text_vert1']:
                self.visuals['text_vert1'][j].visible=False
                self.visuals['text_vert2'][j].visible=False   
            
    def setback_gridheightrings(self):
        self.timer_setback_gridheightrings_running=False
        self.postpone_plotting_gridheightrings=False
        self.gridheightrings_removed=False
        # self.set_ghlineproperties(self.panellist)
        for j in range(self.max_panels):
            if j in self.panels_horizontal_ghtext:
                self.visuals['text_hor1'][j].visible=True
                self.visuals['text_hor2'][j].visible=True
            if j in self.panels_vertical_ghtext:
                self.visuals['text_vert1'][j].visible=True
                self.visuals['text_vert2'][j].visible=True 
                            
        self.update_gridheightrings()
        
    def set_timer_setback_gridheightrings(self):
        if self.timer_setback_gridheightrings_running:
            self.timer_setback_gridheightrings.stop()
        self.timer_setback_gridheightrings_running=True
        self.postpone_plotting_gridheightrings=True
        #Time is in ms for the timer, not s
        self.timer_setback_gridheightrings=QTimer()
        self.timer_setback_gridheightrings.setSingleShot(True)
        self.timer_setback_gridheightrings.timeout.connect(self.setback_gridheightrings_signal.emit)
        self.timer_setback_gridheightrings.start(int(self.gui.showgridheightrings_panzoom_time*1000.))
        

            
    def change_interpolation(self):
        self.use_interpolation = not self.use_interpolation
        panellist = [j for j in self.panellist if self.crd.products[j] in ('z','a','m')]
        self.set_newdata(panellist)
        
    def set_interpolation(self):
        for j in self.panellist:
            try:
                if self.use_interpolation and self.data_attr['product'][j] in gv.products_with_interpolation:                
                    self.visuals['radar_polar'][j].interpolation='bilinear'
                    self.visuals['radar_cartesian'][j].interpolation='bilinear'
                else: 
                    self.visuals['radar_polar'][j].interpolation='nearest'
                    self.visuals['radar_cartesian'][j].interpolation='nearest'
            except Exception: # Happens when j not in self.data_attr['product']
                continue
        self.update()
            
    def set_radarmarkers_data(self):
        # Selected radar should be drawn last, in order to always plot it on top.
        face_colors = [self.gui.radar_colors['Automatic download + selected' if self.crd.selected_radar in self.gui.radars_automatic_download else 'Selected']]
        radars = [self.crd.selected_radar]
        for j in self.gui.radars_automatic_download:
            if j != self.crd.selected_radar:
                face_colors.append(self.gui.radar_colors['Automatic download'])
                radars.append(j)
        for j in gv.radars_all:
            if not j in self.gui.radars_automatic_download+[self.crd.selected_radar]:
                face_colors.append(self.gui.radar_colors['Default'])
                radars.append(j)
        # Use slightly different marker sizes for different radar wavelength bands, the biggest for S-band
        scale_fac = {'S':1.1, 'C':1, 'X':1/1.1}
        coords_xy, sizes = [], []
        for i,j in enumerate(radars):
            if j in self.gui.radars_download_older_data:
                #Blend the color with black, to get a darker color that indicates that download of older data is being performed.
                face_colors[i] = ft.blend_rgba_colors_1D(np.append(face_colors[i],0), np.array([64,64,64,0]), 0.5)[:3]
            coords_xy.append(self.radarcoords_xy[gv.radars_all.index(j)])
            sizes.append(scale_fac[gv.radar_bands[j]]*self.scale_pixelsize(self.gui.radar_markersize))
        # Reverse array entries in order to put selected radar last            
        coords_xy, face_colors, sizes = np.array(coords_xy)[::-1], np.array(face_colors)[::-1], np.array(sizes)[::-1]
        
        self.visuals['radar_markers'][0].set_data(pos=coords_xy*np.array([1,-1]),symbol='disc',size=sizes,edge_width=1,face_color=face_colors/255.,edge_color='black')

            
    def set_newdata(self,panellist,change_datetime=False,source_function=None,set_data=True,plain_products_parameters_changed=False,apply_storm_centering=False):
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable()
        """
        This is the function that handles changes in radar, date, time and dataset. Other functions, from which this function is called, change the
        selected radar, date, time and dataset, but these changes are only actually realized when no exceptions are raised during the evaluation of this 
        function.       
        At first, changes in radar and dataset are handled, which require an update of the lists with scans information, and in the case of a change
        of radar changes in the map.
        Secondly the need to update the colormaps and colorbars is evaluated, and they are updated if needed. After that the new data is imported by
        calling self.dsg.get_data. In the case of exceptions, parameters are set back to their previous values.
        Finally the data in the image visual gets updated, as is its transform and colormap when required.  
        
        set_data=False if this function is only called to obtain the radar volume attributes, which are then returned as the second output argument, next
        to error_received.
        """
        self.crd.date=self.crd.selected_date; self.crd.time=self.crd.selected_time
                
        if self.firstplot_performed and set_data:
            #Save data about the previous plot, which is used in the function self.dsg.check_need_scans_change.
            self.get_corners()
            #self.crd.radar en self.crd.dataset have not yet been updated here, so I don't need to use self.crd.before_variables['radar'] etc.
            radar_dataset=self.crd.radar+(' '+self.crd.dataset if self.crd.radar in gv.radars_with_datasets else '')
            #self.crd.scans can already have been updated, so use self.scans_before here.
            self.dsg.scans_radars[radar_dataset]={'time':pytime.time(),'panellist':self.panellist,'scans':self.scans_before.copy(),
                                                  'scanangles_all':self.dsg.scanangles_all_m.copy(),'corners':self.corners.copy()}
            
        if set_data:
            radar_changed = self.crd.radar != self.crd.selected_radar
            dataset_changed = self.crd.dataset != self.crd.selected_dataset
            
            if radar_changed:
                old_radar=self.crd.radar; self.crd.radar=self.crd.selected_radar #self.crd.radar must be updated before calling self.change_map_center,
                #because there self.update_combined_lineproperties is called, which requires self.crd.radar to be updated.
                self.change_map_center(old_radar,self.crd.radar)
                self.set_radarmarkers_data()
                if self.gui.stormmotion[1] != 0.:
                    self.gui.update_stormmotion_change_radar()
            self.crd.dataset=self.crd.selected_dataset
            
            if self.gui.switch_to_case_running:
                panel_center = self.gui.current_case['panel_center']
                center_shift = self.panel_centers[0] - panel_center
                scale = self.gui.current_case['scale']
                trans = self.gui.current_case['translate'][:2]+center_shift
                if not self.gui.cases_use_case_zoom:
                    # In this case the original zoom level should be maintained (not the one saved with the case). This is achieved by first 
                    # changing view to that saved with the case, and then zooming back to the original zoom level using the zoom technique from 
                    # https://github.com/vispy/vispy/blob/main/vispy/visuals/transforms/linear.py
                    zoom = self.panels_sttransforms[0].scale[0]/scale[0]
                    scale = scale*zoom # Don't multiply in-place, since that would change self.gui.current_case['scale']
                    trans = self.panel_centers[0] - (self.panel_centers[0] - trans[:2]) * zoom
                self.set_panels_sttransforms_manually(scale, trans, panzoom_action=False)
                            
            self.changed_colortables=self.set_cmaps([self.crd.products[j] for j in self.panellist])
            #Updates colormaps if the corresponding color tables have been changed.
            
            if radar_changed or dataset_changed or self.crd.changing_subdataset or change_datetime:
                #Update the 'before' variables
                self.update_before_variables(radar_changed,self.crd.changing_subdataset,dataset_changed)
        else: 
            radar_changed=False; dataset_changed=False
            save_radar=self.crd.radar; save_dataset=self.crd.dataset
            #self.crd.radar and self.crd.dataset are restored after checking the presence of data.
            self.crd.radar=self.crd.selected_radar; self.crd.dataset=self.crd.selected_dataset
            
        
        self.data_empty_before = self.data_empty.copy()
        self.data_attr_before = copy.deepcopy(self.data_attr)
        if set_data and self.crd.process_datetimeinput_running:
            #Reset self.dsg.data, to prevent that data for the old radar/dataset is shown when new data is unavailable for a particular panel.
            #Also reset it when changing the time through self.crd.process_datetimeinput, because otherwise the old data could differ greatly
            #in time from the curently selected date and time.
            self.dsg.data={j:-1e6*np.ones((360,1)).astype('float32') for j in range(self.max_panels)}
            self.data_empty={j:True for j in range(self.max_panels)}
            self.data_isold={j:True for j in range(self.max_panels)}
            self.data_attr = {j:{} for j in self.data_attr}
            
        t=pytime.time()
        try:
            # print('getting data', self.crd.lrstep_beingperformed, 0 in self.data_attr_before['scandatetime'])
            returns=self.dsg.get_data(panellist,radar_changed,dataset_changed, set_data)
            if set_data:
                data_changed, total_file_size = returns
            else:
                retrieved_attrs, total_file_size = returns
                
            if self.crd.lrstep_beingperformed and self.crd.change_radar_running and 0 in self.data_attr_before['scandatetime']:
                scandate = ft.get_scandate(self.crd.date, self.crd.time, self.dsg.scantimes[0])
                scandatetime = scandate+ft.scantime_to_flattime(self.dsg.scantimes[0])
                if np.sign(int(scandatetime)-int(self.data_attr_before['scandatetime'][0])) != np.sign(self.crd.lrstep_beingperformed):
                    self.data_attr = copy.deepcopy(self.data_attr_before)
                    self.data_empty = self.data_empty_before.copy()
                    print('again')
                    # from_timer can be set to True, since it's not needed to check for sleep time again
                    return self.crd.process_keyboardinput(self.crd.lrstep_beingperformed, from_timer=True)
                # Given the return statement above, only one subcall of self.crd.process_keyboardinput will continue here, which is the last.
                # But for this last subcall radar_changed will have been set to False, since the change of radar took place in a previous subcall.
                # This is not desired, since a change of radar has in fact taken place. So set it to True here.
                radar_changed = True
                
        except Exception as e:
            # print('exxie', self.dsg.scannumbers_all, self.dsg.scanangles_all, self.crd.products, self.crd.scans)
            traceback.print_exception(type(e), e, e.__traceback__)
            description = str(e.args[0])
            if type(e) == MemoryError:
                self.gui.set_textbar('Not enough memory available to read file.', 'red', 1)
            elif description == 'get_scans_information':
                # Add datetime to self.crd.filedatetimes_errors and thereby exclude it in self.crd.get_filedatetimes.
                # Exception is when the total file size changes, in which case it won't be excluded. This total file size is therefore
                # saved with the datetime.
                ft.create_subdicts_if_absent(self.crd.filedatetimes_errors, [self.crd.directory, self.crd.date+self.crd.time])
                total_files_size = self.dsg.get_total_volume_files_size()
                self.crd.filedatetimes_errors[self.crd.directory][self.crd.date+self.crd.time] = total_files_size
                self.crd.determine_list_filedatetimes()
                self.gui.set_textbar('Could not read file(s). Try a different time.', 'red', 1)
            elif 'none' in description.lower():
                raise Exception('None!!!')
            else:                
                self.gui.set_textbar('Could not obtain requested product and/or scan. Try a another option.', 'red', 1)
            # For set_data=False return retrieved attrs, which is now an empty dictionary
            
            retrieved_attrs = {}
            data_changed = {j:False for j in panellist}
            # In the case of errors we usually continue below, for example in order to update the panels with an empty data array when
            # switching from radar (in which case it's undesired to keep showing the previous data, since it's for the previous radar).
            # But return when self.firstplot_performed = False, since otherwise errors occur below.
            if set_data and not self.firstplot_performed:
                return
        # print(pytime.time()-t, 't_total_import')

        
        if set_data:            
            for j in panellist:
                self.data_isold[j] = not data_changed[j]
                if not data_changed[j]:
                    continue
                
                self.data_empty[j]=False
                # Remove any previous dictionary entries for this panel, since params are different between plain and non-plain products. 
                # And the number of panels in self.data_attr['scanangle'] is for example used further down.
                ft.remove_keys_from_all_subdicts(self.data_attr, [j])
                self.data_attr['product'][j] = self.crd.products[j]
                self.data_attr['scantime'][j] = self.dsg.scantimes[j]
                scandate = ft.get_scandate(self.crd.date, self.crd.time, self.dsg.scantimes[j])
                self.data_attr['scandatetime'][j] = scandate+ft.scantime_to_flattime(self.dsg.scantimes[j])
                if not self.crd.products[j] in gv.plain_products:
                    p = gv.i_p[self.crd.products[j]]
                    self.data_attr['scanangle'][j]=self.dsg.scanangle(p, self.crd.scans[j], self.dsg.scannumbers_forduplicates[self.crd.scans[j]])
                    self.data_attr['radial_bins'][j]=self.dsg.radial_bins_all[p][self.crd.scans[j]]
                    self.data_attr['radial_res'][j]=self.dsg.radial_res_all[p][self.crd.scans[j]]
                    self.data_attr['azimuthal_bins'][j]=self.dsg.data[j].shape[0]
                    self.data_attr['azimuthal_res'][j]=360/self.data_attr['azimuthal_bins'][j]
                    self.data_attr['proj'][j] = 'pol'
                else:
                    for param in self.data_attr:
                        if param in self.dp.meta_PP[self.crd.products[j]]:
                            self.data_attr[param][j]=self.dp.meta_PP[self.crd.products[j]][param]


        if not set_data:
            self.crd.radar=save_radar; self.crd.dataset=save_dataset
        else:            
            move_view_with_storm = self.gui.use_storm_following_view and apply_storm_centering
            # print('move', move_view_with_storm, apply_storm_centering)
            if move_view_with_storm:
                panellist_move = [j for j in panellist if data_changed[j]]
                if panellist_move:
                    move_view_with_storm = self.move_view_with_storm(panellist_move)
                
            if source_function!=self.change_panels:
                # If source_function==self.change_panels, then it is always desired to update grid and height rings because of changed 
                # panel dimensions. In that case that is done in a simple manner in the function itself.
                
                gh_panellist = panellist_move if move_view_with_storm else panellist
                # if self.data_empty has changed, then the grid and heightrings should always be updated, because for at least 1 panel they change
                # from visible to invisible or vice versa.
                gh_changed = any([self.data_empty_before[j] != self.data_empty[j] for j in panellist]) 
                #When self.timer_setback_gridheightrins_running=True, then heightrings are currently not visible, and  will be plotted after the
                #timer is finished. In this case it is therefore not desired to update them here, as this will make them visible.
                if not self.firstplot_performed or (not self.postpone_plotting_gridheightrings and 
                source_function==self.crd.process_datetimeinput) or move_view_with_storm:
                    if 'grid' in self.gui.lines_show:
                        self.set_grid(gh_panellist); gh_changed=True
                
                products_plainproducts=[[j,self.data_attr['product'][j]] for j in panellist if j in self.data_attr['product'] and self.data_attr['product'][j] in gv.plain_products]
                productsbefore_plainproducts=[[j,self.products_before[j]] for j in panellist if self.products_before[j] in gv.plain_products]
                
                #Update height rings only if the scanangle differs by more than 0.05 degrees from the angle on which the height rings are currently
                #based, to prevent that there are too much changes (which leads to undesired flickering and slows down plotting a bit).
                scanangles_updated = self.update_heightrings_scanangles()
                if not self.firstplot_performed or (not self.postpone_plotting_gridheightrings and (radar_changed or scanangles_updated or (
                plain_products_parameters_changed or products_plainproducts != productsbefore_plainproducts))) or move_view_with_storm:
                    if 'heightrings' in self.gui.lines_show:
                        self.set_heightrings(gh_panellist); gh_changed=True
                
                if gh_changed: self.set_ghlineproperties(gh_panellist)
                if gh_changed: self.set_ghtextproperties(gh_panellist)
                
                
            for j in panellist:     
                if data_changed[j] and self.data_attr['proj'][j] == 'pol' and (self.use_interpolation or self.dsg.data_azimuth_offset[j]):
                    self.dsg.data[j] = np.concatenate(([self.dsg.data[j][-1]], self.dsg.data[j], [self.dsg.data[j][0]]))
                    #The last azimuth is prepended to the array, and the first azimuth is appended to the array to ensure that interpolation 
                    #works correctly between 360 and 0 degrees.
                
                radar_image = 'radar_polar' if not data_changed[j] or self.data_attr['proj'][j] == 'pol' else 'radar_cartesian'
                other_radar_image = 'radar_cartesian' if radar_image == 'radar_polar' else 'radar_polar'
                self.visuals[radar_image][j].set_data(self.dsg.data[j])
                self.visuals[radar_image][j].visible = True
                self.visuals[other_radar_image][j].visible = False
                
                if data_changed[j]:
                    if self.visuals[radar_image][j].cmap != self.cm1[self.data_attr['product'][j]] or\
                    list(map(int, self.visuals[radar_image][j].clim)) != self.clim_int[self.data_attr['product'][j]]:
                        # Changing the image attributes unnecessarily for every panel is relatively slow.
                        self.visuals[radar_image][j].cmap=self.cm1[self.data_attr['product'][j]]
                        self.visuals[radar_image][j].clim=self.clim_int[self.data_attr['product'][j]]
                    
                    if self.data_attr['proj'][j] == 'pol':
                        if self.data_attr['product'][j] in gv.plain_products:
                            radial_res=self.data_attr['radial_res'][j]
                            azimuthal_res=self.data_attr['azimuthal_res'][j]
                            scanangle=0.
                        else:
                            radial_res=self.data_attr['radial_res'][j]
                            azimuthal_res=self.data_attr['azimuthal_res'][j]
                            scanangle=self.data_attr['scanangle'][j]
                        azimuthal_bins, radial_bins = self.dsg.data[j].shape
                        rbr_product = radial_bins*radial_res
                        abr_product = azimuthal_bins*azimuthal_res
                        
                        if self.polar_transforms_individual['scanangle'][j].scanangle!=scanangle or\
                        self.polar_transforms_individual['scale'][j].scale[0]!=rbr_product or self.polar_transforms_individual['scale'][j].scale[1]!=abr_product:                            
                            self.polar_transforms_individual['scanangle'][j].scanangle=scanangle
                            # self.ref_radial_bins and self.ref_azimuthal_bins are set in self.on_draw
                            ref_radial_bins = self.ref_radial_bins.get(j, radial_bins)
                            ref_azimuthal_bins = self.ref_azimuthal_bins.get(j, azimuthal_bins)
                            self.polar_transforms_individual['scale'][j].scale=(rbr_product/ref_radial_bins, abr_product/ref_azimuthal_bins)
                            #Division by the reference values is necessary to get the correct scaling when the number of radial bins changes.
                        
                        #The -azimuthal_res is for the added radial for interpolation
                        delta_a = -azimuthal_res if azimuthal_bins > self.data_attr['azimuthal_bins'][j] else 0
                        translate = np.array([self.dsg.data_radius_offset[j], self.dsg.data_azimuth_offset[j]+delta_a])
                        if (self.polar_transforms_individual['scale'][j].translate[:2] != translate).any():
                            self.polar_transforms_individual['scale'][j].translate = translate
                    else:
                        res = self.data_attr['res'][j]
                        xy_bins = self.data_attr['xy_bins'][j]
                        self.cartesian_transforms_individual['scale_translate'][j].scale = (res, res)
                        self.cartesian_transforms_individual['scale_translate'][j].translate = (-xy_bins*res/2, -xy_bins*res/2)                    
                    
                    #self.products_before and self.scans_before contain information about the products and scans that were used during the last.
                    #plotting session.
                    self.products_before[j]=self.data_attr['product'][j]
                    self.scans_before[j]=self.crd.scans[j]                
                                                
                elif self.data_empty[j] and not j in self.ref_radial_bins:
                    #Ensure that the reference number of radial bins is correct when the first image that is viewed after starting 
                    #the program is 'empty' (or in fact a 362*1 invisible array).
                    self.ref_azimuthal_bins[j], self.ref_radial_bins[j] = self.dsg.data[j].shape
                       
            self.set_interpolation()
            self.set_titles()
            update_cbars_products = self.set_cbars(set_cmaps=False) #The colormaps have already been set.
            
            plot_vwp = self.gui.show_vwp and (self.vwp.data_name != self.vwp.get_current_data_name() or 
                                              self.gui.switch_to_case_running)
            # Always plot VWP when switching to a case, because if the datetime doesn't change, then the storm motion might still do so
            if plot_vwp:
                self.vwp.set_newdata()
                
            # t = pytime.time()                
            # self.update_data_readout()
            # print(pytime.time()-t,'readout')
            
            if plot_vwp: self.set_draw_action('plotting_vwp')
            elif len(update_cbars_products)>0: self.set_draw_action('updating_cbars')
            elif not self.changing_panels: self.set_draw_action('plotting')
            self.update()
            
            if (not self.firstplot_performed or radar_changed or dataset_changed or self.crd.changing_subdataset or change_datetime) and\
            any(data_changed.values()):
                self.update_current_variables()
                                                                    
            if any(data_changed.values()):
                self.firstplot_performed = True
                                
            if self.gui.current_case_shown():
                self.gui.set_textbar()  

            # profiler.disable()
            # import pstats
            # stats = pstats.Stats(profiler).sort_stats('cumtime')
            # stats.print_stats(50)        
        if not set_data:
            return retrieved_attrs
    
    
    def update_current_variables(self):
        radar_dataset = self.dsg.get_radar_dataset()
        self.crd.current_variables['date']=self.crd.date
        self.crd.current_variables['time']=self.crd.time
        self.crd.current_variables['datetime']=self.crd.date+self.crd.time
        self.crd.current_variables['dataset']=self.crd.dataset
        self.crd.current_variables['radar']=self.crd.radar
        self.crd.current_variables['radardir_index']=self.gui.radardata_dirs_indices[radar_dataset]
        self.crd.current_variables['product_version']=self.gui.radardata_product_versions[radar_dataset]        
        self.crd.current_variables['scannumbers_forduplicates']=self.dsg.scannumbers_forduplicates.copy()
    
    def update_before_variables(self,radar_changed,radardir_index_changed,dataset_changed):
        """It is important that self.crd.before_variables['radar'] etc are updated at the same moment, to ensure a consistent set of previous variables.
        """ 
        if len(self.crd.current_variables):
            if any([radar_changed,radardir_index_changed,dataset_changed]):
                self.crd.rd_before_variables={j:self.crd.current_variables[j] for j in ('radar','dataset','radardir_index','product_version')}
                self.crd.rd_before_variables['scannumbers_forduplicates']=self.crd.current_variables['scannumbers_forduplicates'].copy()
            self.crd.before_variables=copy.deepcopy(self.crd.current_variables)
            
            
    def change_map_center(self,old_radar,new_radar):
        center_screencoords=self.panel_centers[0]
        center_xycoord_before=self.screencoord_to_xy(center_screencoords)
        center_latlon=np.array(ft.aeqd(gv.radarcoords[old_radar],center_xycoord_before,inverse=True))
        center_xycoord_after=ft.aeqd(gv.radarcoords[new_radar],center_latlon) 
        for j in self.panellist:
            self.panels_sttransforms[j].move(np.array([1,-1])*(center_xycoord_before-center_xycoord_after)*np.array(self.panels_sttransforms[j].scale[:2]))
        #The view is changed in such a way that the center of the view is not displaced under the change of projection.
    
        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'].scale = (scale_x, scale_y)
        self.map_transforms['st'].translate = (t_x, t_y)
        self.map_transforms['aeqd'].latlon_0 = gv.radarcoords[new_radar]
        self.update_combined_lineproperties(range(self.max_panels),changing_radar=True)
        
        self.radarcoords_xy = np.array(ft.aeqd(gv.radarcoords[self.crd.radar], np.array([gv.radarcoords[j] for j in gv.radars_all])))
        if self.gui.sm_marker_present or len(self.gui.pos_markers_positions): 
            self.set_sm_pos_markers() 
        
    def get_map_sttransform_parameters(self):
        if self.map_initial_bounds is None: 
            self.map_initial_bounds = copy.deepcopy(self.map_bounds)
            
            map_scalefac=(self.map_bounds['lat'][1]-self.map_bounds['lat'][0])/self.map_data.shape[0]
            self.map_initial_scale = map_scalefac*np.array([1,-1])
        
        mapbound_ratio_lat = (self.map_initial_bounds['lat'][1]-self.map_initial_bounds['lat'][0])/(self.map_bounds['lat'][1]-self.map_bounds['lat'][0])
        mapbound_ratio_lon = (self.map_initial_bounds['lon'][1]-self.map_initial_bounds['lon'][0])/(self.map_bounds['lon'][1]-self.map_bounds['lon'][0])
        scale_x, scale_y = self.map_initial_scale/np.array([mapbound_ratio_lon, mapbound_ratio_lat])
        t_x, t_y=self.map_bounds['lon'][0], self.map_bounds['lat'][1]
        return (scale_x, scale_y), (t_x, t_y)

        
    def get_latlon_bounds(self, xy_bounds=None):
        if xy_bounds is None:
            self.get_corners()
            x_min, x_max = min(self.corners[j][0,0] for j in self.panellist), max(self.corners[j][-1,0] for j in self.panellist)
            y_min, y_max = min(self.corners[j][1,1] for j in self.panellist), max(self.corners[j][0,1] for j in self.panellist)            
        else:
            x_min, x_max, y_min, y_max = xy_bounds
            
        x_range = np.linspace(x_min, x_max, 10)
        y_range = np.linspace(y_min, y_max, 10)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        xy = np.transpose([x_grid.flatten(), y_grid.flatten()])
        
        radar_latlon = gv.radarcoords[self.crd.radar]
        latlon = ft.aeqd(radar_latlon, xy, inverse=True)
        lat_min, lat_max = latlon[:,0].min(), latlon[:,0].max()
        lon_min, lon_max = latlon[:,1].min(), latlon[:,1].max()
        return lat_min, lat_max, lon_min, lon_max
        
    def update_map_tiles(self, xy_bounds = None, separate_thread = True, draw_map = False):
        if not self.gui.mapvisibility and not self.starting: return
        
        if self.mt.isRunning():
            self.mt.quit()
            
        lat_min, lat_max, lon_min, lon_max = self.get_latlon_bounds(xy_bounds)
        try:
            update_map = (self.map_panels_scale_before != self.panels_sttransforms[0].scale).any() or (
                         lat_min < self.map_bounds['lat'][0] or lat_max > self.map_bounds['lat'][1] or
                         lon_min < self.map_bounds['lon'][0] or lon_max > self.map_bounds['lon'][1])
        except Exception:
            update_map = True
            
        if update_map:
            self.mt.set_radar_and_mapbounds(self.crd.radar, lat_min, lat_max, lon_min, lon_max)
            if separate_thread: 
                self.mt.start()
            else:
                self.map_data, self.map_bounds = self.mt.run_outside_thread()
                if draw_map:
                    self.draw_map_tiles()
        
    def set_timer_update_map_tiles(self):
        if not self.gui.mapvisibility: return
        if self.timer_update_map_tiles_running:
            self.timer_update_map_tiles.stop()
        #Time is in ms for the timer, not s
        self.timer_update_map_tiles_running = True
        self.timer_update_map_tiles=QTimer()
        self.timer_update_map_tiles.setSingleShot(True)
        self.timer_update_map_tiles.timeout.connect(self.update_map_tiles)
        self.timer_update_map_tiles.start(int(self.gui.maptiles_update_time*1000)) #s to ms
        
    def draw_map_tiles(self, tiles_map = None, tiles_bounds = None):
        if not tiles_map is None:
            self.map_data = tiles_map; self.map_bounds = tiles_bounds
            self.map_panels_scale_before = self.panels_sttransforms[0].scale
        self.visuals['map'][0].set_data(self.map_data)

        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'].scale = (scale_x, scale_y)
        self.map_transforms['st'].translate = (t_x, t_y)
        self.update()
        
        
    def set_titles(self):
        relwidth = self.wsize['main'][0] / self.gui.screen_size()[0]
        if any(j not in self.data_attr['scandatetime'] or j not in self.data_isold for j in self.panellist):
            print(list(self.data_attr['scandatetime']), self.data_isold)
        scandates = [self.data_attr['scandatetime'][j][:8] for j in self.panellist if not self.data_isold[j]]
        # During the switch to another day, it could happen that some scans are for the previous day, and others for the next. In that case show the
        # date that is most common in the panels.
        date = max(set(scandates), key=scandates.count) if scandates else self.crd.date
        title_top, title_bottom, paneltitles=bg.get_titles(relwidth,self.gui.fontsizes_main['titles'],self.crd.radar,self.panels,self.panellist,self.panelnumber_to_plotnumber,self.plotnumber_to_panelnumber,self.data_empty,self.data_isold,self.data_attr['product'],date,self.data_attr['scantime'],self.data_attr['scanangle'],self.crd.using_unfilteredproduct,self.crd.using_verticalpolarization,self.crd.apply_dealiasing,self.productunits,self.gui.stormmotion,self.gui.PP_parameter_values,self.gui.PP_parameters_panels,self.gui.show_vwp)
        
        titles_text_top=[title_top]+[paneltitles[j] for j in paneltitles if j<5 or self.panels==2]
        titles_text_bottom=[title_bottom]+[paneltitles[j] for j in paneltitles if j>=5 and self.panels!=2]
        
        dy_bottom = dy_top = self.scale_pixelsize(2)
        dy_bottom += 0.5*self.scale_pixelsize(self.panel_borders_width) # For the bottom panel borders
        titles_top_ypos = self.wpos['top'][0,1]+dy_top; titles_bottom_ypos = self.wpos['bottom'][0,1]+dy_bottom
        xleft=self.wpos['top'][0,0]; xright=self.wpos['top'][-1,0]
        xdim_1p=(xright-xleft)/self.ncolumns
        
        textpos_title_top=np.array([[(xleft+xright)/2,titles_top_ypos]])
        textpos_title_bottom=np.array([[(xleft+xright)/2,titles_bottom_ypos]])

        if self.panels!=2:
            textpos_paneltitles_top=np.array([[xleft+(j+0.5)*xdim_1p,titles_top_ypos] for j in paneltitles if j<5])
            textpos_paneltitles_bottom=np.array([[xleft+(j-4.5)*xdim_1p,titles_bottom_ypos] for j in paneltitles if j>=5])
        else:
            textpos_paneltitles_top=np.array([[xleft+(j+0.5-(4 if j==5 else 0))*xdim_1p,titles_top_ypos] for j in paneltitles])    
            textpos_paneltitles_bottom=[]
            
        titles_textpos_top=textpos_title_top if len(textpos_paneltitles_top)==0 else np.concatenate((textpos_title_top,textpos_paneltitles_top),axis=0)
        titles_textpos_bottom=textpos_title_bottom if len(textpos_paneltitles_bottom)==0 else np.concatenate((textpos_title_bottom,textpos_paneltitles_bottom),axis=0)

        self.visuals['titles'].text=titles_text_top+titles_text_bottom
        if len(titles_textpos_bottom)==0:
            self.visuals['titles'].pos=titles_textpos_top
        else:         
            self.visuals['titles'].pos=np.concatenate([titles_textpos_top,titles_textpos_bottom])



    def change_panels(self,new_panels):
        if new_panels==self.panels: return
        self.panels=new_panels
        self.panel=0
        self.changing_panels=True
                
        self.determine_panellist_nrows_ncolumns()
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()
             
                                
        products_panels=[self.crd.products[j] for j in self.panellist]
        for product in gv.plain_products_with_parameters:
            if product in products_panels:
                self.gui.change_PP_parameters_panels(product)

                    
        if self.firstplot_performed and self.crd.plot_mode in ('Row','Column') and not self.gui.setting_saved_choice: 
            #If self.gui.setting_saved_choice=True, then the products and scans should not change here.
            #When plot_mode in ('Row','Column') the products and scans are changed in such a way that their values are in agreement with what
            #is desired in the row or column mode.
            products_panellist=[self.crd.products[j] for j in self.panellist]
            n_products=len(set(products_panellist))
            if n_products>1:
                #Only when displaying at least 2 different products, because when displaying only one product it is likely not desired to show the
                #same scan for all panels in the same row/column.
                for j in tuple(range(0,self.panels//2))+((5,) if self.panels>=5 else ()):
                    #The leftmost and upper panels are taken as reference panels, where the number of reference panels depends on the number of panels.
                    if self.crd.plot_mode=='Row': self.crd.row_mode(panel=j)
                    else: self.crd.column_mode(panel=j)
                self.crd.update_selected_scanangles()
        elif self.firstplot_performed:
            notplain_products_panels=[j for j in products_panels if not j in gv.plain_products]
            if len(set(notplain_products_panels))==len(notplain_products_panels) and not self.gui.setting_saved_choice: 
                #When all not-plain products are unique, then it is assumed that it is desired to view those products for the same scan, 
                #which is set to be the scan of the first panel.
                for j in [i for i in self.panellist if self.crd.products[i] in gv.products_with_tilts]:                        
                    #It is possible that there are 2 scans with the same scanangle, which are in the case of changing panels both allowed in
                    #the same row. If no exception would be made for a change in panels, then you would always have either in both panels
                    #the first or in both panels the second of those 2 scans with the same scanangle, while it can be desired to show them
                    #both (because 1 has a larger range and the other a higher Nyquist velocity for the new KNMI radars).
                    scanangle_j=self.data_attr['scanangle'].get(j, 999)
                    scanangle_0=self.data_attr['scanangle'].get(0, scanangle_j)
                    if scanangle_j != scanangle_0:  
                        self.crd.scans[j]=self.crd.scans[0]
                self.crd.update_selected_scanangles()                     

        if not self.firstplot_performed:
            self.crd.process_datetimeinput()
        else:
            #Panels which show data for a previous radar are updated here
            self.set_newdata(self.panellist,source_function=self.change_panels)
             
            if not self.postpone_plotting_gridheightrings:
                if 'grid' in self.gui.lines_show: self.set_grid()
                if 'heightrings' in self.gui.lines_show: self.set_heightrings() 
                if any([j in self.gui.lines_show for j in ('grid','heightrings')]):
                    self.set_ghlineproperties(self.panellist)
            if len(self.gui.ghtext_show)>0 and not self.postpone_plotting_gridheightrings: self.set_ghtextproperties(self.panellist)
        
        self.set_timer_update_map_tiles() #This is done here instead of in the function self.on_draw, because self.draw_action is not always set to
        #'changing_panels' here.
        
        self.set_draw_action('changing_panels')
        self.update()
        
        self.changing_panels = False
        
        

    def set_cmaps(self,products):
        tickslim_modified,ticks_steps,excluded_values_for_ticks,included_values_for_ticks,changed_colortables,self.productunits=bg.set_colortables(self.gui.colortables_dirs_filenames,products,self.productunits) 
        
        for product in products: 
            if product in ('v','s','w'):
                self.scale_factors[product]=gv.scale_factors_velocities[self.productunits[product]]
                
            if not product in changed_colortables and (self.gui.cmaps_minvalues[product]!=self.cmaps_minvalues_before[product] or 
            self.gui.cmaps_maxvalues[product]!=self.cmaps_maxvalues_before[product]):
                changed_colortables.append(product)

            if product in changed_colortables:
                product_cbar=product
                cbar_colors = np.flipud(genfromtxt(opa(os.path.join(gv.programdir+'/Generated_files','colortable_'+product_cbar+'_added.csv')), delimiter=','))
                scale = self.scale_factors[product]
                self.data_values_colors[product]=cbar_colors[:,0]/scale
                step=ticks_steps[product_cbar]
                if step!='-': step /= self.scale_factors[product]
                tickslim = tickslim_modified[product]
                for j in tickslim:
                    if not tickslim[j] is None:
                        tickslim[j] /= self.scale_factors[product]
                
                
                remove_indices=[]
                prepend_productvalue=[]; append_productvalue=[]
                prepend_colors=np.array([]); append_colors=np.array([])
                
                for j in range(0,len(self.data_values_colors[product])):
                    if self.gui.cmaps_minvalues[product]!='' and self.gui.cmaps_minvalues[product]>=self.data_values_colors[product][0] and (
                    self.gui.cmaps_minvalues[product]<self.data_values_colors[product][-1] and self.data_values_colors[product][j]<self.gui.cmaps_minvalues[product]):
                        if self.data_values_colors[product][j+1]>self.gui.cmaps_minvalues[product]:
                            prepend_productvalue=[self.gui.cmaps_minvalues[product]]
                            color1=cbar_colors[j][1:4]
                            color2=cbar_colors[j][4:] if not cbar_colors[j][4]==-1. else cbar_colors[j+1][1:4]
                            position=(self.gui.cmaps_minvalues[product]-self.data_values_colors[product][j])/(self.data_values_colors[product][j+1]-self.data_values_colors[product][j])
                            prepend_colors=np.append(ft.interpolate_2colors(color1,color2,position),color2)
                        remove_indices.append(j) 
                    if self.gui.cmaps_maxvalues[product]!='' and self.gui.cmaps_maxvalues[product]>self.data_values_colors[product][0] and (
                    self.gui.cmaps_maxvalues[product]<=self.data_values_colors[product][-1] and self.data_values_colors[product][j]>=self.gui.cmaps_maxvalues[product]):
                        if self.data_values_colors[product][j-1]<self.gui.cmaps_maxvalues[product]:
                            append_productvalue=[self.gui.cmaps_maxvalues[product]]
                            color1=cbar_colors[j-1][1:4]
                            color2=cbar_colors[j-1][4:] if not cbar_colors[j-1][4]==-1. else cbar_colors[j][1:4]
                            position=(self.gui.cmaps_maxvalues[product]-self.data_values_colors[product][j-1])/(self.data_values_colors[product][j]-self.data_values_colors[product][j-1])
                            append_colors=np.append(ft.interpolate_2colors(color1,color2,position),color2)
                        remove_indices.append(j) 
                        
                self.data_values_colors[product]=np.delete(self.data_values_colors[product],remove_indices)
                self.data_values_colors[product]=np.array(prepend_productvalue+list(self.data_values_colors[product])+append_productvalue)
                if product=='r': self.data_values_colors[product]=np.log10(self.data_values_colors[product])
                
                
                """For an explanation of the process of converting floating point data values to unsigned integers, see nlr_globalvars.py.
                """
                c_lim=gv.cmaps_maxrange[product]
                cm_lim=gv.cmaps_maxrange_masked[product]
                pm_lim=gv.products_maxrange_masked[product]                    
                n_bits=gv.products_data_nbits[product]

                self.data_values_colors[product][self.data_values_colors[product]<c_lim[0]]=c_lim[0]
                self.data_values_colors[product][self.data_values_colors[product]>c_lim[1]]=c_lim[1]
                
                self.masked_values[product]=pm_lim[0]    
                self.masked_values_int[product]=0
                self.clim_int[product]=list(ft.convert_float_to_uint(np.array(cm_lim),n_bits,pm_lim))
                self.data_values_colors_int[product]=ft.convert_float_to_uint(self.data_values_colors[product],n_bits,pm_lim)
                
                         
                cbar_colors=np.delete(cbar_colors,remove_indices,axis=0)
                if len(prepend_productvalue)==1:
                    cbar_colors=np.concatenate([[np.append([prepend_productvalue],prepend_colors)],cbar_colors],axis=0)
                if len(append_productvalue)==1:
                    cbar_colors=np.concatenate([cbar_colors,[np.append([append_productvalue],append_colors)]],axis=0)
                        
                color_list1=cbar_colors[:,1:4]
                color_list2=cbar_colors[:,4:]
                color_list=np.zeros(8)
                color_data_values=[self.masked_values[product],c_lim[0]]
                
                if not product in gv.products_possibly_exclude_lowest_values:
                    #Ensure that product values below the minimum value in self.data_values_color[product], but above the minimum allowed value (given by c_lim[0]),
                    #get the correct color, which is the color for the minimum product value in self.data_values_colors[product]
                    
                    #Not when product in gv.products_possibly_exclude_lowest_values, because here it is desired that values below the minimum value in 
                    #self.data_values_color[product] are filtered away.
                    color_list=np.append(color_list,np.append(color_list1[0]/255.,1))
                    color_data_values.append(c_lim[0])
                    
                    cmap2_starti=3 #Is used below in determining the array with control points
                else:
                    cmap2_starti=2
                
                for i in range(0,len(color_list2)):
                    color_list=np.append(color_list,np.append(color_list1[i]/255.,1))
                    color_data_values.append(self.data_values_colors[product][i])
                    if i<len(color_list2)-1 and color_list2[i][0]!=-1.:
                        color_list=np.append(color_list,np.append(color_list2[i]/255.,1.))
                        color_data_values.append(self.data_values_colors[product][i+1])
                     
                #Ensure that product values above the maximum value in self.data_values_color[product], but below the maximum allowed value (given by c_lim[1]),
                #get the correct color, which is the color for the maximum product value in self.data_values_colors[product]
                color_list=np.append(color_list,color_list[-4:])
                color_data_values.append(c_lim[1])
                                            
                color_list=np.reshape(color_list,(int(len(color_list)/4),4))
                color_data_values=np.array(color_data_values)
            
                
                cmap1_range=cm_lim[1]-cm_lim[0]
                controls1=(color_data_values-cm_lim[0])/cmap1_range
                if gv.productunits_default[product] == 'dBZ':
                    # Ensure that values exactly at color boundary get assigned the 'next' color (i.e. for the higher value)
                    controls1[1:-1] -= 1e-6
                
                self.tick_map[product] = {}
                if not tickslim['start'] is None and\
                (self.data_values_colors[product][0] < tickslim['start'] < self.data_values_colors[product][1]):
                    self.tick_map[product][tickslim['start']*scale] = self.data_values_colors[product][0]*scale
                    self.data_values_colors[product][0] = tickslim['start']
                if not tickslim['end'] is None and\
                (self.data_values_colors[product][-1] > tickslim['end'] > self.data_values_colors[product][-2]):
                    self.tick_map[product][tickslim['end']*scale] = self.data_values_colors[product][-1]*scale
                    self.data_values_colors[product][-1] = tickslim['end']
                d_lim=[self.data_values_colors[product][0],self.data_values_colors[product][-1]]
                cmap2_range=d_lim[1]-d_lim[0]
                controls2=(color_data_values[cmap2_starti:-1]-d_lim[0])/cmap2_range
                controls2[controls2 < 0.] = 0.
                controls2[controls2 > 1.] = 1.

                #self.cm1 contains the color map that is used for the image visual (with a blank space for missing data), while 
                #self.cm2 contains the color map that is used for the color bar (without a blank space).

                self.cm1[product]=color.Colormap(color_list, controls=controls1, interpolation='linear')
                self.cm2[product]=color.Colormap(color_list[cmap2_starti:-1], controls=controls2, interpolation='linear')
                
                if step=='-':
                    self.data_values_ticks[product]=self.data_values_colors[product].copy()
                else:
                    #The step assigned in the color table is used as the step between subsequent ticks.
                    start=np.floor(self.data_values_colors[product][0]/step)*step
                    stop=np.ceil(self.data_values_colors[product][-1]/step)*step
                    dvt=np.linspace(start,stop,int(np.round((stop-start)/step+1)))
                    #The addition/subtraction of 1e-6 is to prevent that ticks are removed at the boundaries of the cbar range
                    remove_indices=np.nonzero((dvt+1e-6<self.data_values_colors[product][0]) | (dvt-1e-6>self.data_values_colors[product][-1]))
                    self.data_values_ticks[product]=np.delete(dvt,remove_indices)
                
                # Obtain the values that correspond to the chosen unit.
                self.data_values_ticks[product]*=self.scale_factors[product]
                self.data_values_colors[product]*=self.scale_factors[product]
                
                #Data values that are included for ticks are added here.
                for j in included_values_for_ticks[product_cbar]:
                    if not j in self.data_values_ticks[product] and not (j < self.data_values_colors[product][0] or j > self.data_values_colors[product][-1]):
                        self.data_values_ticks[product]=np.append(self.data_values_ticks[product],j)
                self.data_values_ticks[product]=np.sort(self.data_values_ticks[product])
                
                #Data values that are excluded for ticks are removed here.
                self.data_values_ticks[product]=np.array([j for j in self.data_values_ticks[product] if j not in excluded_values_for_ticks[product_cbar]])
                # Make sure that no repeated values occur. These could otherwise for example occur when the color table extends far
                # beyond the supported product value range, in which case all values beyond the limit are changed to the limit value
                self.data_values_ticks[product] = np.unique(self.data_values_ticks[product])
                
                self.cmap_lastmodification_time[product]=pytime.time()

        self.cmaps_minvalues_before=self.gui.cmaps_minvalues.copy(); self.cmaps_maxvalues_before=self.gui.cmaps_maxvalues.copy()
        return changed_colortables

    def set_individual_cbar(self,product,cbar_posnumber):
        j=cbar_posnumber
        j_side=int(np.mod(j,4))
        p='left' if j<4 else 'right'
        s=-1 if p=='left' else 1
        n_cbars_side=len([j for j in self.cbars_pos.values() if (j < 4 if p=='left' else j>3)])
        dv=self.data_values_colors[product][::-1]
        dvt=self.data_values_ticks[product][::-1]
        i=0 if s==-1 else -1 #Index of relevant corner used below
        
        cbar_blocksize=np.array([0.28*self.wsize[p][0],self.wsize[p][1]/n_cbars_side])
        cbar_size=cbar_blocksize-np.array([0,self.wsize['top'][1]+self.wsize['bottom'][1]+1]) #The +1 appears to provide better results
        if cbar_size[1]<cbar_size[0]: cbar_size=np.array([1,1]) #The major axis is not allowed to be smaller than the minor axis
        if cbar_size[0] == 0: cbar_size[0] = 1 #Ensure that the width is at least 1 pixel
        dx = 1 if p == 'left' else 0 #An extra pixel of whitespace appears to be needed on the left side in order to get the same white margin between the main widget
        # and the colorbars on both sides
        cbar_pos=np.array([self.wpos[p][s,0]+s*self.scale_pixelsize(3+dx)+0.5*s*cbar_blocksize[0],(j_side+0.5)*cbar_blocksize[1]])
        
        cbar_corners=cbar_pos+0.5*cbar_size*np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        
        cbars_ticks_xpos=cbar_corners[i,0]+0.5*s*np.abs(self.wbounds[p][0,i]-cbar_corners[i,0])
        cbars_ticks_relpos=np.array([0,cbar_size[1]])*np.reshape((dvt-dv[0])/(dv[-1]-dv[0]),(len(dvt),1))
        a = -self.scale_pixelsize(2)*self.visuals['cbars_ticks'].font_size/7.5
        cbars_ticks_relpos[0, 1] -= a; cbars_ticks_relpos[-1, 1] += a
        
        ctr=cbars_ticks_relpos
        cbars_reflines_relpos=ctr[(np.abs(dvt-dv[0])>1e-3) & (np.abs(dvt-dv[-1])>1e-3)] #Don't put reference lines next to ticks that are located near the upper
        #or lower edge of the cbar, because that is not needed, and looks ugly.
        
        self.visuals['cbar'+str(j)].cmap=self.cm2[product]
        self.visuals['cbar'+str(j)].pos=cbar_pos
        self.visuals['cbar'+str(j)].size=cbar_size[::-1] #Size should have format (major_axis, minor_axis)
        self.visuals['cbar'+str(j)].border_width = self.scale_pixelsize(1) # Setting border_width is important for when screen properties changes
                
        dvt = [k if not k in self.tick_map[product] else self.tick_map[product][k] for k in dvt]
        self.cbars_ticks[j]=[str(ft.rifdot0(ft.rndec(k,3))) for k in (dvt if product!='r' else np.power(10,dvt))]
        self.cbars_ticks_pos[j]=np.array([cbars_ticks_xpos,cbar_corners[i,1]])+cbars_ticks_relpos
        
        self.cbars_reflines_pos[j]=np.zeros((4*len(cbars_reflines_relpos),2))
        self.cbars_reflines_pos[j][::4]=cbar_corners[0]+cbars_reflines_relpos
        self.cbars_reflines_pos[j][1::4]=cbar_corners[0]+cbars_reflines_relpos+0.25*np.array([cbar_size[0],0])
        self.cbars_reflines_pos[j][2::4]=cbar_corners[-1]+cbars_reflines_relpos
        self.cbars_reflines_pos[j][3::4]=cbar_corners[-1]+cbars_reflines_relpos-0.25*np.array([cbar_size[0],0])
        
        self.cbars_labels[j]=[str(self.productunits[product]),str(gv.productnames_cmaps[product])]
        widget_avg_xpos=np.mean(self.wbounds[p][0])
        ts_y=self.wsize['top'][1]
        self.cbars_labels_pos[j]=np.array([[widget_avg_xpos,cbar_corners[0,1]-ts_y+self.scale_pixelsize(1)],
                                           [widget_avg_xpos,cbar_corners[1,1]+self.scale_pixelsize(4.5)]])
           
    def set_cbars_ticks_and_labels(self):        
        cbars_ticks=np.concatenate(list(self.cbars_ticks.values()))
        cbars_ticks_pos=np.concatenate(list(self.cbars_ticks_pos.values()))
        
        cbars_reflines_pos=np.concatenate(list(self.cbars_reflines_pos.values()))
        cbars_reflines_connect=np.ravel(np.array([[1,0] for j in range(0,int(len(cbars_reflines_pos)/2))])).astype('bool')

        cbars_labels=np.concatenate(list(self.cbars_labels.values()))
        cbars_labels_pos=np.concatenate(list(self.cbars_labels_pos.values()))

        self.visuals['cbars_ticks'].text=cbars_ticks; self.visuals['cbars_ticks'].pos=cbars_ticks_pos
        self.visuals['cbars_reflines'].set_data(pos=cbars_reflines_pos,connect=cbars_reflines_connect,width=self.scale_pixelsize(1))
        self.visuals['cbars_labels'].text=cbars_labels; self.visuals['cbars_labels'].pos=cbars_labels_pos

    def set_cbars(self,resize=False,set_cmaps=True):   
        self.cbars_products, self.cbars_nproducts, self.cbars_pos, update_cbars_pos, update_cbars_products=bg.determine_colortables(self.data_attr['product'],self.panels,[j for j in self.panellist if not self.data_empty[j]])

        if set_cmaps:
            self.changed_colortables = self.set_cmaps(self.cbars_products)
        for j in range(0,len(self.cbars_products)):
            if self.cbars_products[j] in self.changed_colortables and update_cbars_products.count(self.cbars_products[j])<self.cbars_products.count(self.cbars_products[j]):
                update_cbars_pos.append(self.cbars_pos[j])
                update_cbars_products.append(self.cbars_products[j])
        
        if resize or self.cbars_products!=self.cbars_products_before:
            self.cbars_ticks={}; self.cbars_reflines={}; self.cbars_labels={}
            self.cbars_ticks_pos={}; self.cbars_reflines_pos={}; self.cbars_labels_pos={}
            
            update_cbars_products=self.cbars_products; update_cbars_pos=[self.cbars_pos[j] for j in sorted(list(self.cbars_pos.keys()))] #Sorting is important
            
            for j in range(self.max_panels):
                self.visuals['cbar'+str(j)].visible=j in self.cbars_pos.values()
                              
        for j in range(0,len(update_cbars_products)):
            self.set_individual_cbar(update_cbars_products[j],update_cbars_pos[j])
        if len(update_cbars_products)>0:
            self.set_cbars_ticks_and_labels()
            
        self.cbars_products_before=self.cbars_products.copy()
        return update_cbars_products                            
                
    def get_corner_specs(self):
        return [str(self.panels_sttransforms[j].translate[:2])+str(self.panels_sttransforms[j].scale[:2]) for j in self.panellist]
    def get_corners(self):
        if not hasattr(self, 'corner_specs') or self.get_corner_specs() != self.corner_specs:
            # Without setting panel=j, the right edge of the panel is actually considered to be part of the next panel, causing the calculated
            # right panel corners to be the same as the left corners 
            self.corners = {j:self.screencoord_to_xy(self.panel_corners[j], panel=j) for j in self.panellist}
            self.xdim, self.ydim = self.corners[0][-1]-self.corners[0][1]
            self.corner_specs = self.get_corner_specs()
          
    def set_grid(self, panellist=None):
        panellist = self.panellist if panellist is None else panellist
        
        self.get_corners()
        physical_size_cm_main = self.physical_size_cm() - 2*self.scale_physicalsize(self.wdims)
        rel_xdim = self.physical_size[0]/self.gui.screen_size()[0] * (1-self.vwp_relxdim*self.gui.show_vwp)
        self.gridlines_vertices, self.gridlines_connect, self.gridlines_text_hor_pos, self.gridlines_text_hor, self.gridlines_text_vert_pos, self.gridlines_text_vert =\
            bg.determine_gridpos(physical_size_cm_main,rel_xdim,self.corners,self.visuals['text_hor1'][0].font_size,self.panels,panellist,self.nrows,self.ncolumns,self.gui.show_vwp)
        self.update_combined_lineproperties(panellist,changing_grid=True)
        self.update_combined_ghtextproperties(panellist,changing_grid=True)
         
    def set_heightrings(self, panellist=None):
        panellist = self.panellist if panellist is None else panellist
        panellist = [j for j in panellist if not self.data_empty[j]]
        
        self.get_parameters_heightrings()
        
        for j in panellist:
            product = self.data_attr['product'][j]
            derivedproduct_noheightrings=product in self.gui.show_heightrings_derivedproducts and (
            self.gui.show_heightrings_derivedproducts[product]==0)
            if not derivedproduct_noheightrings and len(self.vertices_circles[j])>0:
                if product in gv.plain_products_show_true_elevations:
                    # Don't show text for heights for which the plain product has been calculated, since it is already shown in the title, 
                    # and it can lead to annoyingly closely spaced text labels for low-elevation PCAPPIs.
                    h_remove = {'a':self.gui.PP_parameter_values['a'][self.gui.PP_parameters_panels[j]], 'r':gv.CAPPI_height_R}
                    i_heights_remove = [i for i,h in enumerate(self.heights[j]) if h == h_remove[product]]
                    for o in ('heights', 'radii', 'textangles'):
                        self.__dict__[o][j] = np.delete(self.__dict__[o][j], i_heights_remove)
                
                self.heights_text[j] = ft.format_nums(self.heights[j], dec=2)
                if product not in gv.plain_products_show_max_elevations:
                    self.heights_text_pos[j]=[self.radii[j][i]*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(0,len(self.radii[j]))]
                else:
                    pos_offset=20*self.ydim*self.nrows/1e3
                    if product in gv.plain_products:
                        heights_text_pos1=[(self.radii[j][i]+pos_offset)*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(len(self.radii[j])-len(self.dp.meta_PP[product]['elevations_minside']),len(self.radii[j]))]
                    heights_text_pos2=[(self.radii[j][i]-pos_offset)*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(0,len(self.radii[j]))]
                    if product in gv.plain_products and len(heights_text_pos1)>0:
                        self.heights_text_pos[j]=np.concatenate(tuple([heights_text_pos1,heights_text_pos2]))
                    else: self.heights_text_pos[j]=heights_text_pos2
            else:
                self.vertices_circles[j]=[]; self.vertices_circles_connect[j]=[]
                self.heights_text[j]=[]; self.heights_text_pos[j]=[] 
                
        self.update_combined_lineproperties(panellist,changing_heightrings=True)
        self.update_combined_ghtextproperties(panellist,changing_heightrings=True)
            
    def update_heightrings_scanangles(self):
        heightrings_updated = False
        for j in self.data_attr['scanangle']:
            if not self.crd.lrstep_beingperformed or abs(self.heightrings_scanangles[j]-self.data_attr['scanangle'][j]) > 0.05:
                self.heightrings_scanangles[j] = self.data_attr['scanangle'][j]
                heightrings_updated = True
        return heightrings_updated
        
    def get_parameters_heightrings(self,start=0):
        if not self.dsg.scanangles_all['z']:
            return
        
        self.get_corners()
        
        panellist_nonempty=[j for j in self.panellist if not self.data_empty[j]]
        panellist_no_pp=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] not in gv.plain_products]
        panellist_pp_max_elevations=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] in gv.plain_products_show_max_elevations] 
        panellist_pp_true_elevations=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] in gv.plain_products_show_true_elevations] 
        panellist_determine_heightrings = panellist_no_pp+panellist_pp_true_elevations
        
        self.heights={}; self.radii={}
        if len(panellist_determine_heightrings):
            rel_xdim = self.physical_size[0]/self.gui.screen_size()[0] * (1-self.vwp_relxdim*self.gui.show_vwp)
            self.update_heightrings_scanangles()
            angle1 = self.dsg.scanangle('z', 1, 0)
            scanangles = np.array([self.heightrings_scanangles[j] if j in panellist_no_pp else angle1 for j in panellist_determine_heightrings])
            use_previous_hrange = self.crd.lrstep_beingperformed
            self.heights, self.radii = bg.determine_heightrings(rel_xdim,self.corners,self.ncolumns,panellist_determine_heightrings,scanangles,use_previous_hrange)
        if len(panellist_pp_true_elevations):
            panel = panellist_pp_true_elevations[0]
            heights_scan1, radii_scan1 = self.heights[panel], self.radii[panel]
            new_heights, new_radii = self.dp.get_true_elevations_plainproducts(panellist_pp_true_elevations, heights_scan1, radii_scan1)
            self.heights={**self.heights,**new_heights}; self.radii={**self.radii,**new_radii}
        if len(panellist_pp_max_elevations):
            for j in panellist_pp_max_elevations:
                product = self.data_attr['product'][j]
                self.heights[j]=np.concatenate((self.dp.meta_PP[product]['elevations_minside'],self.dp.meta_PP[product]['elevations_plusside']))
                self.radii[j]=self.dp.meta_PP[product]['scans_ranges']
        
        self.textangles=bg.determine_textangles(self.corners,self.panels,panellist_nonempty,self.data_attr['product'],self.radii)
                    
        self.vertices_circles={}; self.vertices_circles_connect={}
        for i in self.panellist:
            if i in panellist_nonempty and len(self.radii[i])>0:
                for j, r in enumerate(self.radii[i]):
                    vertices_add=r*self.unitcircle_vertices
                    if j==0:
                        self.vertices_circles[i]=vertices_add
                        self.vertices_circles_connect[i]=np.ones(len(vertices_add))
                    else:
                        self.vertices_circles[i]=np.concatenate((self.vertices_circles[i],vertices_add),axis=0)
                        self.vertices_circles_connect[i]=np.append(self.vertices_circles_connect[i],np.ones(len(vertices_add)))
                    self.vertices_circles_connect[i][-1]=0
                self.vertices_circles_connect[i]=self.vertices_circles_connect[i].astype('bool')
            else: self.vertices_circles[i]=[]; self.vertices_circles_connect[i]=[]
                                    
              
    def update_combined_ghtextproperties(self,panellist,changing_heightrings=False,changing_grid=False):
        for i in panellist:
            # if i in getattr(self, 'dont_draw', []):
            #     continue
            if changing_grid:
                self.ghtext_hor_pos_combined[i]['grid']=self.gridlines_text_hor_pos[i]
                self.ghtext_hor_strings_combined[i]['grid']=self.gridlines_text_hor[i]
                self.ghtext_vert_pos_combined[i]['grid']=self.gridlines_text_vert_pos[i]
                self.ghtext_vert_strings_combined[i]['grid']=self.gridlines_text_vert[i]
            if changing_heightrings:
                self.ghtext_hor_pos_combined[i]['heightrings']=self.heights_text_pos[i]
                self.ghtext_hor_strings_combined[i]['heightrings']=self.heights_text[i]
            
    def include_texttype(self,panel,text_type,hor_vert):
        pos_combined=self.ghtext_hor_pos_combined if hor_vert=='hor' else self.ghtext_vert_pos_combined
        return text_type in self.gui.ghtext_show and not len(pos_combined[panel][text_type])==0 and not (
        text_type in ('grid','heightrings') and (self.gridheightrings_removed or self.data_empty[panel]))
    def set_ghtextproperties(self,panellist):
        self.ghtext_order = ['grid','heightrings'] #From bottom to top
        for i in self.panels_horizontal_ghtext:
            if i in panellist:
                pos = []
                try: #The multiplication by -1 for the y coordinates is because the y axis is flipped in VisPy
                    pos = np.array([1,-1])*np.concatenate([self.ghtext_hor_pos_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'hor')])
                    text = np.concatenate([self.ghtext_hor_strings_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'hor')])
                    self.visuals['text_hor1'][i].text = text
                    self.visuals['text_hor1'][i].pos = pos
                except Exception:
                    pass
                self.visuals['text_hor1'][i].visible = self.visuals['text_hor2'][i].visible = len(pos) > 0
        for i in self.panels_vertical_ghtext:
            if i in panellist:
                pos = []
                if 'grid' in self.gui.ghtext_show:
                    try:
                        pos = np.array([1,-1])*np.concatenate([self.ghtext_vert_pos_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'vert')])
                        ghtext_vert_strings = np.concatenate([self.ghtext_vert_strings_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'vert')])
                        self.visuals['text_vert1'][i].text = ghtext_vert_strings
                        self.visuals['text_vert1'][i].pos = pos
                    except Exception:
                        pass
                self.visuals['text_vert1'][i].visible = self.visuals['text_vert2'][i].visible = len(pos) > 0
                
    def update_combined_lineproperties(self,panellist,changing_radar=False,changing_colors=False,changing_grid=False,changing_heightrings=False,start=False):
        for i in panellist:
            if changing_radar:
                for j in self.shapefiles_latlon_combined:
                    self.lines_pos_combined[i][j]=self.shapefiles_latlon_combined[j]
                    if start: 
                        self.lines_connect_combined[i][j]=self.shapefiles_connect_combined[j]
                        self.lines_colors_combined[i][j]=np.ones((len(self.lines_pos_combined[i][j]),4))*self.gui.lines_colors[j]/255.
            if changing_grid:
                self.lines_pos_combined[i]['grid']=self.gridlines_vertices[i]
                self.lines_connect_combined[i]['grid']=self.gridlines_connect[i]
                self.lines_colors_combined[i]['grid']=np.ones((len(self.gridlines_vertices[i]),4))*self.gui.lines_colors['grid']/255.
            if changing_heightrings:
                self.lines_pos_combined[i]['heightrings']=self.vertices_circles[i]
                self.lines_connect_combined[i]['heightrings']=self.vertices_circles_connect[i]
                self.lines_colors_combined[i]['heightrings']=np.ones((len(self.vertices_circles[i]),4))*self.gui.lines_colors['heightrings']/255.
                        
            if changing_colors:
                for j in self.gui.lines_names:
                    self.lines_colors_combined[i][j]=np.ones((len(self.lines_pos_combined[i][j]),4))*self.gui.lines_colors[j]/255.
                    
    
    def include_linetype(self,panel,line_type):
        if line_type in self.gui.lines_show and not len(self.lines_pos_combined[panel][line_type])==0 and not (
        line_type in ('grid','heightrings') and (self.gridheightrings_removed or self.data_empty[panel])):
            return True
        else:
            return False       
    def set_maplineproperties(self,panellist):   
        lines = [j for j in self.lines_order[:3] if j in self.gui.lines_show]
        if any([j in self.gui.lines_show for j in lines]):
            for i in range(self.max_panels):
                self.visuals['map_lines'][i].visible=True
                
            lines_pos=np.concatenate([self.lines_pos_combined[i][j] for j in lines if self.include_linetype(i,j)])
            lines_connect=np.concatenate([self.lines_connect_combined[i][j] for j in lines if self.include_linetype(i,j)])
            lines_colors=np.concatenate([self.lines_colors_combined[i][j] for j in lines if self.include_linetype(i,j)])
            
            self.visuals['map_lines'][0].set_data(pos=lines_pos,connect=lines_connect,color=lines_colors,width=self.scale_pixelsize(self.gui.lines_width))   
        else: 
            for i in range(self.max_panels):
                self.visuals['map_lines'][i].visible=False
                
    def set_ghlineproperties(self,panellist):
        lines = [j for j in self.lines_order[-2:] if j in self.gui.lines_show]
        for i in panellist:
            if len(lines)>0:
                self.visuals['gh_lines'][i].visible=True
                try:
                    lines_pos=np.array([1,-1])*np.concatenate([self.lines_pos_combined[i][j] for j in lines if self.include_linetype(i,j)])                 
                    lines_connect=np.concatenate([self.lines_connect_combined[i][j] for j in lines if self.include_linetype(i,j)])
                    lines_colors=np.concatenate([self.lines_colors_combined[i][j] for j in lines if self.include_linetype(i,j)])
                    # if self.visuals['gh_lines'][i].pos is None or lines_pos.shape != self.visuals['gh_lines'][i].pos.shape or (lines_pos != self.visuals['gh_lines'][i].pos).any():
                    self.visuals['gh_lines'][i].set_data(pos=lines_pos,connect=lines_connect,color=lines_colors,width=self.scale_pixelsize(self.gui.lines_width))   
                except Exception:
                    #Occurs when no line coordinates are available, such that np.concatenate raises an exception.
                    self.visuals['gh_lines'][i].visible=False
            else: 
                self.visuals['gh_lines'][i].visible=False
                            
    def set_sm_pos_markers(self):
        # Update cartesian positions because of a possible switch of radar
        if self.gui.sm_marker_present:
            self.gui.sm_marker_position = ft.aeqd(gv.radarcoords[self.crd.radar], self.gui.sm_marker_latlon)
        self.gui.pos_markers_positions = [ft.aeqd(gv.radarcoords[self.crd.radar], j) for j in self.gui.pos_markers_latlons]
        
        pos = self.gui.pos_markers_positions + ([self.gui.sm_marker_position] if self.gui.sm_marker_present else [])
        if len(pos):
            face_color = ['white']*len(self.gui.pos_markers_positions) + (['red'] if self.gui.sm_marker_present else [])
            #-1 because y-coordinate is flipped
            self.visuals['sm_pos_markers'][0].set_data(pos=np.array(pos)*np.array([1,-1]),symbol='disc',size=self.scale_pixelsize(9),
                                                       edge_width=1,face_color=face_color,edge_color='black',scaling=False)
        for j in range(self.max_panels):
            self.visuals['sm_pos_markers'][j].visible = len(pos) > 0
                                   
    def change_mapvisibility(self,visibility):
        self.gui.mapvisibility=True if self.gui.mapvis_true.isChecked() else False
        for j in self.panellist:
            if not self.gui.mapvisibility:
                self.visuals['map'][j].visible=False
            else:                        
                self.visuals['map'][j].visible=True
        self.update_map_tiles()
        self.update()
                                   
    def change_mapcolorfilter(self):
        inputfilter=self.gui.mapcolorfilterw.text()
        color_filter= ft.rgb(inputfilter,alpha=True)
        if color_filter: #ft.rgb returns False if the input is not an RGB(A) series
            self.gui.mapcolorfilter = color_filter
            self.map_colorfilter.filter = self.gui.mapcolorfilter
            self.update()
        else:
            self.gui.mapcolorfilterw.setText(ft.list_to_string(self.gui.mapcolorfilter))
            
    def change_radarimage_visibility(self):
        for j in self.panellist:
            radar_image = 'radar_polar' if self.data_attr['proj'][j] == 'pol' else 'radar_cartesian'
            if self.visuals[radar_image][j].visible:
                self.visuals[radar_image][j].visible=False
            else:
                self.visuals[radar_image][j].visible=True      
        self.update()