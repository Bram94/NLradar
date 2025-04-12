# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np
import time as pytime
import traceback

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

import vispy
vispy.set_log_level(verbose='warning')
from vispy import visuals
from vispy import color
from vispy.visuals.transforms import STTransform
from vispy.visuals.filters import Clipper

import nlr_globalvars as gv
import nlr_functions as ft
from VWP.vvp import VVP
from VWP import sfc_obs
from VWP import vwp_functions as f



class PlottingVWP(QObject):
    def __init__(self, gui_class, pb_class):
        super(PlottingVWP, self).__init__()        
        self.gui=gui_class
        self.pb = pb_class
        self.crd=self.gui.crd
        self.dsg=self.crd.dsg
        
        n_sectors = 36
        min_sector_pairs_filled = 9
        self.vvp_min_frac_sectors_filled = min_sector_pairs_filled/n_sectors
        self.vvp = VVP(gui_class = self.gui, range_limits = self.gui.vvp_range_limits, height_limits = [0.1, 11.9], v_min = self.gui.vvp_vmin_mps,
                       n_sectors = n_sectors, min_sector_pairs_filled = min_sector_pairs_filled)
        self.sfcobs_classes = {'KNMI': sfc_obs.SfcObsKNMI(gui_class=self.gui), 'DWD': sfc_obs.SfcObsDWD(gui_class=self.gui)}
                              #'KMI': sfc_obs.SfcObsKMI(gui_class=self.gui), 'skeyes': sfc_obs.SfcObsKMI(gui_class=self.gui)}
        for source in (j for j in gv.data_sources_all if not j in self.sfcobs_classes):
            self.sfcobs_classes[source] = sfc_obs.SfcObsMETAR(gui_class=self.gui)

        
        self.dr_circles_unit = {'m/s': 5, 'kts': 10, 'km/h': 20}
        self.circles = {}
        self.base_sm = None
        self.firstplot_performed = False      
        self.updating_vwp = False #Is used in the dealias_velocity functions in nlr_importdata.py
        self.display_manual_sfcobs = False # Is updated in nlr_vwp.py
        self.display_manual_axlim = False # Is updated in nlr_vwp.py        
        self.data_name = None
        self.data_radar = None
        self.data_datetime = None
        
        self.xmin, self.xmax = -10, 10
        self.ymin, self.ymax = -10, 10
        self.plotrange = 20
        
        self.visuals = {}
        self.visuals_order = ['title','hodo_axes+circles','hodo_sigmacircles','hodo_line','hodo_points','hodo_axtext','hodo_hmarkers','hodo_htext','hodo_sm','hodo_line_legendtext','cbar_streamwise_vorticity_sign','cbar_filled_sectors','general_text_anchorx=left', 'general_text_anchorx=right', 'general_text_anchorx=center', 'legend_markers', 'legend_labels']
        self.visuals_hodo = ['hodo_axes+circles','hodo_sigmacircles','hodo_line','hodo_line_legendtext','hodo_points','hodo_axtext','hodo_hmarkers','hodo_htext','hodo_sm']
        self.visuals_bottom = ['cbar_streamwise_vorticity_sign','cbar_filled_sectors','general_text_anchorx=left','general_text_anchorx=right', 'general_text_anchorx=center', 'legend_markers', 'legend_labels']
        
        self.font_sizes = {'title':"self.gui.fontsizes_main['titles']", 'hodo_line_legendtext':'8.5', 'hodo_axtext':'8', 'hodo_htext':'6', 
                           'general_text_anchorx=left':'8.5', 'general_text_anchorx=right':'8.5', 'general_text_anchorx=center':'8.5',
                           'legend_labels':'9'}
        
        self.visuals['title'] = visuals.TextVisual(bold=True, font_size=eval(self.font_sizes['title']), anchor_y = 'top')
        self.visuals['hodo_axes+circles'] = visuals.LineVisual(pos = None, color = 'black', width = self.pb.scale_pixelsize(1))
        self.visuals['hodo_axes+circles'].antialias = True
        self.visuals['hodo_sigmacircles'] = visuals.MarkersVisual()
        
        self.visuals['hodo_line'] = []
        self.hodo_line_widths = np.append(list(range(1, 11)), np.delete(np.linspace(0, 4, 13), [0, 3, 6, 9, 12 ]))
        self.hodo_line_widths = np.sort(self.hodo_line_widths)[::-1]
        index = np.where(self.hodo_line_widths == 4.)[0][0]+1
        self.hodo_line_widths = np.insert(self.hodo_line_widths, index, 3.)
        for width in self.hodo_line_widths:        
            self.visuals['hodo_line'] += [visuals.LineVisual(pos=None, color=None, width=width, connect='segments', antialias=True)]
            
        self.visuals['hodo_line_legendtext'] = visuals.TextVisual(font_size = eval(self.font_sizes['hodo_line_legendtext']), color = 'black')
        self.visuals['hodo_points'] = visuals.MarkersVisual()
        self.visuals['hodo_axtext'] = visuals.TextVisual(font_size = eval(self.font_sizes['hodo_axtext']), color = np.array([0.5,0.5,0.5,1]))
        self.visuals['hodo_hmarkers'] = visuals.MarkersVisual()
        self.visuals['hodo_htext'] = visuals.TextVisual(font_size = eval(self.font_sizes['hodo_htext']), bold = True, color = 'white')
        self.visuals['hodo_sm'] = visuals.MarkersVisual()
        self.visuals['general_text_anchorx=left'] = visuals.TextVisual(font_size = eval(self.font_sizes['general_text_anchorx=left']), color = 'black', anchor_x = 'left', anchor_y = 'top')
        self.visuals['general_text_anchorx=right'] = visuals.TextVisual(font_size = eval(self.font_sizes['general_text_anchorx=right']), color = 'black', anchor_x = 'right', anchor_y = 'top')
        self.visuals['general_text_anchorx=center'] = visuals.TextVisual(font_size = eval(self.font_sizes['general_text_anchorx=center']), color = 'black', anchor_x = 'center', anchor_y = 'top')
        self.visuals['legend_markers'] = visuals.MarkersVisual()
        self.visuals['legend_labels'] = visuals.TextVisual(font_size = eval(self.font_sizes['legend_labels']), color = 'black', anchor_x = 'left', anchor_y = 'center')
        
        start_color = ft.interpolate_2colors(np.array([1,0,0]), np.array([0,0,1]), self.vvp_min_frac_sectors_filled)
        self.cm_filled_sectors = color.Colormap([start_color,[0,0,1]], controls=[0,1], interpolation='linear')
        color1, color2 = [1,1,0], [0,1,0]
        self.cm_streamwise_vorticity_sign = color.Colormap([color1,color1,color2,color2], controls=[0,0.5,0.5,1], interpolation='linear')
        cbar_height = 0.02
        self.cbarcenter_ypos = cbar_height/2+0.045
        self.cbartop_ypos, self.cbarbottom_ypos = self.cbarcenter_ypos+np.array([-1,1])*cbar_height/2
        self.visuals['cbar_streamwise_vorticity_sign'] = visuals.ColorBarVisual(pos=[0.25,self.cbarcenter_ypos],size=[0.36,cbar_height],cmap=self.cm_streamwise_vorticity_sign,orientation='top',clim=[0,1],label_color=(0,0,0,0))
        self.visuals['cbar_filled_sectors'] = visuals.ColorBarVisual(pos=[0.75,self.cbarcenter_ypos],size=[0.36,cbar_height],cmap=self.cm_filled_sectors,orientation='top',clim=[0,1],label_color=(0,0,0,0))
        for j in ('cbar_streamwise_vorticity_sign', 'cbar_filled_sectors'):
            self.visuals[j].visible=False
            for i in self.visuals[j]._ticks:
                i.visible=False 
                #Set the visibility of the TextVisuals for the ticks and the label to False, to prevent that they are drawn. This is because I don't use them.
            self.visuals[j]._label.visible=False

        self.hodo_clipper = Clipper()        
        self.hodo_sttransform = STTransform()
        self.bottom_sttransform = STTransform()
        for j in self.visuals:
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    if j in self.visuals_hodo:
                        visual.attach(self.hodo_clipper, view = visual)
                        visual.transform = self.hodo_sttransform
                    elif not j in self.visuals_notransform:
                        visual.transform = self.bottom_sttransform
            else:
                if j in self.visuals_hodo:
                    self.visuals[j].attach(self.hodo_clipper, view = self.visuals[j])
                    self.visuals[j].transform = self.hodo_sttransform
                elif j in self.visuals_bottom:
                    self.visuals[j].transform = self.bottom_sttransform
        
        for j in self.visuals:
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    visual.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            else:
                self.visuals[j].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                        
        
        
    def dy_center_text(self, font_size):
        #Vertical centering of text doesn't produce the desired results. This function calculates and returns a correction term
        return -0.00325*font_size/self.pb.scale_pointsize(8)*self.plotrange
        
                
    def on_resize(self, event=None):
        # Update all visuals in order to use the new canvas dimensions and/or resolution.
        self.hodo_size_px = self.pb.vwp_relxdim*self.pb.size[0]
        self.set_bottom_sttransform()
        self.set_newdata()
        
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.pb.physical_size[0], self.pb.physical_size[1])
        for j in self.visuals_order:
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    visual.transforms.configure(canvas=self.pb, viewport=vp)
                    if hasattr(visual, 'font_size'):
                        visual.font_size = self.pb.scale_pointsize(eval(self.font_sizes[j]))
            else:
                self.visuals[j].transforms.configure(canvas=self.pb, viewport=vp)
                if hasattr(self.visuals[j], 'font_size'):
                    self.visuals[j].font_size = self.pb.scale_pointsize(eval(self.font_sizes[j]))
                                
    def on_draw(self):
        visuals_to_plot = self.visuals_order
        for j in self.visuals_order:
            if not j in visuals_to_plot: continue
            
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    if visual.visible:
                        visual.draw()
            elif self.visuals[j].visible:
                self.visuals[j].draw()
                
    def set_bottom_sttransform(self):
        bottom_ypx = self.pb.size[1]-self.hodo_size_px-self.pb.wbounds['main'][1,0]
        title_px = self.pb.wbounds['main'][1,0]
        self.bottom_sttransform.scale = (self.pb.vwp_relxdim*self.pb.size[0], bottom_ypx)
        self.bottom_sttransform.translate = ((1-self.pb.vwp_relxdim)*self.pb.size[0], self.hodo_size_px+title_px)
                
    def get_hodo_xyrange(self):
        if self.display_manual_axlim:
            self.xmin, self.xmax, self.ymin, self.ymax = self.gui.vwp_manual_axlim        
        elif len(self.V) > 0:
            #if len(self.V) == 0 then these variables are kept the same as for the previous plot
            try:
                stormmotions = [getattr(self, j) for j in self.sms_display if not getattr(self, j) is None]
            except Exception: #happens for example when self.V has a length of 1, in which case the storm motions are not defined.
                stormmotions = []
            stormmotions = np.array(stormmotions)
            if len(stormmotions) == 0:
                stormmotions = self.V[:1] #Is done to prevent that errors occur below
            
            dr, do = self.dr_circles, self.dr_circles/2. 
            # Axes will be extended in such a way that they extend at least a distance of
            # do beyond the origin in both the negative and positive direction
            umin, umax = min(self.V[:,0].min(), stormmotions[:,0].min()), max(self.V[:,0].max(), stormmotions[:,0].max())
            vmin, vmax = min(self.V[:,1].min(), stormmotions[:,1].min()), max(self.V[:,1].max(), stormmotions[:,1].max())
            if umin < self.xmin or umax > self.xmax or vmin < self.ymin or vmax > self.ymax or\
            abs(ft.datetimediff_m(getattr(self, 'hodo_xy_datetime', '195001010000'), self.crd.date+self.crd.time)) > 90:
                self.xmin, self.xmax = min(int(np.floor(umin/do)*do)-dr, -do), max(int(np.ceil(umax/do)*do)+dr, do)
                self.ymin, self.ymax = min(int(np.floor(vmin/do)*do)-dr, -do), max(int(np.ceil(vmax/do)*do)+dr, do)
        self.hodo_xy_datetime = self.crd.date+self.crd.time
            
        xrange = self.xmax-self.xmin; yrange = self.ymax-self.ymin
        if xrange>yrange: 
            self.ymin -= (xrange-yrange)/2.; self.ymax += (xrange-yrange)/2.
        else:
            self.xmin -= (yrange-xrange)/2.; self.xmax += (yrange-xrange)/2.
        self.plotrange = self.xmax - self.xmin
                
    def set_hodo_sttransform(self):
        title_px = self.pb.wbounds['main'][1,0]
        scale = self.hodo_size_px / self.plotrange
        translate_x = self.hodo_size_px * (1-self.pb.vwp_relxdim)/self.pb.vwp_relxdim - scale * self.xmin
        translate_y = title_px + self.hodo_size_px + scale * self.ymin
        self.hodo_sttransform.scale = (scale, -scale) #Flip y axis because Vispy's y axis is upside down
        self.hodo_sttransform.translate = (translate_x, translate_y)
        
    def update_hodo_clipper(self):
        sx, sy = self.pb.size
        hodo_px, title_px = self.hodo_size_px, self.pb.wbounds['main'][1,0]
        self.hodo_clipper.bounds = tuple(np.array([sx-hodo_px, sy-hodo_px-title_px, hodo_px, hodo_px])*self.gui.screen_pixel_ratio())
                          
    def plot_title(self):
        self.visuals['title'].text = 'Radar VWP ('+self.pb.productunits['v']+')'+'      '+self.volume_starttime+'-'+self.volume_endtime+'Z'
        self.visuals['title'].pos = [(1-0.5*self.pb.vwp_relxdim)*self.pb.size[0],self.pb.wpos['top'][0,1]+1]
        
    def plot_hodo_circles(self):
        abs_values = np.abs([[self.xmin, self.xmax], [self.ymin, self.ymax]])
        min_radius = self.dr_circles
        max_radius = int(np.floor(np.linalg.norm([abs_values[0].max(), abs_values[1].max()]) / self.dr_circles) * self.dr_circles)
        
        pos = []
        connect = []
        color = []
        for r in range(min_radius, max_radius+1, self.dr_circles):
            if not r in self.circles:
                self.circles[r] = r * self.pb.unitcircle_vertices
            pos += [self.circles[r]]
            connect += [np.ones(len(pos[-1]), dtype = bool)]
            connect[-1][-1] = 0
            color += [np.tile([0.75, 0.75, 0.75, 1], (len(pos[-1]), 1))]
            
            if not self.base_sm is None:
                pos += [self.base_sm + pos[-1]]
                connect += [np.ones(len(pos[-1]), dtype = bool)]
                connect[-1][-1] = 0
                color += [np.tile([0.25, 0.25, 0.25, 0.104], (len(pos[-1]), 1))]
            
        pos += [np.array([[self.xmin, 0], [self.xmax, 0], [0, self.ymin], [0, self.ymax]])] #For the axes
        connect += [np.array([1,0,1], dtype = bool)]
        color += [np.tile(np.array([0,0,0,1]), (4,1))]
        pos, connect, color = np.concatenate(pos), np.concatenate(connect), np.concatenate(color)
        self.visuals['hodo_axes+circles'].set_data(pos = pos, connect = connect, color = color)
        
    def plot_hodo_axislabels(self):
        offset = 2
        xax_ticks = np.arange(int(np.ceil((self.xmin+offset)/self.dr_circles)*self.dr_circles), self.xmax-offset, self.dr_circles).astype(int)
        yax_ticks = np.arange(int(np.ceil((self.ymin+offset)/self.dr_circles)*self.dr_circles), self.ymax-offset, self.dr_circles).astype(int)
        xax_ticks, yax_ticks = xax_ticks[xax_ticks != 0], yax_ticks[yax_ticks != 0] #Don't plot ticks at the origin
        dxy_x, dxy_y = np.array([0, -0.025*self.plotrange]), np.array([-0.025*self.plotrange, self.dy_center_text(self.visuals['hodo_axtext'].font_size)])
        pos = np.concatenate([xax_ticks[:, np.newaxis]*np.array([1, 0])+dxy_x, yax_ticks[:, np.newaxis]*np.array([0, 1])+dxy_y])
        self.visuals['hodo_axtext'].pos = pos
        self.visuals['hodo_axtext'].text = np.abs(np.append(xax_ticks, yax_ticks)).astype(str)
                
    def artificially_increase_linewidth(self, pos, desired_width, actual_width):
        line = pos[1]-pos[0]
        normal = np.cross(np.array([0,0,1]), line)[:2]
        unit_normal = normal/np.linalg.norm(normal)
        # Choose the number of lines such that ugly boundary effects due to anti-aliasing are minimised
        n_lines = 2+int((desired_width-actual_width)//2)
        center_shift_max = (desired_width-actual_width)/2
        center_shifts = np.linspace(-center_shift_max, center_shift_max, n_lines)
        displacements = [i/self.hodo_size_px*self.plotrange for i in center_shifts]
        return np.concatenate([pos+unit_normal*i for i in displacements])
        
    def plot_hodo_line_and_legend(self):
        plot_line = len(self.V) > 1
        self.visuals['hodo_line_legendtext'].visible = plot_line
        for i in range(len(self.hodo_line_widths)):
            self.visuals['hodo_line'][i].visible = False
            
        if not plot_line:
            return
            
        legend_ypos = self.ymin+0.03*self.plotrange if self.ymin+0.1*self.plotrange < 0 else 0.03*self.plotrange
        side = np.sign(self.xmin+0.5*self.plotrange) if not -self.xmin == self.xmax else 1
        Vx_far_side = (self.V[:, 0]-self.xmin if side == -1 else self.xmax-self.V[:, 0]) < 0.4*self.plotrange
        Vy_low = self.V[:, 1] < legend_ypos+0.06*self.plotrange
        if np.count_nonzero(Vx_far_side & Vy_low) > np.count_nonzero(~Vx_far_side & Vy_low):
            side *= -1
        legend_xpos = (self.xmax if side == 1 else self.xmin) - side*0.215*self.plotrange
        legend_lines_xpos = legend_xpos+0.0375*np.array([-1, 1])*self.plotrange
        legend_lines_ypos = legend_ypos+0.0175*np.array([-1, 1])*self.plotrange
        legend_lines_pos = np.array([np.array([i*np.array([1, 1]), legend_lines_ypos]).T for i in legend_lines_xpos])
            
        
        i = np.count_nonzero(self.h_layers < 3)
        if i == 0:
            i = len(self.h_layers)
        max_vorticity = np.abs(self.vorticity[:i]).max() if i else None
        
        n1, n2, n3 = len(self.vorticity), len(self.sw_vorticity), 2
        
        vertices = np.empty((n1+n2+n3, 2, 2), dtype='float64')
        vertices[:n1, 0] = self.V[:-1]; vertices[:n1, 1] = self.V[1:]
        if n2:
            vertices[n1:-n3, 0] = self.V_int[:-1]; vertices[n1:-n3, 1] = self.V_int[1:]
        vertices[-n3:] = legend_lines_pos
        
        dwidth_dvorticity = self.pb.scale_pixelsize(600)
        widths = np.zeros(len(vertices), dtype='float64')
        widths[:n1] = dwidth_dvorticity*self.vorticity
        widths[n1:-n3] = dwidth_dvorticity*np.abs(self.sw_vorticity)
        legend_width1, legend_width2 = dwidth_dvorticity*0.01, widths.max()
        widths[-n3:] = [legend_width1, legend_width2]
        
        colors = np.tile(np.array([0.5,0.5,0.5,1]), (len(vertices), 1))
        colors[n1:-n3] = self.cm_streamwise_vorticity_sign.map(0.5+0.5*np.sign(self.sw_vorticity))
        
        """10 is the maximum width for GL lines.
        But the width can be artificially increased by drawing more than one line as is done below, which works quite well.
        In fact, from a width of 4 onward every line is plotted as a combination of multiple thinner lines with integer width. 
        This is done to reduce the number of lines visuals that is needed to display all desired lines with their varying thicknesses.
        Drawing multiple thinner lines instead of one thicker works well as long as the anti-aliasing edges don't overlap, which 
        requires at least a 1-pixel distance between the edges of different lines.
        For widths up to 4 pixels rounding is performed to a select group of widths (defined in the __init__).
        """
        line_widths = np.empty(len(widths), dtype='float64')
        edge_width = 4.
        round_widths = widths < edge_width; floor_widths = ~round_widths
        factor = self.hodo_line_widths[-2]-self.hodo_line_widths[-1]
        line_widths[round_widths] = np.round(widths[round_widths]/factor)*factor
        line_widths[floor_widths] = np.floor(widths[floor_widths])
        draw_multiple_lines = line_widths >= edge_width
        line_widths[draw_multiple_lines] = np.minimum(line_widths[draw_multiple_lines]-1, 10)
        unique_line_widths = np.unique(line_widths)
        
        for width in unique_line_widths:
            if width == 0.:
                continue
            
            loop_range = 2 if width == 3. else 1
            for j in range(loop_range):
                select = line_widths == width
                if loop_range == 2:
                    select &= (draw_multiple_lines if j == 0 else ~draw_multiple_lines)
                
                indices = np.where(select)[0]
                pos, color = [], []
                for i in indices:
                    pos += [vertices[i]]
                    if widths[i] > max([edge_width, width]):
                        pos[-1] = self.artificially_increase_linewidth(pos[-1], widths[i], width)
                    color += [np.tile(colors[i], (len(pos[-1]), 1))]
                    
                if len(pos):
                    pos = np.concatenate(pos)
                    color = np.concatenate(color)
                    index = np.where(np.abs(self.hodo_line_widths-width) < 0.1)[0][j]
                    self.visuals['hodo_line'][index].set_data(pos=pos, color=color)
                    self.visuals['hodo_line'][index].visible = True
      
        if max_vorticity:
            self.visuals['hodo_line_legendtext'].text = ['\u03c9  .01', format(max_vorticity, '.3f').lstrip('0')+'/s']
            delta_left = 0.5*legend_width1/self.hodo_size_px*self.plotrange
            delta_right = 0.5*legend_width2/self.hodo_size_px*self.plotrange
            legend_text_xpos = [legend_lines_xpos[0]-delta_left-0.0565*self.plotrange]
            legend_text_xpos += [legend_lines_xpos[1]+delta_right+0.0625*self.plotrange]
            self.visuals['hodo_line_legendtext'].pos = [[i, legend_ypos-0.005*self.plotrange] for i in legend_text_xpos]
                
               
    def plot_hodo_points_and_sigmacircles(self):
        V_available = len(self.V) > 0
        self.visuals['hodo_sigmacircles'].visible = self.visuals['hodo_points'].visible = V_available
        if V_available:
            marker_colors = self.cm_filled_sectors.map(self.filled_sectors/36-self.vvp_min_frac_sectors_filled)
            self.visuals['hodo_points'].set_data(pos=self.V, symbol='o', size=self.pb.scale_pixelsize(5.75),
                                                       face_color=marker_colors, edge_color=None, edge_width=0)
            marker_colors[:, 3] = 0.034
            self.visuals['hodo_sigmacircles'].set_data(pos=self.V, symbol='o', size=2*self.sigma, scaling=True,
                                                       face_color=marker_colors, edge_color=None)
            
    def plot_hodo_hlabels(self):
        if len(self.h_layers) > 0:
            heights_plot = np.arange(np.ceil(self.h_layers[0]),np.floor(self.h_layers[-1])+1e-3,1)
            heights_plot_text = heights_plot.astype(int).astype(str)
            if int(self.h_layers[0])==0 or not self.h_layers[0]/int(self.h_layers[0]) == 1.:
                heights_plot = np.append(self.h_layers[0], heights_plot)
                heights_plot_text = np.append(format(self.h_layers[0], '.1f').strip('0'), heights_plot_text)

            self.visuals['hodo_hmarkers'].visible = True; self.visuals['hodo_htext'].visible = True
            V_heights_plot = np.array([f.interpolate(self.h_layers, self.V, j)[-1] for j in heights_plot])
            self.visuals['hodo_hmarkers'].set_data(pos=V_heights_plot,symbol='o',face_color='black',edge_width=0,size=self.pb.scale_pixelsize(15))
            self.visuals['hodo_htext'].text = heights_plot_text
            dy = self.dy_center_text(self.visuals['hodo_htext'].font_size)
            #To compensate for the fact that anchor_y = 'center' for TextVisual doesn't fully center the text
            self.visuals['hodo_htext'].pos = V_heights_plot + np.array([0, dy])
        else:
            self.visuals['hodo_hmarkers'].visible = False; self.visuals['hodo_htext'].visible = False
            
            
    def plot_sm_markers(self):
        if len(self.V) > 1 and (not self.MW is None or self.is_SM_defined()):
            pos, edge_color = [], []
            for sm in self.sms_display:
                SM = getattr(self, sm)
                if not SM is None:
                    pos.append(SM)
                    edge_color.append(gv.vwp_sm_colors[sm])
            pos = np.array(pos)
            if len(pos):
                self.visuals['hodo_sm'].visible = True
                self.visuals['hodo_sm'].set_data(pos=pos,symbol='x',face_color=None,edge_color=edge_color,edge_width=self.pb.scale_pixelsize(2.75),size=self.pb.scale_pixelsize(21))
        else:
            self.visuals['hodo_sm'].visible = False
        
    def plot_cbar_legends(self):
        self.visuals['cbar_streamwise_vorticity_sign'].visible = self.visuals['cbar_filled_sectors'].visible = True
        sm = 'SM' if self.is_SM_defined() else 'RM'
        self.general_text_left_text += ['Streamwise component of vorticity (\u03c9) for '+sm]
        self.general_text_left_pos += [[0.01, 0]]
        self.general_text_right_text += ['% of 36 azimuthal sectors that passed data check']
        self.general_text_right_pos += [[0.99, self.cbarbottom_ypos+0.01]]
        self.general_text_center_text += ['-','+',str(int(self.vvp_min_frac_sectors_filled*100)),'100']
        self.general_text_center_pos += [[0.035, self.cbartop_ypos-0.0045],[0.465, self.cbartop_ypos-0.0045],
                                         [0.535, self.cbartop_ypos-0.0045],[0.965, self.cbartop_ypos-0.0045]]
                                
    def plot_height_list(self):
        n = min(len(self.h_layers), 5)
        text = 'h (km AGL) = '
        
        for i in range(0, n):
            if self.h_layers[i] == 0.:
                text += '%d' % self.h_layers[i]
            else:
                text += '%.1f' % self.h_layers[i]# if ft.rndec(self.h_layers[i], 2) == ft.rndec(self.h_layers[i], 1) else\
                           #'%.2f' % self.h_layers[i]
            if not i == n-1:
                text += ', '
        if not n == len(self.h_layers):
            text += '...'
        if text == 'h (km AGL) = ':
            text += '--'
            
        if n > 0 and self.h_layers[0] == 0.:
            text += f"\nSurface elevation: {self.sfcobs['station_elev']} m"
        text += f'\nRadar elevation: {gv.radar_elevations[self.crd.radar]} m'
        
        self.general_text_left_text += [text]
        self.general_text_left_pos += [[0.01, self.ytop]]
        
    def display_parameters(self):
        if len(self.V) < 2:
            return
        units = self.pb.productunits['v']
        y0 = y1 = self.ytop+0.155
        dy = 0.041
        dy_tables = 0.0325
        
        text, pos = ['h (km)', f'BWD ({units})'], [[0.01, y0], [0.01, y0+dy]]
        for i, layer in enumerate(self.shear):
            text += [layer, f'{self.shear[layer]:0.0f}']
            pos += [[0.19+0.1*i, y0], [0.19+0.1*i, y0+dy]]
        y1 += 2*dy+dy_tables
            
        text += ['h (km)', '\u03c9s (/s)', '\u03c9c (/s)', f'SRW ({units})']
        pos += [[0.01, y1]] + [[0.16+0.13*i, y1] for i in range(3)]
        for i, layer in enumerate(self.avg_sr_windspeed):
            if not self.avg_sw_vorticity[layer] is None:
                text += [layer, f'{self.avg_sw_vorticity[layer]:0.3f}'.replace('0.', '.'), f'{self.avg_cw_vorticity[layer]:0.3f}'.replace('0.', '.'),
                         f'{self.avg_sr_windspeed[layer]:0.0f}'.replace('0.', '.')]
            else:
                text += [layer, '--', '--', '--']
            pos += [[0.01, y1+(1+i)*dy]] + [[0.16+0.13*j, y1+(1+i)*dy] for j in range(3)]
        y1 += 3*dy+dy_tables
            
        text += ['SRH (m\u00B2/s\u00B2)']
        pos += [[0.01, y1]]
        text += ['h (km)']+list(self.srh)[::-1]
        pos += [[0.01, y1+dy]] + [[0.16+0.115*i, y1+dy] for i in range(len(self.srh))]
        for i, sm in enumerate(list(self.srh)[::-1]):
            for j, layer in enumerate(self.srh[sm]):
                if i == 0:
                    text += [layer]
                    pos += [[0.01, y1+(2+j)*dy]]
                text += [f'{self.srh[sm][layer]:0.0f}' if not self.srh[sm][layer] is None else '--']
                pos += [[0.16+0.115*i, y1+(2+j)*dy]]
            
        self.general_text_left_text += text
        self.general_text_left_pos += pos
            
    def format_h(self, numbers):
        return ft.format_nums(numbers, dec=1, separator="-")
        
    def calculate_params(self):
        units = self.pb.productunits['v']
        
        self.MW, self.LM, self.RM = f.get_bunkers_stormmotions(self.V, self.h_layers, units)
        self.SM = None
        if self.is_SM_defined():
            SM = self.gui.stormmotion*np.array([np.pi/180, self.pb.scale_factors['v']])
            self.SM = -SM[1] * np.array([np.sin(SM[0]), np.cos(SM[0])])
            
        self.base_sm_name = 'SM' if self.is_SM_defined() else 'RM'
        self.base_sm = self.__dict__[self.base_sm_name]
        
        self.DTM = None
        if not self.base_sm is None:
            self.DTM = f.get_deviant_tornado_motion(self.V, self.h_layers, self.base_sm)
        
        shear_kmranges = [j for j in list(self.gui.vwp_shear_layers.values()) if len(j)]
        self.shear = {}
        for kmrange in shear_kmranges:
            h_min, h_max, shear = f.calculate_bulk_shear(self.V, self.h_layers, kmrange[0], kmrange[1])
            self.shear[self.format_h([h_min, h_max])] = shear
        
        srh_kmranges = [j for j in list(self.gui.vwp_srh_layers.values()) if len(j)]
        self.srh = {}
        for sm in (j for j in self.sms_display if j != 'DTM'):
            self.srh[sm] = {}
            for kmrange in srh_kmranges:
                if not self.__dict__[sm] is None:
                    h_min, h_max, srh = f.calculate_SRH(self.V, self.h_layers, kmrange[0], kmrange[1], self.__dict__[sm], units)
                    self.srh[sm][self.format_h([h_min, h_max])] = srh
                else:
                    self.srh[sm][self.format_h(kmrange)] = None
        
        self.vorticity = f.calculate_shear_profile(self.V, self.h_layers, units)
        if not self.base_sm is None:
            self.V_int, self.h_int = f.interpolate_velocity_profile(self.V, self.h_layers, units, 3.)
            self.sw_vorticity, self.cw_vorticity = f.calculate_vorticity_components(self.V_int, self.h_int, self.base_sm, units)
            self.sr_windspeed = f.calculate_SR_windspeed(self.V_int, self.base_sm)
        else:
            self.sw_vorticity = self.cw_vorticity = self.sr_windspeed = []
        
        self.avg_sw_vorticity, self.avg_cw_vorticity, self.avg_sr_windspeed = {}, {}, {}
        for kmrange in ([0, 0.5], [0, 1]):
            if not self.base_sm is None:
                h_min, h_max, avg_sw_vorticity = f.calculate_avg_quantity_halflevels(self.sw_vorticity, self.h_int, self.V_int, kmrange[0], kmrange[1])
                layer = self.format_h([h_min, h_max])
                self.avg_sw_vorticity[layer] = avg_sw_vorticity
                _, _, avg_cw_vorticity = f.calculate_avg_quantity_halflevels(self.cw_vorticity, self.h_int, self.V_int, kmrange[0], kmrange[1])
                self.avg_cw_vorticity[layer] = avg_cw_vorticity
                _, _, avg_sr_windspeed = f.calculate_avg_quantity_halflevels(self.sr_windspeed, self.h_int, self.V_int, kmrange[0], kmrange[1])
                self.avg_sr_windspeed[layer] = avg_sr_windspeed
            else:
                layer = self.format_h(kmrange)
                self.avg_sw_vorticity[layer] = self.avg_cw_vorticity[layer] = self.avg_sr_windspeed[layer] = None
                
    def plot_legend(self):
        if len(self.V) > 1:
            legend_labels = [gv.vwp_sm_names[j] for j in self.sms_display]
            x = 0.68
            y_top = self.ytop+0.025
            dy = 0.056
            legend_pos = np.array([[x, y_top+i*dy] for i in range(len(legend_labels))])
            edge_color = [gv.vwp_sm_colors[j] for j in self.sms_display]
            if len(legend_pos) > 0:
                self.visuals['legend_markers'].visible = True; self.visuals['legend_labels'].visible = True
                self.visuals['legend_markers'].set_data(pos = legend_pos, symbol='x',face_color=None,edge_color=edge_color,edge_width=self.pb.scale_pixelsize(2.75),size=self.pb.scale_pixelsize(21))
                self.visuals['legend_labels'].pos = legend_pos + np.array([0.0425, 0])
                self.visuals['legend_labels'].text = legend_labels
                return
        self.visuals['legend_markers'].visible = False; self.visuals['legend_labels'].visible = False            
        
    def plot_sm_text(self):
        if len(self.V) > 1:
            units = self.pb.productunits['v']
            if len(self.sms_display) == 0:
                return    
            
            SMs = []
            for sm in self.sms_display:
                if not getattr(self, sm) is None:
                    uv = getattr(self, sm)
                    direction = (np.rad2deg(np.arctan2(uv[0], uv[1])) + 180) % 360
                    speed = np.linalg.norm(uv)
                    SMs.append([format(int(round(direction)), '03d'), format(int(round(speed)), '02d')])
                else:
                    SMs.append('--')
            n_SMs = len(SMs)
            
            #Try to get about equal amounts of white space above and below the text with storm motions.
            b = (self.n_params+4)*0.04125 #Relative y-pos of the top of the VWP legend text
            t = n_SMs*0.058 #Relative y-pos of the bottom of the hodograph legend
            ydim_sm_text = (n_SMs+1)*0.04125 #Relative y-dimension of the sm text
            n = int(round((b-t-ydim_sm_text)/2/0.04125)) #Number of white lines between the VWP legend and the sm text
            n_white = self.n_params - n_SMs - n
            text1 = ' \n'*n_white+'Storm motions\n'; text2 = ' \n'*n_white+' \n'
            for i in range(n_SMs):
                text1 += self.sms_display[i]+':'
                text2 += (SMs[i][0]+u'\u00b0'+SMs[i][1]+' '+units) if isinstance(SMs[i], list) else SMs[i]
                if i != n_SMs-1:
                    text1 += '\n'; text2 += '\n'
            
            self.general_text_left_text += [text1, text2]
            self.general_text_left_pos += [[0.66, self.ytop+0.1], [0.77, self.ytop+0.1]]
            
    def is_SM_defined(self):
        return self.gui.stormmotion[1] != 0.
    
    def plot_vwp_legend(self):
        text = 'VWP retrieval method: VVP'
        if not self.gui.vwp_sigmamax_mps is None:
            text += ',  \u03C3max: '+format(self.scale_velocity(self.gui.vwp_sigmamax_mps), '.1f')+' '+self.pb.productunits['v']
        text += '\nVVP horizontal data range: '+str(self.gui.vvp_range_limits[0])+'-'+str(self.gui.vvp_range_limits[1])+' km,  vmin: '+format(self.scale_velocity(self.gui.vvp_vmin_mps), '.1f')+' '+self.pb.productunits['v']
        self.general_text_center_text += [text]
        self.general_text_center_pos += [[0.5, self.ytop+0.66 - (12-self.n_params)*0.04125]]
        
    def plot_sfcobs_legend(self):
        relative_pos = ft.aeqd(gv.radarcoords[self.crd.radar], self.sfcobs['station_coords'])
        if all([j in self.sfcobs for j in ('T', 'Td')]):
            T_string = f"T/Td: {self.sfcobs['T']}/{self.sfcobs['Td']} \u00b0C"
        elif 'T' in self.sfcobs:
            T_string = f"T: {self.sfcobs['T']} \u00b0C"
            
        if self.display_manual_sfcobs:
            text = ['Surface (manual): '+self.sfcobs['station_fullname']+\
                   ' ('+self.sfcobs['datetime'][-4:-2]+':'+self.sfcobs['datetime'][-2:]+'Z)'+\
                   (', source: '+self.sfcobs['datasource'] if 'datasource' in self.sfcobs else ''),
                   (T_string+',  p' if 'T' in self.sfcobs else 'P')+\
                   'osition: ('+format(relative_pos[0], '.1f')+', '+format(relative_pos[1], '.1f')+') km']
        else:    
            text = ['Surface: '+self.sfcobs['station_fullname']+\
                   ' ('+self.sfcobs['datetime'][-4:-2]+':'+self.sfcobs['datetime'][-2:]+'Z)',
                   (T_string+',  p' if 'T' in self.sfcobs else 'P')+\
                   'osition: ('+format(relative_pos[0], '.1f')+', '+format(relative_pos[1], '.1f')+') km']
                
        self.general_text_center_text += ['\n'.join(text)]
        self.general_text_center_pos += [[0.5, self.ytop+0.765 - (12-self.n_params)*0.04125]]
    
    
    def import_sfc_obs(self):
        #Tries to import surface observations and returns a boolean stating whether this succeeded
        if self.display_manual_sfcobs:
            self.sfcobs = self.gui.vwp_manual_sfcobs.copy()
        else:
            try:
                self.sfcobs = self.sfcobs_classes[self.dsg.data_source()].get_sfcobs_at_time(self.crd.radar, self.crd.date, self.crd.time)            
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print(e, 'retrieving_sfc_obs')
      
        
    def scale_velocity(self, v):
        return v*gv.scale_factors_velocities[self.pb.productunits['v']]
    
    def change_axlim_only(self):
        self.get_hodo_xyrange()
        self.set_hodo_sttransform()
        self.update_hodo_clipper()
        self.plot_hodo_circles()
                
        self.plot_hodo_circles()
        self.plot_hodo_line_and_legend()
        self.plot_hodo_axislabels()        
        self.plot_hodo_hlabels()
        self.plot_sm_markers() 
        
    def get_current_data_name(self):
        radar_dataset = self.dsg.get_radar_dataset()
        subdataset = self.dsg.get_subdataset(product='v')
        #Include self.dsg.scannumbers_all['v'], to ensure that updating takes place when more data gets added to the currently viewed radar volume
        scannumbers = str(self.dsg.scannumbers_all.get('v', ''))
        return radar_dataset+subdataset+self.crd.date+self.crd.time+scannumbers+str(self.display_manual_sfcobs)+str(self.gui.vwp_manual_sfcobs)+str(self.gui.include_sfcobs_vwp)+str(self.gui.vvp_range_limits)+str(self.gui.vvp_height_limits)+str(self.gui.vvp_vmin_mps)
    
    def set_newdata(self):
        if all(self.pb.data_empty.values()):
            return
        self.updating_vwp = True
        
        self.general_text_left_text = []; self.general_text_left_pos = []
        self.general_text_right_text = []; self.general_text_right_pos = []
        self.general_text_center_text = []; self.general_text_center_pos = []
         
        if self.data_name != self.get_current_data_name():
            self.sfcobs = None
            t = pytime.time()
            if not self.sfcobs and self.gui.include_sfcobs_vwp:
                self.import_sfc_obs()
            print(pytime.time()-t, 't_sfc')
                
            try:
                h0 = (self.sfcobs['station_elev']-gv.radar_elevations[self.crd.radar])/1e3
                dd, ff = self.sfcobs['DD']*np.pi/180., self.sfcobs['FF']
                V0 = -ff*np.array([np.sin(dd), np.cos(dd)])
            except Exception:
                h0 = 0
                V0 = None
            
            t = pytime.time()
            try:
                self.h_layers_all, self.V_all, self.w_all, self.sigma_all, self.filled_sectors_all, self.volume_starttime, self.volume_endtime = \
                    self.vvp(range_limits=self.gui.vvp_range_limits, height_limits=self.gui.vvp_height_limits, v_min=self.gui.vvp_vmin_mps, h0=h0, V0=V0)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print(e, 'VVP')
                # In this case either an empty hodograph should be displayed, or the last non-empty hodograph.
                # The latter can be useful when switching between datasets, with one dataset not having velocities
                # available. This option is chosen when the datetimes do not differ by more than 15 minutes, otherwise
                # an empty hodograph is displayed.                    
                if self.crd.radar != self.data_radar or abs(ft.datetimediff_s(self.crd.date+self.crd.time, self.data_datetime)) > 15*60:
                    self.h_layers_all = self.V_all = self.w_all = self.sigma_all = self.filled_sectors_all = np.array([])
                    self.volume_starttime = '--'; self.volume_endtime = '--'                 
                else:
                    return # The hodograph is not altered
            print(pytime.time()-t, 't_vvp')
            
            if not V0 is None:
                try:
                    if self.h_layers_all[0] != 0.:
                        for i in ('h_layers', 'V', 'sigma', 'filled_sectors', 'w'):
                            ndims = len(self.__dict__[i+'_all'].shape)
                            self.__dict__[i+'_all'] = np.insert(self.__dict__[i+'_all'], 0, [0.]*ndims, axis=0)
                    self.V_all[0] = V0
                    self.sigma_all[0] = 0
                    self.filled_sectors_all[0] = 36
                    self.w_all[0] = np.nan
                except Exception as e:
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print(e, 'setting_sfc_obs')
        
            
        select = self.sigma_all <= self.gui.vwp_sigmamax_mps if self.gui.vwp_sigmamax_mps else np.s_[:]
        for i in ('h_layers', 'V', 'sigma', 'filled_sectors', 'w'):
            self.__dict__[i] = self.__dict__[i+'_all'][select].copy() # Copy is needed when all elements are selected
            if i in ('V', 'sigma', 'w'):
                self.__dict__[i] *= self.pb.scale_factors['v']
                
                
        self.sms_display = [j for j in gv.vwp_sm_names if (j == 'SM' and self.is_SM_defined() or j != 'SM' and self.gui.vwp_sm_display[j])]
        
        if len(self.V) > 1:
            self.calculate_params()
        
        self.plot_title()        
        
        self.dr_circles = self.dr_circles_unit[self.pb.productunits['v']]
        self.get_hodo_xyrange()
        self.set_hodo_sttransform()
        self.update_hodo_clipper()
        self.plot_hodo_circles()
                
        self.plot_hodo_line_and_legend()
        self.plot_hodo_points_and_sigmacircles()
        self.plot_hodo_axislabels()        
        self.plot_hodo_hlabels()
        self.plot_sm_markers()        
            
        self.plot_cbar_legends()
        self.ytop = self.cbarbottom_ypos+0.0725
        self.plot_height_list()
        self.n_params = 12 # this has to be set in order to prevent errors in other functions.
        self.display_parameters()
        self.plot_legend()
        self.plot_sm_text()
        self.plot_vwp_legend()
        if self.sfcobs:
            self.plot_sfcobs_legend()
            
        self.visuals['general_text_anchorx=left'].text = self.general_text_left_text
        self.visuals['general_text_anchorx=left'].pos = self.general_text_left_pos
        self.visuals['general_text_anchorx=right'].text = self.general_text_right_text
        self.visuals['general_text_anchorx=right'].pos = self.general_text_right_pos
        self.visuals['general_text_anchorx=center'].text = self.general_text_center_text
        self.visuals['general_text_anchorx=center'].pos = self.general_text_center_pos
        
        
        self.firstplot_performed=True
        self.updating_vwp = False
        self.data_name = self.get_current_data_name()
        self.data_radar = self.crd.radar
        self.data_datetime = self.crd.date+self.crd.time