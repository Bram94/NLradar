# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import numpy as np

import nlr_globalvars as gv
import nlr_functions as ft

from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *



class GUI_VWP():
    def __init__(self, gui_class, parent=None):
        super(GUI_VWP, self).__init__() 
        self.gui = gui_class
        self.pb = self.gui.pb
        self.vwp = self.gui.vwp
        
        self.manual_sfcobs_parameternames = {'station_fullname':'Station name', 'station_elev':'Station elevation (m)', 
                  'datetime':'Datetime (YYYYMMDDHHMM)', 'station_coords':'Coordinates (latitude, longitude)', 
                  'datasource':'Data source (optional)',
                  'DD':'Wind direction (degrees)', 'FF':'Wind speed ('+self.pb.productunits['v']+')',
                  'T':'Temperature ('+u'\u00b0C, optional)', 'Td':'Dewpoint ('+u'\u00b0, optional)'}
    
    
    def update_vwp(self):
        if self.pb.firstplot_performed:
            self.vwp.set_newdata()
            self.pb.set_draw_action('vwp_only')
            self.pb.update()
    
    def showrightclickMenu(self, pos):
        menu=QMenu(self.gui)
        
        if self.gui.include_sfcobs_vwp:
            self.remove_sfcobs_action = QAction('Remove surface observations', menu)
            menu.addAction(self.remove_sfcobs_action)
            self.remove_sfcobs_action.triggered.connect(lambda: self.change_include_sfcobs_vwp(False))
        else:
            self.add_sfcobs_action = QAction('Add surface observations', menu)
            menu.addAction(self.add_sfcobs_action)
            self.add_sfcobs_action.triggered.connect(lambda: self.change_include_sfcobs_vwp(True))
                        
        self.set_manual_sfcobsly_action = QAction('Set surface observations manually', menu)
        menu.addAction(self.set_manual_sfcobsly_action)
        self.set_manual_sfcobsly_action.triggered.connect(self.show_manual_sfcobs_window)
        
        if self.vwp.display_manual_sfcobs:
            self.switch_to_automatic_sfcobs_action = QAction('Switch to automatic surface observations', menu)
            menu.addAction(self.switch_to_automatic_sfcobs_action)
            self.switch_to_automatic_sfcobs_action.triggered.connect(lambda: self.change_display_manual_sfcobs(False))
        else:
            self.switch_to_manual_sfcobs_action = QAction('Switch to manual surface observations', menu)
            menu.addAction(self.switch_to_manual_sfcobs_action)
            self.switch_to_manual_sfcobs_action.triggered.connect(lambda: self.change_display_manual_sfcobs(True))
            if len(self.gui.vwp_manual_sfcobs) == 0:
                self.switch_to_manual_sfcobs_action.setEnabled(False)
                
        menu.addSeparator()
        
        self.fix_current_axlim_action = QAction('Fix current axis limits', menu)
        menu.addAction(self.fix_current_axlim_action)
        self.fix_current_axlim_action.triggered.connect(self.fix_current_axlim)
        
        self.set_manual_axlim_action = QAction('Set axis limits manually', menu)
        menu.addAction(self.set_manual_axlim_action)
        self.set_manual_axlim_action.triggered.connect(self.show_manual_axlim_window)
        
        if self.vwp.display_manual_axlim:
            self.switch_to_automatic_axlim_action = QAction('Switch to automatic axis limits', menu)
            menu.addAction(self.switch_to_automatic_axlim_action)
            self.switch_to_automatic_axlim_action.triggered.connect(lambda: self.change_display_manual_axlim(False))
        else:
            self.switch_to_manual_axlim_action = QAction('Switch to manual axis limits', menu)
            menu.addAction(self.switch_to_manual_axlim_action)
            self.switch_to_manual_axlim_action.triggered.connect(lambda: self.change_display_manual_axlim(True))
            if len(self.gui.vwp_manual_axlim) == 0:
                self.switch_to_manual_axlim_action.setEnabled(False)   
                
        menu.addSeparator()
                       
        self.change_vvp_settings_action = QAction('Change VVP settings')
        menu.addAction(self.change_vvp_settings_action)
        self.change_vvp_settings_action.triggered.connect(self.show_vvp_settings_window)
                
        self.change_display_settings_action = QAction('Change display settings')
        menu.addAction(self.change_display_settings_action)
        self.change_display_settings_action.triggered.connect(self.show_display_settings_window)        
            
        menu.popup(self.gui.mapToGlobal(pos))
        
        
    def change_include_sfcobs_vwp(self, new_bool):
        self.gui.include_sfcobs_vwp = new_bool
        self.update_vwp()
    
    def show_manual_sfcobs_window(self):
        self.manual_sfcobs_window=QWidget()
        self.manual_sfcobs_window.setWindowTitle('Manual surface observations')
        layout=QVBoxLayout()
        form_layout=QFormLayout()
                
        self.manual_sfcobsw = {}
        for parameter in self.manual_sfcobs_parameternames:
            if parameter in self.gui.vwp_manual_sfcobs:
                parameter_value = self.gui.vwp_manual_sfcobs[parameter]
                if parameter == 'FF':
                    parameter_value = self.vwp.scale_velocity(parameter_value)                
                text = str(parameter_value) if not type(parameter_value) in (list, np.ndarray) else\
                       ft.list_to_string(parameter_value, separator=', ')
            else:
                text = ''
            self.manual_sfcobsw[parameter] = QLineEdit(text)
            if parameter == 'station_coords':
                self.set_station_coords_from_markerw = QPushButton('Use marker position')
                self.set_station_coords_from_markerw.clicked.connect(self.set_station_coords_from_marker)
                hbox_layout=QHBoxLayout()
                hbox_layout.addWidget(self.manual_sfcobsw[parameter])
                hbox_layout.addWidget(self.set_station_coords_from_markerw)
                form_layout.addRow(QLabel(self.manual_sfcobs_parameternames[parameter]), hbox_layout)
            else:    
                form_layout.addRow(QLabel(self.manual_sfcobs_parameternames[parameter]), self.manual_sfcobsw[parameter])
        layout.addLayout(form_layout)
        
        self.manual_sfcobs_setw = QPushButton('Set observations')
        self.manual_sfcobs_setw.clicked.connect(self.set_manual_sfcobs)
        layout.addWidget(self.manual_sfcobs_setw)
        
        self.manual_sfcobs_window.setLayout(layout)
        self.manual_sfcobs_window.resize(self.manual_sfcobs_window.sizeHint())
        self.manual_sfcobs_window.show() 
        
    def set_station_coords_from_marker(self):
        if not self.gui.marker_present:
            self.gui.set_textbar('Marker not set', 'red', 1)
        else:
            self.manual_sfcobsw['station_coords'].setText(ft.list_to_string(ft.rndec(self.pb.marker_coordinates, 6), separator=', '))
        
    def set_manual_sfcobs(self):
        input_incorrect = False
        new_manual_sfcobs = {}
        for parameter in self.manual_sfcobsw:
            input_text = self.manual_sfcobsw[parameter].text()
            if parameter == 'station_fullname' or (parameter == 'datasource' and input_text != ''):
                new_manual_sfcobs[parameter] = input_text
            elif parameter in ('station_elev', 'DD', 'FF') or (parameter in ('T', 'Td') and input_text != ''):
                try:
                    number = ft.to_number(input_text)
                    if parameter == 'FF':
                        number /= self.vwp.scale_velocity(1)
                    new_manual_sfcobs[parameter] = ft.rifdot0(number)
                except Exception:
                    input_incorrect = True
            elif parameter == 'datetime':    
                input_text = input_text.replace(' ','')
                if not ft.correct_datetimeinput(input_text[:8],input_text[-4:]):
                    input_incorrect = True
                else:
                    new_manual_sfcobs[parameter] = input_text
            elif parameter == 'station_coords':
                try:
                    lat, lon = ft.determine_latlon_from_inputstring(input_text)
                    new_manual_sfcobs[parameter] = np.array([lat,lon])    
                except Exception:
                    input_incorrect = True
                    
        if not input_incorrect:
            self.gui.vwp_manual_sfcobs = new_manual_sfcobs
            self.vwp.display_manual_sfcobs = True
            self.update_vwp()
            
    def change_display_manual_sfcobs(self, new_bool):
        self.vwp.display_manual_sfcobs = new_bool
        self.update_vwp()            
            
    def fix_current_axlim(self):
        self.gui.vwp_manual_axlim = [self.vwp.xmin, self.vwp.xmax, self.vwp.ymin, self.vwp.ymax]
        self.vwp.display_manual_axlim = True
        self.update_VWP_axlim()
        
    def show_manual_axlim_window(self):
        self.manual_axlim_window=QWidget()
        self.manual_axlim_window.setWindowTitle('Manual axis limits') 
        layout=QVBoxLayout()
        form_layout=QFormLayout()
        
        self.manual_axlimw = QLineEdit(ft.list_to_string(self.gui.vwp_manual_axlim))
        form_layout.addRow(QLabel('xmin, xmax, ymin, ymax (x and y range should be equal)'), self.manual_axlimw)
        layout.addLayout(form_layout)
        
        self.manual_axlim_setw = QPushButton('Set axis limits')
        self.manual_axlim_setw.clicked.connect(self.set_manual_axlim)
        layout.addWidget(self.manual_axlim_setw)
        
        self.manual_axlim_window.setLayout(layout)
        self.manual_axlim_window.resize(self.manual_axlim_window.sizeHint())
        self.manual_axlim_window.show() 
        
    def set_manual_axlim(self):
        input_incorrect = False
        new_manual_axlim = []

        input_string = self.manual_axlimw.text()
        input_list = ft.string_to_list(input_string)
        if len(input_list) != 4:
            input_incorrect = True
        else:
            for string in input_list:
                try:
                    number = ft.to_number(string)
                    new_manual_axlim += [ft.rifdot0(number)]
                except Exception:
                    input_incorrect = True  
                    break
                    
        if not input_incorrect:
            self.gui.vwp_manual_axlim = new_manual_axlim
            self.vwp.display_manual_axlim = True
            self.update_VWP_axlim()
            
    def change_display_manual_axlim(self, new_bool):
        self.vwp.display_manual_axlim = new_bool
        self.update_VWP_axlim()
            
    def update_VWP_axlim(self):
        self.vwp.change_axlim_only()
        self.pb.set_draw_action('vwp_only')
        self.pb.update()
        
        
    def show_vvp_settings_window(self):
        self.vvp_settings_window=QWidget()
        self.vvp_settings_window.setWindowTitle('VVP settings')
        layout=QVBoxLayout()
        
        self.vvp_range_limitsw=QLineEdit()
        self.vvp_range_limitsw.setText(ft.list_to_string(self.gui.vvp_range_limits, separator=', '))
        self.vvp_range_limitsw.editingFinished.connect(self.change_vvp_range_limits)
        layout.addWidget(QLabel('Horizontal data range limits (min. radius, max. radius) in km. Default: 2, 25 km.'))
        layout.addWidget(self.vvp_range_limitsw)
        
        self.vvp_height_limitsw=QLineEdit()
        self.vvp_height_limitsw.setText(ft.list_to_string(self.gui.vvp_height_limits, separator=', '))
        self.vvp_height_limitsw.editingFinished.connect(self.change_vvp_height_limits)
        layout.addWidget(QLabel('Height limits (min. height, max. height) in km. Default: 0.1, 11.9 km.'))
        layout.addWidget(self.vvp_height_limitsw)
        
        self.vvp_vmin_mpsw=QLineEdit()
        self.vvp_vmin_mpsw.setText(format(self.vwp.scale_velocity(self.gui.vvp_vmin_mps), '.1f'))
        self.vvp_vmin_mpsw.editingFinished.connect(self.change_vvp_vmin_mps)
        layout.addWidget(QLabel('Minimum radial velocity (vmin) in '+self.pb.productunits['v']+'. Default: '+format(self.vwp.scale_velocity(2), '.1f')+' '+self.pb.productunits['v']))
        layout.addWidget(self.vvp_vmin_mpsw)
        
        self.vvp_settings_window.setLayout(layout)
        self.vvp_settings_window.resize(self.vvp_settings_window.sizeHint())
        self.vvp_settings_window.show()
            
    def change_vvp_range_limits(self):
        try:
            new_range_limits = [ft.rifdot0(j) for j in ft.string_to_list(self.vvp_range_limitsw.text())]
            if new_range_limits[1] > new_range_limits[0]:
                self.gui.vvp_range_limits = new_range_limits
        except Exception:
            pass
        self.vvp_range_limitsw.setText(ft.list_to_string(self.gui.vvp_range_limits, separator=', '))  
        self.update_vwp()
        
    def change_vvp_height_limits(self):
        try:
            new_height_limits = [ft.rifdot0(j) for j in ft.string_to_list(self.vvp_height_limitsw.text())]
            if new_height_limits[1] > new_height_limits[0]:
                self.gui.vvp_height_limits = new_height_limits
        except Exception:
            pass
        self.vvp_height_limitsw.setText(ft.list_to_string(self.gui.vvp_height_limits, separator=', ')) 
        self.update_vwp()
            
    def change_vvp_vmin_mps(self):
        try:
            self.gui.vvp_vmin_mps = ft.rifdot0(ft.to_number(self.vvp_vmin_mpsw.text())) / self.vwp.scale_velocity(1)
        except Exception:
            pass
        self.vvp_vmin_mpsw.setText(str(ft.rifdot0(self.vwp.scale_velocity(self.gui.vvp_vmin_mps))))
        self.update_vwp()
        
        
    def show_display_settings_window(self):
        self.display_settings_window=QWidget()
        self.display_settings_window.setWindowTitle('VWP display settings')
        
        layout=QVBoxLayout()
        
        self.vwp_sigmamax_mpsw = QLineEdit(format(self.vwp.scale_velocity(self.gui.vwp_sigmamax_mps), '.1f') if not self.gui.vwp_sigmamax_mps is None else '')
        self.vwp_sigmamax_mpsw.editingFinished.connect(self.change_vwp_sigmamax_mps)
        
        self.vwp_shear_layersw, self.vwp_vorticity_layersw, self.vwp_srh_layersw = {}, {}, {}
        hbox_shear, hbox_vorticity, hbox_srh = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        for j in range(1, 5):
            self.vwp_shear_layersw[j] = QLineEdit(ft.list_to_string(self.gui.vwp_shear_layers[j], separator=', '))
            hbox_shear.addWidget(self.vwp_shear_layersw[j])
            self.vwp_shear_layersw[j].editingFinished.connect(lambda j=j: self.change_vwp_shear_layers(j))
        for j in range(1, 3):
            self.vwp_vorticity_layersw[j] = QLineEdit(ft.list_to_string(self.gui.vwp_vorticity_layers[j], separator=', '))
            hbox_vorticity.addWidget(self.vwp_vorticity_layersw[j])
            self.vwp_vorticity_layersw[j].editingFinished.connect(lambda j=j: self.change_vwp_vorticity_layers(j))
        for j in range(1, 4):    
            self.vwp_srh_layersw[j] = QLineEdit(ft.list_to_string(self.gui.vwp_srh_layers[j], separator=', '))
            hbox_srh.addWidget(self.vwp_srh_layersw[j])
            self.vwp_srh_layersw[j].editingFinished.connect(lambda j=j: self.change_vwp_srh_layers(j))
            
        self.vwp_sm_displayw = {}
        hbox_sm = QHBoxLayout()
        for j in ('MW', 'LM', 'RM'):
            self.vwp_sm_displayw[j] = QCheckBox(gv.vwp_sm_names[j])
            self.vwp_sm_displayw[j].setTristate(False)
            self.vwp_sm_displayw[j].setCheckState(2 if self.gui.vwp_sm_display[j] else 0)
            hbox_sm.addWidget(self.vwp_sm_displayw[j])
            self.vwp_sm_displayw[j].stateChanged.connect(lambda state, j=j: self.change_vwp_sm_display(j))
        
        layout.addWidget(QLabel('Maximum standard deviation (\u03C3max) in '+self.pb.productunits['v']+' for which to show VVP velocities.'))
        layout.addWidget(QLabel('Leave empty for no maximum.'))
        layout.addWidget(self.vwp_sigmamax_mpsw)
        layout.addWidget(QLabel('Height limits in km for the BWD, \u03c9+SRW and SRH calculations. Leave one or more'))
        layout.addWidget(QLabel('boxes empty if you want to limit calculations to less layers.'))
        layout.addLayout(hbox_shear)
        layout.addLayout(hbox_vorticity)
        layout.addLayout(hbox_srh)
        layout.addWidget(QLabel('Storm motions to display. The observed storm motion cannot be selected here, since'))
        layout.addWidget(QLabel('it is always (and only then) shown when a storm motion vector is set.'))
        layout.addLayout(hbox_sm)
        
        self.display_settings_window.setLayout(layout)
        self.display_settings_window.resize(self.display_settings_window.sizeHint())
        self.display_settings_window.show()
        
    def change_vwp_sigmamax_mps(self):
        try:
            input_text = self.vwp_sigmamax_mpsw.text()
            self.gui.vwp_sigmamax_mps = ft.rifdot0(ft.to_number(input_text)) / self.vwp.scale_velocity(1) if input_text else None
        except Exception:
            pass
        self.vwp_sigmamax_mpsw.setText(str(ft.rifdot0(self.vwp.scale_velocity(self.gui.vwp_sigmamax_mps))) if not self.gui.vwp_sigmamax_mps is None else '')
        self.update_vwp()
        
    def change_vwp_shear_layers(self, layer):
        try:
            new_shear_layer = [ft.rifdot0(j) for j in ft.string_to_list(self.vwp_shear_layersw[layer].text()) if not j == '']
            if len(new_shear_layer) == 0 or new_shear_layer[1] > new_shear_layer[0]:
                self.gui.vwp_shear_layers[layer] = new_shear_layer
        except Exception:
            pass
        self.vwp_shear_layersw[layer].setText(ft.list_to_string(self.gui.vwp_shear_layers[layer], separator=', '))
        self.update_vwp()
        
    def change_vwp_vorticity_layers(self, layer):
        try:
            new_vorticity_layer = [ft.rifdot0(j) for j in ft.string_to_list(self.vwp_vorticity_layersw[layer].text()) if not j == '']
            if len(new_vorticity_layer) == 0 or new_vorticity_layer[1] > new_vorticity_layer[0]:
                self.gui.vwp_vorticity_layers[layer] = new_vorticity_layer
        except Exception:
            pass
        self.vwp_vorticity_layersw[layer].setText(ft.list_to_string(self.gui.vwp_vorticity_layers[layer], separator=', '))
        self.update_vwp()
        
    def change_vwp_srh_layers(self, layer):
        try:
            new_srh_layer = [ft.rifdot0(j) for j in ft.string_to_list(self.vwp_srh_layersw[layer].text()) if not j == '']
            if len(new_srh_layer) == 0 or new_srh_layer[1] > new_srh_layer[0]:
                self.gui.vwp_srh_layers[layer] = new_srh_layer
        except Exception:
            pass
        self.vwp_srh_layersw[layer].setText(ft.list_to_string(self.gui.vwp_srh_layers[layer], separator=', '))
        self.update_vwp()
            
    def change_vwp_sm_display(self, sm):
        self.gui.vwp_sm_display[sm] = True if self.vwp_sm_displayw[sm].checkState() == 2 else False
        self.update_vwp()