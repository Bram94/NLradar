import os

radar_idmap = {
    'Den Helder': 'rad_nl61',
    'Herwijnen': 'rad_nl62',
    'Zaventem': 'rad_be61',
    'Jabbeke': 'rad_be63',
    'Wideumont': 'rad_be62',
    'Helchteren': 'rad_be64',
    'Neuheilenbach': 'de_nhb_10605',
    'Essen': 'de_ess_10410',
    'Borkum': 'de_asb_10103'
}

radar_sourcemap = {
    'Den Helder': 'KNMI',
    'Herwijnen': 'KNMI',
    'Zaventem': 'KMI',
    'Jabbeke': 'KMI',
    'Wideumont': 'KMI',
    'Helchteren': 'KMI',
    'Neuheilenbach': 'DWD',
    'Essen': 'DWD',
    'Borkum': 'DWD'}

id_radarmap = {j: i for i, j in radar_idmap.items()}

used_radars = ['Den Helder','Herwijnen','Jabbeke','Wideumont','Borkum','Essen','Neuheilenbach']
used_radars = ['Wideumont']
# used_radars = ['Jabbeke']
available_radars = []

directory = 'D:/Nowcasting/Code/'
directory_ecmwf = ''
directory_savedtrackingdata = 'D:/Nowcasting/Saved_trackingdata/'
directory_tmp = ''
directory_provenance = ''
directories_radardata = {}

os.environ['WIDI_BUCKET'] = 'knmi-radar-obs-data-prd'
    
def get_radar_directory(radar, datetime, filename):
    date, time = datetime[:8], datetime[-4:]
    if radar in ('Jabbeke','Wideumont'):
        dataset = filename[filename.index('.hdf')-1].upper()
        return '/mnt/f/radar_data_NLradar/KMI/'+date+'/'+radar+'_'+dataset+'/'
    elif radar in ('Helchteren','Zaventem'):
        return '/mnt/f/radar_data_NLradar/KMI/'+date+'/'+radar+'/'
    elif radar in ('Den Helder','Herwijnen'):
        return '/mnt/f/radar_data_NLradar/Current/KNMI/'+radar.replace(' ','')+'/'+date+'/'
    elif radar in ('Borkum','Essen','Neuheilenbach'):
        hour = str(int(time)//100*100)
        next_hour = format((int(hour)+100)%2400, '04d')
        dataset = 'V' if 'vol5' in filename else 'Z'
        return '/mnt/f/radar_data_NLradar/DWD/'+date+'/'+radar+'_'+dataset+'/'+hour+'-'+next_hour+'/'

def create_directories():
    print('Creating directories...')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory_ecmwf):
        os.makedirs(directory_ecmwf)
    if not os.path.exists(directory_savedtrackingdata):
        os.makedirs(directory_savedtrackingdata)
    if not os.path.exists(directory_tmp):
        os.makedirs(directory_tmp)
    if not os.path.exists(directory_provenance):
        os.makedirs(directory_provenance)
        
    for radar in used_radars:
        if not os.path.exists(directories_radardata[radar]):
            os.makedirs(directories_radardata[radar])

template_filename = directory + 'cell_track/config/RAD_NL25_NA_NA.h5'
nowcast_pickle_filepath = os.path.join(directory_tmp, 'nowcast_obj.p')

def get_filename_product(datetime, product):
    return directory_tmp+'preflits/'+product+'_'+datetime+'.npy'
def get_filename_PREFLITS_finished(datetime):
    return directory_tmp+'preflits/preflits_finished_'+datetime+'.txt'
def get_output_dir_cell_track(datetime):
    return directory_savedtrackingdata+datetime[:8]+'/'


provenance_output_celltrack = os.path.join(directory_provenance, 'prov_celltrack_output_{iteration_id}.json')
provenance_input_meshname = os.path.join(directory_provenance, 'prov_mesh_input_{iteration_id}.json')
provenance_input_ivs = os.path.join(directory_provenance, 'prov_ivs_input_{iteration_id}.json')