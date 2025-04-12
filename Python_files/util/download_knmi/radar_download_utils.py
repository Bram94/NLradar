from datetime import datetime, timedelta
import boto3
import os
import re

import directories as d

if 'WIDI_BUCKET' in os.environ and os.environ['WIDI_BUCKET']:
    s3 = boto3.client('s3')
    s3_paginator = s3.get_paginator('list_objects_v2')

radar_definitions = {
    "DWD": {
        "Borkum": {
            "radar_prefix": "de_asb_10103",
            "scan_root_prefix": {"ras07-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-pcpng01_sweeph5onem_vradh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_vradh_": ["00-"]
                                 },
            "scan_postfix": "",
        },
        "Essen": {
            "radar_prefix": "de_ess_10410",
            "scan_root_prefix": {"ras07-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-pcpng01_sweeph5onem_vradh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_vradh_": ["00-"]
                                 },
            "scan_postfix": "",
        },
        "Neuheilenbach": {
            "radar_prefix": "de_nhb_10605",
            "scan_root_prefix": {"ras07-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_dbzh_": ["%02i-" % i for i in range(10)],
                                  "ras07-stqual-vol5minng01_sweeph5onem_vradh_": ["%02i-" % i for i in range(10)],
                                  "ras07-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-pcpng01_sweeph5onem_vradh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_dbzh_": ["00-"],
                                  "ras07-stqual-pcpng01_sweeph5onem_vradh_": ["00-"]
                                 },
            "scan_postfix": "",
        }
    },
    "KMI": {
        "Wideumont": {
            "radar_prefix": "rad_be62",
            "scan_root_prefix": "",
            "scan_postfix": ""
        },
        "Zaventem": {
            "radar_prefix": "rad_be61",
            "scan_root_prefix": "",
            "scan_postfix": ""
        },
        "Jabbeke": {
            "radar_prefix": "rad_be63",
            "scan_root_prefix": "",
            "scan_postfix": ""
        },
        "Helchteren": {
            "radar_prefix": "rad_be64",
            "scan_root_prefix": "",
            "scan_postfix": ""
        }
    },
    "KNMI": {
        "Den Helder": {
            "radar_prefix": "rad_nl61",
            "scan_root_prefix": "RAD_NL61_VOL_NA_",
            "scan_postfix": ""
        },
        "Herwijnen": {
            "radar_prefix": "rad_nl62",
            "scan_root_prefix": "RAD_NL62_VOL_NA_",
            "scan_postfix": ""
        }        
    }
}

def floor_5_minutes(date_time): #date_time can either be a datetime object, or it can be a string of the format 'YYYYMMDDHHMM'
    convert = not isinstance(date_time, datetime)
    if convert:
        date_time = datetime.strptime(date_time, '%Y%m%d%H%M')
    rounded = date_time - timedelta(minutes=date_time.minute % 5, seconds=date_time.second, microseconds=date_time.microsecond)
    return rounded if not convert else rounded.strftime('%Y%m%d%H%M')

def check_missing_radars(radar_dict):
    '''
    Determines if any radars are currently missing by looking at availability 
    of radar data in the last 3 timesteps of the radar file dictionary
    '''
    radar_timestamp_counts = {}
    d.available_radars = []

    # count per radar how many timestamps are available (last three timesteps)
    filtered_radar_dict = sorted(radar_dict.items(), reverse=True)
    if len(radar_dict) >= 3:
        filtered_radar_dict = filtered_radar_dict[:3]

    for radar in d.used_radars:
        radar_timestamp_counts[radar] = 0
        for timestamp in filtered_radar_dict:
            if radar in timestamp[1]:
                radar_timestamp_counts[radar] += 1

    # include radars with minimal of 1 timestamp available over the last 3 timestamps
    for radar in d.used_radars:
        if radar_timestamp_counts[radar] > 0:
            d.available_radars.append(radar)

    print('Available radars:', ', '.join(d.available_radars))

def find_files(timestamp_end, count=1, timestamp_start=None):
    '''
    Find all radar files for all used radars for a total of count timestamps before timstamp_end.
    Alternatively, if timestamp_start is specified, find all radar files for all used radars between
    timestamp_start and timestamp_end. Returns a dictionary with for each requested timestamp, for
    all found radars, a list of filenames.
    '''
    if not timestamp_start is None and not isinstance(timestamp_start, datetime):
        timestamp_start = datetime.strptime(timestamp_start, '%Y%m%d%H%M')
    if not isinstance(timestamp_end, datetime):
        timestamp_end = datetime.strptime(timestamp_end, '%Y%m%d%H%M')
    if timestamp_start is None:
        timestamp_start = timestamp_end - timedelta(minutes=5 * count)
    
    search_timestrings = []
    search_timestamp = timestamp_start - timedelta(minutes=timestamp_start.minute)
    while search_timestamp <= timestamp_end - timedelta(minutes=timestamp_end.minute):
        search_timestrings += [search_timestamp.strftime('%Y%m%d%H')]
        search_timestamp += timedelta(hours=1)

    found_files = {}

    for radar in d.used_radars:
        radar_definition = radar_definitions[d.radar_sourcemap[radar]][radar]
        
        radar_prefix = radar_definition.get('radar_prefix', False) or ""
        scan_root_prefix = radar_definition.get('scan_root_prefix', False) or ""
        scan_root_prefix = scan_root_prefix if isinstance(scan_root_prefix, dict) else {scan_root_prefix: [""]}
        scan_postfix = radar_definition.get('scan_postfix', False)  or ""
        
        for search_timestring in search_timestrings:
            for scan_root_pf, scan_parts in scan_root_prefix.items():
                for scan_part in scan_parts:
                    scan_prefix = scan_root_pf + scan_part
                    datetime_subpos = len(scan_prefix)
                    
                    bucket_prefix = 'radar/' + radar_prefix + '/' + scan_prefix + search_timestring
                    bucket_path = os.path.dirname(bucket_prefix)
                    if 'WIDI_BUCKET' in os.environ and os.environ['WIDI_BUCKET']:
                        pages = s3_paginator.paginate(Bucket=os.environ['WIDI_BUCKET'], Prefix=bucket_prefix)
                        
                        for page in pages:
                            if page['KeyCount'] == 0: break
                        
                            for obj in page['Contents']:
                                filename = os.path.basename(obj['Key'])

                                if len(scan_postfix) > 0 and scan_postfix not in filename: continue

                                str_datetime = filename[ datetime_subpos:datetime_subpos+12 ]
                                scan_datetime = datetime.strptime(str_datetime, '%Y%m%d%H%M')
                                if timestamp_start <= scan_datetime < timestamp_end + timedelta(minutes=5):
                                    floor_scan_datetime = floor_5_minutes(scan_datetime).strftime('%Y%m%d%H%M')
                                    if not floor_scan_datetime in found_files:
                                        found_files[floor_scan_datetime] = {}
                                    if not radar in found_files[floor_scan_datetime]:
                                        found_files[floor_scan_datetime][radar] = []
                                    found_files[floor_scan_datetime][radar].append(os.path.join(bucket_path,filename))
                    else:
                        directory = d.directories_radardata[radar]
                        for filename in os.listdir(directory):
                            if len(scan_postfix) > 0 and scan_postfix not in filename: continue

                            str_datetime = filename[ datetime_subpos:datetime_subpos+12 ]
                            scan_datetime = datetime.strptime(str_datetime, '%Y%m%d%H%M')
                            if timestamp_start <= scan_datetime < timestamp_end + timedelta(minutes=5):
                                floor_scan_datetime = floor_5_minutes(scan_datetime).strftime('%Y%m%d%H%M')
                                if not floor_scan_datetime in found_files:
                                    found_files[floor_scan_datetime] = {}
                                if not radar in found_files[floor_scan_datetime]:
                                    found_files[floor_scan_datetime][radar] = []
                                found_files[floor_scan_datetime][radar].append(os.path.join(bucket_path,filename))

    return found_files

def download_file(radar_id, filename, bucket_name, object_key):
    '''Download dataset into container if it isn't already, return its timestamp'''
    if radar_id.startswith('rad_nl'):
        timestamp = filename.split('_')[4].split('.')[0] # Truncate and remove file extension
    if radar_id.startswith('rad_be'):
        timestamp = filename[:12]
    if radar_id.startswith('de'):
        timestamp = floor_5_minutes(filename.split('-')[-4][:12])
        
    radar = d.id_radarmap[radar_id]
    directory = d.get_radar_directory(radar, timestamp, object_key)
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(directory + filename):
        print(f'Downloading {filename}')
        s3.download_file(bucket_name, object_key, directory + filename)

    return timestamp

def timestamp_is_complete(timestamp_file_dict):
    '''
    Check for each radar in timestamp_file_dict if we expect any more data to arrive from it,
    if not we consider the radar to be complete. Return if all radars are complete and the total
    number of complete radars.
    '''
    radar_complete_status = {}
    
    for radar in d.used_radars:
        if not radar in d.available_radars:
            # Skip missing radars
            continue

        if radar in timestamp_file_dict:
            radar_definition = radar_definitions[d.radar_sourcemap[radar]][radar]
            radar_check_regexes = []

            radar_prefixes = radar_definition.get('scan_root_prefix')
            radar_prefixes = radar_prefixes if isinstance(radar_prefixes, dict) else {radar_prefixes: [""]}

            radar_postfixes = radar_definition.get('scan_postfix')
            radar_postfixes = radar_postfixes if isinstance(radar_postfixes, list) else [ radar_postfixes ]

            # construct regexes from pre- and postfixes
            for prefix, scan_parts in radar_prefixes.items():
                for postfix in radar_postfixes:
                    radar_check_regexes.append(f'{prefix}{scan_parts[-1]}.*{postfix}')

            # check all regexes
            total_matches = 0
            for file in timestamp_file_dict[radar]:
                file_basename = os.path.basename(file)
                for radar_check_regex in radar_check_regexes:
                    pattern = re.compile(radar_check_regex)
                    if pattern.match(file_basename):
                        total_matches += 1

            radar_complete_status[radar] = len(radar_check_regexes) == total_matches
        else:
            radar_complete_status[radar] = False

    radars_available = list(radar_complete_status.values())

    return all(radars_available), sum(radars_available)




if __name__ == "__main__":
    file_dict = find_files('202207222300', count=12, timestamp_start=None)
    
    for timestamp in file_dict:
        for radar in file_dict[timestamp]:
            print(radar)
            for key in file_dict[timestamp][radar]:
                print(key)
                download_file(d.radar_idmap[radar], os.path.basename(key), os.environ['WIDI_BUCKET'], key)