import numpy as np
import pandas as pd
import time
import glob
import os
from datetime import datetime
import shutil
import re
from astropy.coordinates import SkyCoord, Distance
import astropy.units as un
from astropy.time import Time
from astropy.visualization import quantity_support
# quantity_support()
from astropy.io.votable import parse
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from astroquery.utils.tap.core import Tap
from astropy.io.votable import parse, parse_single_table
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import keyring

keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))
print("Keyring method: " + str(keyring.get_keyring()))

OPAL_USER = "acentauri_huangst@163.com"
casda = Casda()
casda.login(username=OPAL_USER, store_password=True)

# 读取Visibility中所有的数据名称与SBID，用于后续Cutout下载
def get_folders_and_sb_numbers(base_path):
    records = []
    
    # 遍历主文件夹下的所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            longobs_path = os.path.join(folder_path, 'LongObs')
            
            # 检查LongObs文件夹是否存在
            if os.path.exists(longobs_path) and os.path.isdir(longobs_path):
                temp_sb_numbers = []
                
                # 遍历LongObs文件夹中的内容
                for item in os.listdir(longobs_path):
                    item_path = os.path.join(longobs_path, item)
                    
                    if os.path.isdir(item_path):
                        # 使用正则表达式匹配SBxxxx格式
                        match = re.match(r'SB(\d+)_beam\d+', item)
                        if match:
                            temp_sb_numbers.append(int(match.group(1)))
                
                # 去重并为每个唯一的SB数字创建记录
                unique_sb_numbers = sorted(list(set(temp_sb_numbers)))
                
                for sb_num in unique_sb_numbers:
                    records.append({
                        'Name': folder_name,
                        'SBID': sb_num
                    })
    
    return records

base_path = '/import/ada1/qhua0119/Visibility/UCS1-50'  # 路径
records = get_folders_and_sb_numbers(base_path)
# 创建DataFrame
DSinfo = pd.DataFrame(records)
# 再次确保没有重复组合（防止意外情况）
DSinfo = DSinfo.drop_duplicates().reset_index(drop=True)
# 显示结果
print(DSinfo)
print(f"\n总共有 {len(DSinfo)} 行记录")
#—————————————————————————————————————————————————————————
# 读取时间信息
Time_info = pd.read_csv('/import/ada1/qhua0119/All_combine_data_unrepetition.csv')
MJD_J2000 = 51545 # J2000 MJD

# 读取恒星信息
ultracool_file = '/import/ada1/qhua0119/UltracoolSheet_Main_index_unbinaryUCD.csv'  # 修改为服务器上的路径
ultracool_df = pd.read_csv(ultracool_file)
#—————————————————————————————————————————————————————————
for i in range(len(DSinfo['Name'])):
    source_name = DSinfo['Name'][i]
    SB = DSinfo['SBID'][i]
    # # Choose where you're going to save the files
    # cutout_path = f"/import/ada1/qhua0119/Cutout_plot/Data/{source_name}/{SB}"

    # # 如果文件夹不存在，则创建
    # if not os.path.exists(cutout_path):
    #     os.makedirs(cutout_path)

    Star = ultracool_df[ultracool_df['UCS_name'] == source_name].copy()
    
    print(Star)

    # Coordinates of your target
    source_coords = SkyCoord(Star['ra_j2000_formula'].item() * un.deg,
                            Star['dec_j2000_formula'].item() * un.deg,
                            pm_ra_cosdec = Star['pmra_formula'].item() * un.mas / un.yr,
                            pm_dec = Star['pmdec_formula'].item() * un.mas / un.yr,
                            frame='icrs',
                            obstime=Time('J2000.0'),
                            distance = Star['dist_formula'].item() * un.pc)
    cutout_width = 1 * un.arcmin # size of the cutout


    Stokes = ['I','V']
    for i in range (len(Stokes)):
        #根据StokeI/V存储文件
        cutout_path = f"/import/ada1/qhua0119/Cutout_plot/Data/{source_name}/{SB}/Stokes{Stokes[i]}"   
        # 如果文件夹不存在，则创建
        if not os.path.exists(cutout_path):
            os.makedirs(cutout_path)
        
        # If you'd like to look for Stokes V images, change '/I/' to '/V/'
        image_tap_qry = (F"SELECT * FROM ivoa.obscore where(pol_states = '/{Stokes[i]}/' and "
                        F"dataproduct_subtype = 'cont.restored.t0' and obs_id='ASKAP-{SB}') "
                        F"AND 1 = CONTAINS(POINT('ICRS',{source_coords.ra.deg},{source_coords.dec.deg}),s_region)")
                        # F"circle('ICRS', s_ra,s_dec, 3))")
        
        # Do you Tap+ query
        tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
        # Get the image file list
        job = tap.launch_job_async(image_tap_qry)
        r = job.get_results()
        r = Casda.filter_out_unreleased(r)
        # Convert the list to pandas
        image_list = r.to_pandas()

        # I do a bunch of filtering on the data quality and type
        image_list = image_list[image_list['obs_id'].str.contains('ASKAP')]
        image_list = image_list[image_list['quality_level'] != 'BAD']
        image_list = image_list[~image_list['filename'].str.contains('raw')]
        image_list = image_list[~image_list['filename'].str.contains('alt')]
        image_list = image_list[~image_list['filename'].str.contains('highres')]
        image_list = image_list[~image_list['filename'].str.contains('iqr')]
        image_list = image_list[~image_list['obs_collection'].str.contains('BETA')]

        # If you don't care about proper motion then you don't care
        # what time the observation happened, so you can remove this
        # line
        # image_list = image_list[~image_list['t_max'].isnull()]
        # 使用与已有表格交叉匹配的方法获取时间，原有t_max中约有一半的源没有时间信息
        # 先merge覆盖匹配的t_min（如果image_list已有t_min，会被新值替换）
        image_list = pd.merge(image_list, Time_info[['obs_id', 't_min']], on='obs_id', how='left', suffixes=('', '_y'))
        if 't_min_y' in image_list.columns:  # 处理冲突
            image_list['t_min'] = image_list['t_min_y'].combine_first(image_list['t_min'])
            image_list.drop(columns=['t_min_y'], inplace=True)
        # 新功能：不匹配的obs_id（t_min为NaN）设为60800
        image_list['t_min'] = image_list['t_min'].fillna(60800)
        # 重命名t_min为Time
        image_list.rename(columns={'t_min': 'Time'}, inplace=True)

        # List the filenames and observation times for the next step
        image_filenames = np.array(image_list['filename'])
        image_dates = Time(np.array(image_list['Time']), format='mjd')

        # If you care about proper motion, do it like this
        for f, filename in enumerate(image_filenames):
            epoch = image_dates[f]
            pm_coords = source_coords.apply_space_motion(epoch)
            url_info_df = Table.from_pandas(image_list[image_list['filename'] == filename])
            url_list = casda.cutout(url_info_df,
                                    coordinates=pm_coords, radius=cutout_width)
            #download cutout
            filelist = casda.download_files(url_list,
                                            savedir=cutout_path)
            cube_id = 'cube-' + (filelist[0].split('-imagecube-')[-1]).split('.fits')[0]
            new_name = F'{cutout_path}{source_name}_{filelist[0]}'