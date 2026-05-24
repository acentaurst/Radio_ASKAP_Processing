import numpy as np
import math
import time
import os
import glob

import pandas as pd

from astropy.io.votable import parse
from astroquery.casda import Casda
from astroquery.utils.tap.core import TapPlus
from astroquery.utils.tap.core import Tap
from astropy.io.votable import parse, parse_single_table

import getpass

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as un

import keyring
keyring.core.set_keyring(keyring.core.load_keyring('keyrings.cryptfile.cryptfile.CryptFileKeyring'))
print("Keyring method: " + str(keyring.get_keyring()))

OPAL_USER = "qhua0119@uni.sydney.edu.au"        # set to opal login username
casda = Casda()
casda.login(username=OPAL_USER, store_password=True)

def convert_xml_to_pandas(xml_file_name):
    votable = parse(xml_file_name)
    table = votable.get_first_table()
    bill = table.to_table(use_names_over_ids=True)
    return bill.to_pandas()

# Set up the TAP url
tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")

# This is the search function, here I'm searching onl
# for continuum catalogues. You can change this to search
# for other data products instead
job = tap.launch_job_async(("SELECT TOP 50000 * FROM ivoa.obscore where(dataproduct_subtype = 'catalogue.continuum.component')"))
r = job.get_results()

# Here I keep only good or uncertain data
data = r[(r['quality_level'] == 'GOOD') | (r['quality_level'] == 'UNCERTAIN')]

# You have to do this step unless you have permission
# for embargoed data associated with you OPAL
# account login
public_data = Casda.filter_out_unreleased(data)

# Get the centre coords of all of the observations
# in the table
public_coords = SkyCoord(np.array(public_data['s_ra'])*un.deg,
                         np.array(public_data['s_dec'])*un.deg)

# Get the file names
public_files = np.array(public_data['filename'])

# 修改这里：更改为服务器上的文件路径
casda_filepath = '/import/ada1/qhua0119/Catalogue/'

# I like pandas better
pubdat = public_data.to_pandas()

print('Number of rows: ', len(pubdat.index))

# ————————————————————————————————————————————————————————————————————————————————————————————————————————-
# 读取Ultracool sheet - 需要先将此文件上传到服务器
ultracool_file = '/import/ada1/qhua0119/UltracoolSheet_Main.csv'  # 修改为服务器上的路径
ultracool_df = pd.read_csv(ultracool_file)



# 修改循环的起始和结束索引
start_idx = 0
end_idx = len(ultracool_df['ra_j2000_formula'])

for i in range(start_idx, end_idx):


    # 计数
    print('Processing of number', i, 'All data number is', len(ultracool_df['ra_j2000_formula']))

    # The coordinates of the source you're interested in
    example_source_coords = SkyCoord(ultracool_df['ra_j2000_formula'][i] * un.deg, ultracool_df['dec_j2000_formula'][i] * un.deg)
    # This sets how far away from the centre of the image
    # that you're searching
    example_radius = 3 * un.deg

    # Find the rows in the public data table
    # that are within example_radius of your
    # source
    seps = example_source_coords.separation(public_coords)
    matches = np.where(seps < example_radius)[0]

    # Do a little coordinate check to
    # Make sure you're getting the files you want
    pubdat.iloc[matches][['s_ra', 's_dec']]

    matching_files = np.array(pubdat.iloc[matches]['filename'])
    url_list = []

    # This part stages the files you want to download
    # so it sometimes takes a minute
    for mfile in matching_files:
        pdata = public_data[public_data['filename'] == mfile]
        url = casda.stage_data(pdata)
        if url not in url_list:
            url_list += url

    # Now download your files to the server
    filelist = casda.download_files(url_list, savedir=casda_filepath)

    # print(filelist)
