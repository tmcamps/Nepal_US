''' Import packages '''
import os
import yaml

from analysis.OLD.main_image_analysis import main_analysis_original, main_analysis_new
from analysis.OLD.utils import extract_parameters
import pydicom
#%%
''' Initialize values '''
path = os.getcwd()                                                      # Get the current working directory of a process
settings_file = '../settings/settings_data2.yaml'  # Name of file with preset settings
path_settings = os.path.join(path, settings_file)                       # Get directory of settings file
settings_dct = yaml.load(open(path_settings), Loader=yaml.FullLoader)   # Load user settings from yaml file as dictionary

path_data = os.path.join(path, settings_dct['path_data'])               # Load directory for data
path_save = os.path.join(path, settings_dct['path_save'])               # Load directory for results
log_filename = settings_dct['log_filename']                             # Load file name for log file
path_LUT_table = os.path.join(path, settings_dct['file_LUT_table'])     # Load directory of LUT table


#%%
''' Run image data analysis '''
filenames = os.listdir(path_data)
data = os.path.join(path_data, filenames[0])
# dcm_data = pydicom.dcmread(data, force=True)
# im = dcm_data.pixel_array

results1 = main_analysis_original(data, path_LUT_table)

#%%
report = main_analysis_new(data)

#%%
''' Load data '''
filenames = os.listdir(path_data)

for filename in filenames:                                              # Loop through dicom data files and add metadata to table

    data = pydicom.dcmread(os.path.join(path_data, filename))           # Read dicom data
    dct_params = extract_parameters(data, get_name=False)               # Extract parameters of metadata
    #
    # df1 = pd.DataFrame(data=dct_params)                                 # Create dataframe using parameters
    # try:
    #     df = pd.read_excel(path_LUT_table)                                # Read excel reader
    # except:
    #     df = pd.DataFrame({})
    # try:
    #     df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # except:
    #     None
    #
    # frames = [df, df1]
    # df_total = pd.concat(frames)
    # df_total.to_excel(path_LUT_table)

