import os
from get_project_root import root_path
import yaml
import pydicom

from PIL import Image
from numpy import asarray

#%%
class initialize_data:
    def __init__(self, settings_file, folder):
        # Initialize the roots and filenames
        self.project_root = root_path(ignore_cwd=True)
        self.analysis_root = os.path.join(self.project_root, 'analysis')
        self.folder = folder
        self.settings_file = settings_file

        self.path_image = ""

        # Create dictionary for dicom meta data parameters
        self.meta_data = {}

        # label to add to results
        self.label = ""

        # Create dictionary for analysis parameters
        self.params = {}

    def open_yaml(self, file, type='settings'):
        analysis_root = self.analysis_root

        if type == 'settings':
            path = os.path.join(analysis_root, 'settings', file)

        elif type == 'meta_data':
            path = os.path.join(analysis_root, self.params['path_data'], self.folder, file)

        dct = yaml.load(open(path), Loader=yaml.FullLoader)

        return dct

    def load_bmp(self):
        folder = self.folder
        filename = self.params['file_name']
        path_image = os.path.join(self.analysis_root, self.params['path_data'], folder, filename)

        self.path_image = path_image

        # load the image
        image = Image.open(path_image).convert('L')

        # convert image to numpy array
        data = asarray(image)

        self.im = data

    def load_dcm(self):
        folder = self.folder
        filename = self.params['file_name']
        path_data = os.path.join(self.analysis_root, self.params['path_data'], folder, filename)

        self.data = pydicom.dcmread(path_data, force=True)
        self.im = self.data.pixel_array

    def extract_metadata_bmp(self):
        ''' This function extract metadata information from text file and outputs to dictionary '''

        folder = self.folder
        filename = self.params['meta_data_name']
        path_file = os.path.join(self.analysis_root, self.params['path_data'], folder, filename)

        # Read text file to dictionairy
        with open(path_file) as f:
            temp = dict(x.rstrip().split("=") for x in f)

        # Clean up dictionairy
        clean = {k.replace('\t', ''): v for k, v in temp.items()}

        # Change data type of dictionary values
        meta_data = {}
        for key, value in clean.items():
            try:
                meta_data[key] = int(value)
            except:
                meta_data[key] = value

        self.meta_data = meta_data

    def extract_metadata_dcm(self):
        ''' This function extract metadata information on the dicom and outputs to dictionary '''

        dct_metadata = self.dct_metadata
        data = self.data
        seq = data[0x0018, 0x6011]
        data_sub = seq[0]

        meta_data = {}

        for key in dct_metadata.keys():

            try:
                meta_data[key] = data[dct_metadata[key]].value
            except:
                meta_data[key] = data_sub[dct_metadata[key]].value

        self.meta_data = meta_data

    def set_label_bmp(self):
        transducer_type = self.meta_data[self.params['transducer_type']]
        date = self.folder[0:8]
        number = self.folder[14:18]

        self.label = transducer_type + '-' + date + '-' + number + '-'

    def set_label_dcm(self):
        transducer_type = self.meta_data['transducer_type']
        date = self.meta_data['date']

        self.label = transducer_type + '-' + date + '-'

    def run(self):
        self.params = self.open_yaml(self.settings_file)

        if '.dcm' in self.params['file_name']:
            self.dct_metadata = self.open_yaml(self.params['meta_data_name'], type='meta_data')

            self.load_dcm()
            self.extract_metadata_dcm()
            self.set_label_dcm()

        elif '.bmp':
            self.load_bmp()
            self.extract_metadata_bmp()
            self.set_label_bmp()

