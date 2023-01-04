import os
from get_project_root import root_path
import yaml
import pydicom

class initialize_data:
    def __init__(self, settings_file, meta_data_file):
        # Initialize the roots and filenames
        self.project_root = root_path(ignore_cwd=True)
        self.analysis_root = os.path.join(self.project_root, 'analysis')
        self.settings_file = settings_file
        self.meta_data_file = meta_data_file

        # Create dictionary for dicom meta data parameters
        self.meta_data = {}

        # label to add to results
        self.label = ""

        # Create dictionary for analysis parameters
        self.params = {}

    def open_yaml(self, file):
        analysis_root = self.analysis_root
        path_settings =  os.path.join(analysis_root, 'settings')
        path_file = os.path.join(path_settings, file)

        dct = yaml.load(open(path_file), Loader=yaml.FullLoader)

        return dct

    def load_data(self):
        path_data = os.path.join(self.analysis_root, self.params['path_data'])
        filename = self.params['file_name']
        path_datafile = os.path.join(path_data, filename)
        self.path_datafile = path_datafile

        self.data = pydicom.dcmread(path_datafile, force=True)
        self.im = self.data.pixel_array

    def extract_parameters_metadata(self):
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

    def set_label(self):
        transducer_type = self.meta_data['transducer_type']
        date = self.meta_data['date']

        self.label = transducer_type + '-' + date + '-'

    def run(self):
        self.params = self.open_yaml(self.settings_file)
        self.dct_metadata = self.open_yaml(self.meta_data_file)

        self.load_data()
        self.extract_parameters_metadata()
        self.set_label()