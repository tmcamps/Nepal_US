''' Import packages and functions '''
from analysis.reverberation_analysis.air_reverberation_analysis import analysis
import os
from get_project_root import root_path

#%%
''' Initialize values for all folders run '''
project_root = root_path(ignore_cwd=True)
data_directory = os.path.join(project_root, 'analysis', 'data')
devices = ['Z50', 'DC40']

# Loop through device folders
for device in devices:
    # Select specific device directory
    device_directory = os.path.join(data_directory, device)

    # Loop through probe folder in device folder
    for probe in os.listdir(device_directory):
        probe = os.fsdecode(probe)

        # Select specific probe directory
        probe_directory = os.path.join(device_directory, probe)

        settings = 'settings_' + probe + '.yaml'

        # Loop through image folders in probe folder
        for folder in os.listdir(probe_directory):
            folder = os.fsdecode(folder)

            # Initialize quality assurance function
            qc = analysis(settings, folder)

            # Run quality assurance for the specific image
            qc.run()


