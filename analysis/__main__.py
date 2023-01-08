''' Import packages '''
from analysis.reverberation_analysis.air_reverberation_analysis import analysis
import os
from get_project_root import root_path

# ''' Initialize values for single folder run '''
# settings = 'settings_data1.yaml'
# folder = 'data1'
# qc = analysis(settings, folder)
# qc.run()
#
# ''' Initialize values for single folder run '''
# settings = 'settings_7L4B.yaml'
# folder = '202301061536450016SMP'
# qc = analysis(settings, folder)
# qc.run()

# ''' Initialize values for single folder run '''
# settings = 'settings_C6-2.yaml'
# folder = '202301061528360006ABD'
# qc = analysis(settings, folder)
# qc.run()


''' Initialize values for all folders run '''
project_root = root_path(ignore_cwd=True)
data_directory = os.path.join(project_root, 'analysis', 'data')

for file in os.listdir(data_directory):
    folder = os.fsdecode(file)
    if 'ABD' in folder:
        settings = 'settings_C6-2.yaml'
    elif 'SMP' in folder:
        settings = 'settings_7L4B.yaml'
    elif 'data1' in folder:
        settings = 'settings_data1.yaml'
    elif 'data2' in folder:
        settings = 'settings_data2.yaml'
    else:
        continue
    qc = analysis(settings, folder)
    qc.run()
