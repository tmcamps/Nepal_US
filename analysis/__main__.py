''' Import packages '''
from analysis.reverberation_analysis.air_reverberation_analysis import analysis

''' Initialize values '''
settings = 'settings_data1.yaml'
qc = analysis(settings, 'meta_data.yaml')
qc.run()
