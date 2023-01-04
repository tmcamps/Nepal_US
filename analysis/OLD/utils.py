import pickle

def extract_parameters(data, get_name=False):
    ''' This function extract metadata information on the dicom and outputs to dictionary '''

    dct_params_1 = {'Model_name': 0x00081090,
                    'Manufacturer': 0x00080070}
    dct_params_2 = {'RCX0': 0x00186018,
                    'RCY0': 0x0018601a,
                    'RCX1': 0x0018601c,
                    'RCY1': 0x0018601e,
                    'Phys_units_X': 0x00186024,
                    'Phys_units_Y': 0x00186026,
                    'Phys_delta_X': 0x0018602c,
                    'Phys_delta_Y': 0x0018602e}

    seq = data[0x0018, 0x6011]
    data_sub = seq[0]

    dct_params = {}
    for key in dct_params_1.keys():
        try:                                                                # Possible that not all headerfields are found
            dct_params[key] = [data[dct_params_1[key]].value]
        except:
            dct_params[key] = ['None']

    for key in dct_params_2.keys():
        try:                                                                # Possible that not all headerfields are found
            dct_params[key] = [data_sub[dct_params_2[key]].value]
        except:
            dct_params[key] = ['None']

    # if get_name:
    #     plt.imshow(data.pixel_array), plt.show()
    #     cond = 'N'
    #     while cond == 'N':
    #         answer = input('Give the name of the transducer: ')
    #         print(answer)
    #         cond = input('Is the name okay [Y/N]?')
    #         dct_params['Transducer_name'] = [answer]

    return dct_params

def save_dct(obj, filename):
    ''' saves dictionary to pickle'''
    with open(filename,'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
