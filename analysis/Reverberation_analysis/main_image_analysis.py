import pydicom
from .in_air_analysis_orginal import in_air_image_analysis
from utils_analysis_original import rgb2gray, transform_convex2linear, crop_image, pixels2mm
from utils import save_dct
import in_air_analysis_new as us_analysis
import os


def main_analysis_original(path_data, path_LUT_table):
    '''
    Main function to perform in-air ultrasound image analysis on the data located on path; path_data

    The analysis is based on the following publications:
        van Horssen P, Schilham A, Dickerscheid D, et al. Automated quality control of ultrasound based on in-air
        reverberation patterns. Ultrasound. 2017;25(4):229-238. doi:10.1177/1742271X17733145

        Aljahdali, M. H., Woodman, A., Al-Jamea, L., et al. Image Analysis for Ultrasound Quality Assurance.
        Ultrasonic Imaging. 2021; 113–123. https://doi.org/10.1177/0161734621992332

    Return dictionary with the following results:
            S_depth: scalar, unit value depth
            U_cov: scalar, horizontal profile covariance
            U_low: list, horizontal profile segment 10%, 20%, 40%, 20%, 10%
                MSE minimum for each segment
            U_skew: scalar, horizontal profile
            hor_prof:  vector, horizontal profile
            vert_prof: vector, vertical profile
            im: cropped image
            name: name of the device
            transducer_name:  name of the transducer
            num_reverb:  number of reverberation lines used in analysis
            unit: unit os S_depth
            date: date when the image was taken

    :param path_data: str, path to data
    :param path_LUT_table: str, path to Look-up-table for transducer names
    :return: dictionary
    '''

    # Read in data
    data = pydicom.dcmread(path_data, force=True)

    # Extract required dicom metadata
    transducer_type = data.TransducerType                                       # transducer type; convex/linear
    # TODO: add function to manual determine transducer type if transducer type is not available
    delta_x = data.SequenceOfUltrasoundRegions[0]['0x0018602c'].value           # physical delta X
    delta_y = data.SequenceOfUltrasoundRegions[0]['0x0018602e'].value           # physical delta Y
    unit = data.SequenceOfUltrasoundRegions[0]['0x00186024'].value              # Physical Units X & Y Direction
    label = data[0x00081010].value                                              # Station Name
    transducer_freq = data.SequenceOfUltrasoundRegions[0]['0x00186030'].value   # Transducer frequency
    department = data[0x00081040].value                                         # Institutional Department Name
    model = data[0x00081090].value                                              # Manufacturer's Model Name
    date = data[0x00080020].value                                               # Study data

    # Create specific name for ultrasound device & transducer
    device_name = department + '_' + model + '_' + label                        # Set up specific device name
    transducer_name = transducer_type + '_' + str(transducer_freq)              # Set up specific transducer name

    # Determine unit type
    if unit == 3:
        unit = 'cm'
    else:
        unit = 'not in meters'

    # Get air scan image
    im = data.pixel_array

    # Convert image to grayscale if image is truecolor RGB
    if im.ndim == 3:
        im = rgb2gray(im)

    # Convert image from convex to linear if transducer type is convex
    if transducer_type == 'CURVED LINEAR':
        # Convert image using the polar transform and bilinear interpolation
        im_transform = transform_convex2linear(im)

        # Crop the ultrasound image
        im_crop = crop_image(im_transform, crop2half=False)

        # Set the number of reverberation lines that should be detected to 5 for all curved linear transducers
        num_reverberations = 5

    else:                                                                        # Transducer type = linear
        # Crop the ultrasound image
        im_crop = crop_image(im, crop2half=True)

        # Set the number of reverberation lines that should be detected to 4 for all linear transducers
        num_reverberations = 4

    # Enhance the data to show more clearly where dips and enhancements occur.
    #im_crop = enhance_data(im_crop)

    # Perform the image analysis of the in air reverberation patterns based on the beforementioned publications
    vertical_prof, horizontal_prof, s_depth, u_cov, u_skew, u_low = in_air_image_analysis(im_crop, num_reverberations)

    # Translate s_depth pixels into mm
    s_depth = pixels2mm(s_depth, delta_x, unit)

    # Return results as dictionary

    return {'s_depth': s_depth,
            'u_cov': u_cov,
            'u_low': u_low,
            'u_skew': u_skew,
            'horizontal_prof': horizontal_prof.tolist(),
            'vertical_prof': vertical_prof.tolist(),
            'im': im_crop.tolist(),
            'device_name': device_name,
            'transducer_name': transducer_name,
            'number_reverberations': num_reverberations,
            'unit': unit,
            'date': date}

def writeimages(qc, results,name):

    fname = 'overview{}.jpg'.format(name)
    qc.save_annotated_image(fname, what='overview')
    results.addObject(os.path.splitext(fname)[0],fname)

def main_analysis_new(path_data):
    params = {'auto_suffix': False,
            'circle_fitfrac': 0.3333333333333333,
            'cluster_fminsize': 300.0,
            'cluster_mode': "all_middle",
            'f_dead': 0.3,
            'f_weak': 0.5,
            'hcor_px': 0,
            'rgbchannel': "B",
            'signal_thresh': 0,
            'vcor_px': 0}


    wrapper_params = [
         'rgbchannel', 'auto_suffix',
     ]  # parameters for wrapper only

    # Read in data
    data = pydicom.dcmread(path_data, force=True)

    # Create ID
    label = data[0x00081010].value                                              # Station Name
    department = data[0x00081040].value                                         # Institutional Department Name
    model = data[0x00081090].value                                              # Manufacturer's Model Name

    device_name = department + '_' + model + '_' + label

    # Transpose pixeldata to create im
    im = data.pixel_array #.transpose()

    # Build and populate qcstructure
    qc = us_analysis.analysis(data, im)

    for name, value in params.items():
        if not name in wrapper_params:
            qc.set_param(name, value)

    qc.run()

    results = {}
    # add results to 'result' object
    report = qc.get_report()
    # for section in report.keys():
    #     for key, vals in report[section].items():
    #         if vals[0] in ['int', 'float']:
    #             results.addFloat(key, vals[1])
    #         elif vals[0] in ['string']:
    #             results.addString(key, vals[1])
    #         elif vals[0] in ['bool']:
    #             results.addBool(key, vals[1])
    #         elif vals[0] in ['object']:
    #             results.addObject(key , vals[1])
    #         else:
    #             raise ValueError("Result '{}' has unknown result type '{}'".format(key, vals[0]))

    # writeimages(qc, results, device_name)
    # save_dct(results, device_name)

    return report




