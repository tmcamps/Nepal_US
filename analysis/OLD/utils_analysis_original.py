import cv2 as cv
from scipy import ndimage
import numpy as np

def rgb2gray(rgb):
    """
    Function to convert rgb image to grayscale image

    :param rgb: array, truecolor image RGB

    :return:gray: array, grayscale image
    """

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def select_largest(BW):
    """
    Function to select largest component; largest connected area of nonzero pixels.
    Performed to select only the transducer image and remove all text and other information

    :param BW: array, input binary image

    :return:BW_largest: array, binary image containing only largest connected area of nonzero pixels
    """


    labels_im, num_features = ndimage.label(BW)                                 # Label connected components in the image
    assert (labels_im.max() != 0)                                               # Assume at least 1 component
    sel_large = np.argmax(np.bincount(labels_im.flat)[1:])                      # Find label of component with the highest amount of occurrences
                                                                                # Do not take first bin into account because that is the background
    BW_new = labels_im == sel_large + 1                                         # Select only largest connected area from image, +1 to include
                                                                                # background bin again

    BW_largest = BW_new

    return BW_largest

def erosion_largest_dilation(BW, dilate=True, kernel=np.ones((5, 5), np.uint8), smooth_factor=2):
    """
    Function to perform first erosion, then take the largest component and finish with dilatation

    :param BW:
    :param dilate: booarray, input binary imagelean, if dilatation should be performed, default = True
    :param kernel: kernel for erosion and dilation, default is 5x5
    :param smooth_factor: number of iterations in image erosion and dilation, default = 2

    :return:BW_new: array, processes binary image
    """

    # Erode the binary image using the erode function from the OpenCV-Python library
    im_erosion = cv.erode(BW.astype("float"), kernel, iterations=smooth_factor)                # Perform erosion, with kernel size and smooth_factor

    # Take the largest component; largest connected area of nonzero pixels
    im_largest = select_largest(im_erosion)

    # Dilate the binary image using the dilate function from the openCV-Python library if input param: dilate = True
    if dilate:
        im_dilation = cv.dilate(im_largest.astype("float"), kernel, iterations=smooth_factor)  # Perform dilation, with kernel size and
        # smooth_factor
        BW_new = im_dilation

    else:
        BW_new = im_largest

    return BW_new

def enhance_data(im):
    """
    Data can be enhanced by subtracting line average (and adding image average to restore abs value).
    That shows more clearly where dips and enhancements occur. Also it can be used to "isolate" the
    reverberation frequency.
    """

    avg = np.average(im)
    im_avg = np.empty(im.shape, dtype=np.float)
    im_prof = np.average(im, axis=0)
    for i, d in enumerate(im_prof):
        im_avg[:, i] = im[:, i] - d + avg

    return im_avg

def transform_convex2linear(im):
    """
    This function transforms the convex transducer image to linear using
    the polar transform.

    :param im: array, grayscale convex image

    :return:im_transform: array, grayscale linear image
    """

    # Crop bottom half (=noise) away
    im = im[0:int(im.shape[0] / 2), :]

    # Find the borders/offset values of the ultrasound image data
    # Create a binary image by using thresholding
    BW = im > 1.5 * np.mean(im[0:int(im.shape[0] / 2), :])

    # Find the offset value by performing erosion and taking the largest component
    dilate = False
    BW_new = erosion_largest_dilation(BW, dilate)           # Also removes all text present in the image
    BW = BW_new

    # Find the indices of the pixels that have value 1 in the new binary image
    values_ind = np.argwhere(BW == 1)

    # Select x indices and y indices
    x = values_ind[:, 1]
    y = values_ind[:, 0]

    # Compute radius for finding offset to polar transform
    # Find the transducer upper edges to detect the peaks of the transducer image
    y_min = np.min(y)                                           # min y index value
    y_min += 1                                                  # Increment by 1 to ensure the detection of the two peaks of the transducer image

    # Find exact locations of the peaks
    peak = BW[y_min, :]                                         # Select binary image of y_min and all x
    x_values = np.argwhere(peak == 1)                           # Find x indices of the peaks of the transducer image
    threshold = np.mean(x_values)                               # Create threshold by calculating the mean value of the x indices
    x_ind = np.argwhere(x_values > threshold)                   # Find indices of x values greater than threshold, to find the right peak
    x_end = int(np.mean(x_values[x_ind][:, 0]))                 # Find the x index of the right edge by taking the centre pixel (=mean)
    x_ind = np.argwhere(x_values < threshold)                   # Find indices of x values smaller than threshold, to find the left peak
    x_start = int(np.mean(x_values[x_ind][:, 0]))               # Find the x index of the left edge by taking the centre pixel (=mean)

    # Determine the length in the x-direction of the transducer image
    length = (x_end - x_start) / 2

    # Check if the peaks are detected
    # x_pos = x_start + length
    # plt.imshow(BW)
    # plt.plot([x_start,x_end], [y_min-1, y_min-1], 'r.')
    # plt.show()

    # Determine the height in the y-direction of the transducer image
    x_center = int(x_start + length)                            # Find centre in x-direction of the transducer image
    y_center = y[np.argwhere(x == x_center)]                    # Find corresponding y values to the center in x-direction
    y_end = np.min(y_center)                                    # Find lower border of the transducer image
    height = y_end - y_min                                      # Determine the height of the transducer image in the y-direction

    # Compute the radius for the polar transform
    radius = (length ** 2 + height ** 2) / (2 * height)

    # Determine the offset for the polar transform
    offset = int(radius - height)

    # Determine mask for the transducer image using dilatation to preserve image regions
    BW = im > 1.5 * np.mean(im[0:int(im.shape[0] / 2), :])      # Create a binary image by using thresholding
    kernel = np.ones((3,3), np.uint8)                          # Set kernel to 3x3 size
    smooth_factor = 1                                           # Set smooth factor to 1, to only iterate once
    dilate = True                                               # Set dilation to True
    BW = erosion_largest_dilation(BW, dilate, kernel, smooth_factor)

    # Find the indices of the pixels that have value 1 in the new binary image
    values_ind = np.argwhere(BW == 1)

    # Select x indices and y indices
    x = values_ind[:, 1]
    y = values_ind[:, 0]

    # Determine the edges of the transducer image
    y_min = np.min(y)
    y_max = np.max(y)
    x_min = np.min(x)
    x_max = np.max(x)

    # Crop image and binary mask to the edges of the transducer image
    im_crop = im[y_min:y_max, x_min:x_max]
    BW_crop = BW[y_min:y_max, x_min:x_max]

    # Segment the transducer image using the binary mask
    im_segment = im_crop * BW_crop                              # Removes all text, noise and other information from the image

    # # Check segmentation
    # plt.subplot(3,1,1)
    # plt.imshow(im_crop)
    # plt.subplot(3,1,2)
    # plt.imshow(BW_crop)
    # plt.subplot(3,1,3)
    # plt.imshow(im_segment)
    # plt.show()

    # Create temporary zeros array
    temp = np.zeros([im_segment.shape[0] + offset, im_segment.shape[1]])

    # Fill the temporary array with the segmented image
    temp[offset:, :] = im_segment

    # Create enlarged temporary zeros array to cover the whole area
    temp_enlarge = np.zeros((2 * temp.shape[0], 2 * temp.shape[0]))

    # Determine the offsets and end set for the enlarged array
    offset_x = int(np.round(temp_enlarge.shape[0] / 2))
    offset_y = int(np.round(-temp.shape[1] / 2 + temp_enlarge.shape[1] / 2))
    end_y = int(offset_y + temp.shape[1])

    # Fill the temporary enlarged array with the segmented image
    temp_enlarge[offset_x:, offset_y:end_y] = temp
    temp_enlarge = temp_enlarge.astype(np.float32)          # Set data type to type float

    # Set temporary image to final enlarged image
    im_enlarge = temp_enlarge

    # To use the entire width/height of the original image is used to express the complete circular range of the resulting polar image:
    # calculate the square root of the sum of squares of the image dimensions
    maxRadius = np.sqrt(((im_enlarge.shape[0] / 2.0) ** 2.0) + ((im_enlarge.shape[1] / 2.0) ** 2.0))

    # Remap an image to polar coordinates space using the linearPolar function
    # Inputs:
    #           src: source image
    #           center: transformation center
    #           maxRadius: The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.
    #           Option: FillOutliers, fills all of the destination image pixels
    # Output:
    #           dst: destination image with same size as src
    polar_image = cv.linearPolar(im_enlarge, (im_enlarge.shape[0] / 2, im_enlarge.shape[1] / 2), maxRadius, cv.WARP_FILL_OUTLIERS)
    polar_image_transpose = np.transpose(polar_image)
    polar_image_fliplr = np.fliplr(polar_image_transpose)

    polar_image = polar_image_fliplr

    return polar_image

def crop_image(im, crop2half=True):
    """
    This function crops the ultrasound image to the image content ie. extracts the outer regions of the US dicom image.

    :param im: array, grayscale image
    :param crop2half: flag, for additional half row crop

    :return: im_crop: array, cropped grayscale image to analysis content
    """


    # Create binary image using a threshold of 0
    BW = im > 0

    # Take the largest component; largest connected area of nonzero pixels
    im_largest = select_largest(BW)

    # Find locations of content
    element_values = np.argwhere(im_largest)                           # Find indices of area of nonzero pixels
    x_values = element_values[:,0]                                      # Select only x indices
    y_values = element_values[:,1]                                      # Select only y indices

    # Find edges of the transducer image
    x_min = np.min(x_values)                                            # Find left boundary of transducer image
    x_max = np.max(x_values)                                            # Find right boundary of transducer image
    y_min = np.min(y_values)                                            # Find upper boundary of transducer image
    y_max = np.max(y_values)                                            # Find lower boundary of transducer image

    # Crop image to content 
    im_crop = im[x_min:x_max, y_min:y_max]

    if crop2half:
        # Crop the image vertically as the reverberations are only present in upper half of the image
        im_crop = im_crop[0:int(im.shape[0] / 2), :]

    return im_crop

def pixels2mm(pixels, delta_x, unit):
    """
    Function to translate pixels into mm

    :param pixels:
    :param data: filedataset of dicom data
    :return: pixels translated into mm
    """

    # Make sure provided dicom units are cm
    if unit != 3: # 3 = cm
        return -1

    # Convert physical delta x from mm to cm
    mm = 10.*delta_x

    return pixels*mm

