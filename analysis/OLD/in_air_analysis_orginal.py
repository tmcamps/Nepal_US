import numpy as np
from scipy import stats
from scipy.signal import argrelextrema

def smooth(y, window_len):
    """
    Function to smooth vector y with box convolution using window size box_pts

    :param y: vector, input signal
    :param window: the length of the smoothing window

    :return y_smooth: smoothed output signal

    """
    #s = np.r_[2 * y[0] - y[window_len:1:-1], y, 2 * y[-1] - y[-1:-window_len:-1]]

    window = np.ones(window_len) / window_len
    y_smooth = np.convolve(y, window, mode='same')
    #y = np.convolve(window, s, mode='same')

    return y_smooth

def get_reverb_lines(vertical_profile, number_reverberations, smooth_factor=5):
    """
    Function to get the reverberation line positions from the detrended vertical profile minimas
    :param vertical_profile: vector, vertical profile
    :param number_reverberations: scalar, number of lines to be detected
    :param smooth_factor: value, how much the detrented profile will be smoothed, default is set to 5

    :return: reverberation_positions: vector, reverberation lines
    """

    detrend = np.zeros(vertical_profile.shape)
    vert_profile2 = smooth(vertical_profile, smooth_factor)

    for t in range(1, vertical_profile.shape[0]):
        detrend[t] = vert_profile2[t] - vert_profile2[t - 1]

    detrend[0] = detrend[1]
    locs = argrelextrema(detrend, np.less)  # find extremas
    locs = locs[0]
    reverberation_positions = locs[0:number_reverberations] + smooth_factor

    return reverberation_positions

def in_air_image_analysis(im_crop, num_reverberations):
    """
    Function that performs the ultrasound in air image analysis on the cropped image

    Calculates the following uniformity parameters:
    1. Ucov  reflects  the  overall  decrease  in  uniformity  due  to  noise  and  signal  loss,  as  well  as  any  changes  leading  to  an
    increase in its value calculated as a percentage.
    2. Uskew represents skewness of the distribution and a negative value that reflects non-symmetrical deformation of the uniformity.
    3. Ulow is  calculated  as  the  lowest  value  of  normalized  deviation  within the intensity profile v in a segment in percentage with
    respect to the entire area of interest median


    :param im_crop: array, cropped grayscale image to analysis content
    :param num_reverberations: number of reverberation lines that should be detected

    :return: vertical_prof: vector, vertical intensity profile
    :return: horizontal_prof: vector, horizontal profile
    :return: s_depth: scalar, pixel value depth
    :return: u_cov: scalar, horizontal profile covariance
    :return: u_skew: scalar, horizontal profile skewness
    :return: u_low: list, MSE minimum for each horizontal profile segment (10, 20, 40, 20, 10 %)
    """

    # The vertical intensity profile is composed of the median values of all sequential horizontal image lines intensity distributions
    vertical_prof = np.median(im_crop, axis=1)

    # Determine s_depth as the intersection of the noise and the vertical intensity profile
    background = im_crop[np.round(im_crop.shape[0] / 2).astype(int):, :]
    background_value = np.median(background.ravel())
    s_depth = np.argmin(np.abs(vertical_prof - background_value))

    # The horizontal intensity profile is derived from the mean intensity values of the vertical image lines within a fixed ROI
    # First calculate the reverberation lines positions
    reverberation_pos = get_reverb_lines(vertical_prof, num_reverberations, smooth_factor=5)

    # Compose the horizontal intensity profile within the ROI that contains the reverberation positions
    horizontal_prof = np.average(im_crop[0:reverberation_pos[-1],:], axis=0)

    # Evaluate the uniformity parameters from the horizontal profile (u)
    u = horizontal_prof
    x_u = np.linspace(0,100, len(u))

    # Calculate the coefficient of Variation over uniformity data = u_cov
    u_cov = 100 * (np.std(u)/np.mean(u))

    # Calculate the coefficient of skewness of the uniformity data = u_skew
    u_skew = stats.skew(u)

    # Calculate the lowest value of normalized deviation within the intensity profile in a segment
    segment = np.array([10, 20, 40, 20, 10])
    u_low = []                                              # init list
    ind_prev = 0                                            # init index
    seg = 0                                                 # init segment

    for s in range(len(segment)):
        seg = seg + segment[s]
        ind = np.argmin(np.abs(x_u - seg))
        v = u[ind_prev:ind]                                 # Select segment from horizontal profile
        low_val = 100 * np.min((v - np.median(u))/np.median(u))     # Calculate lowest value of segment
        u_low.append(np.abs(low_val))

        ind_prev = ind+1

    return vertical_prof, horizontal_prof, s_depth, u_cov, u_skew, u_low
