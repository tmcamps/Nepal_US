import os
import numpy as np
import scipy.ndimage as scind
from PIL import Image
import operator
from scipy.signal import find_peaks

from analysis.utilizations import fit_circle, initialize_data, util_analysis, visualizations

helper = util_analysis.analysis_helper()
visualize = visualizations.visualizations()


class analysis:
    def __init__(self, settings, metadata):

        # Step 1: initialize data and dictionary
        init = initialize_data.initialize_data(settings, metadata)
        init.run()

        self.meta_data = init.meta_data
        self.params = init.params
        self.label = init.label
        self.im = init.im

        # Set path for saving data
        self.path = init.analysis_root
        self.path_save_excel = os.path.join(self.path, self.params['path_save'])
        self.path_save_images = os.path.join(self.path, self.params['path_save'], self.label)
        os.makedirs(self.path_save_images, exist_ok = True)

        # Create dictionary for results report
        self.report = {}

        # Create variables for analysis
        self.mask_rev = None
        self.is_curved = False

    def isolate_us_image(self):
        """
        Create mask containing only the ultrasound image data
        """

        # Initialize report section for mask
        report_section = 'us_image'
        self.report[report_section] = {}

        # Initialize variables
        signal_thresh = self.params['signal_thresh']
        image = self.im

        '''1. Create binary image with provided signal threshold'''
        BW = image > signal_thresh

        '''2. Find ultrasound data as largest connected component'''
        # Determine components and their labels
        cca, labels = scind.label(BW)

        # Remove small clusters
        cca_new, labels_new = helper.remove_small_cluster(cca, labels, BW, self)

        # Search for clusters present in the vertical middle
        labels_middle = helper.select_vertical_middle(cca_new)

        ''' 3. Create mask containing only clusters larger than minimum size and present in vertical middle'''
        mask = np.reshape(np.in1d(cca_new, labels_middle), np.shape(cca_new))

        '''4. Delete color bars from sides if initial values are provided'''
        if not self.params['init_x0x1'] is None:
            x0,x1 = self.params['init_x0x1']        # Provided initial box
            mask[:,:x0] = False                     # Select box from image
            mask[:,x1+1:] = False

        '''5. Save ultrasound image'''
        us_image = image * mask
        us_image = Image.fromarray(us_image)
        us_image.save(f"{self.path_save_images}/{self.label}ultrasound_image.png")

        '''6. Report results'''
        # Save upper and lower edges of the reverberation pattern
        # Create list with indices of the cluster representing the reverberation pattern
        clus = np.where(mask)
        clus = [(x, y) for x, y in zip(clus[0], clus[1])]

        # Determine upper and lower edges of the cluster
        minx = min(clus, key=operator.itemgetter(1))[1]
        maxx = max(clus, key=operator.itemgetter(1))[1]
        maxy = max(clus, key=operator.itemgetter(0))[0]
        miny = min(clus, key=operator.itemgetter(0))[0]

        # Save upper and lower edges
        self.params['us_x0y0x1y1'] = [minx, miny, maxx, maxy]

        self.report[report_section]['us_box_xmin_px'] = ('int', minx)
        self.report[report_section]['us_box_ymin_px'] = ('int', miny)
        self.report[report_section]['tus_box_xmax_px'] = ('int', maxx)
        self.report[report_section]['us_box_ymax_px'] = ('int', maxy)

        return mask

    def check_if_curved(self, mask, edges):
        """
        Function to determine if the image is acquired by a curved transducer

        1. Fit circle through top of reverberation image
        2. Determine if transducer is curved
        3. If the transducer is curved calculate limiting angles and radius of the fitted circle
        """

        x0, y0, x1, y1 = edges

        ''' 1. Fit circle through top of reverberation image '''
        # Find center x value
        center_x = int(0.5 * (x0 + x1) + .5)

        # From top of reverberation image down, look for pixels != 0 from mid to left and from mid to right
        # If both found, add it to the list
        circL_xy = []
        circR_xy = []
        for y in range(y0, y1 + 1):
            for x in reversed(range(x0, center_x)):  # find left point
                xl = -1
                xr = -1
                if mask[y, x]:
                    xl = x
                    if xl > -1:
                        circL_xy.append((xl, y))
                    break
            for x in range(center_x, x1):  # find right point
                if mask[y, x]:
                    xr = x
                    if xr > -1:
                        circR_xy.append((xr, y))
                    break

            if xr - xl < 10:  # Stop when
                break

        # Create complete circle to fit image
        circ_xy = []
        circ_xy.extend(circL_xy)
        circ_xy.extend(circR_xy)
        circ_xy.sort(key=operator.itemgetter(1))

        self.circ_xy = circ_xy
        self.circL_xy = circL_xy
        self.circR_xy = circR_xy

        ''' 2.  Determine if transducer is curved '''
        if len(circ_xy) < 11:
            # at least 10 point, else probably not a curved probe
            is_curved = False

        else:
            is_curved = True

        return is_curved

    def curve_transform(self, im_segment, edges, mask):
        """
        Function to transform reverberation pattern to rectangle through interpolation at coords

        1. Calculate limiting angles and radii of the fitted circle
        2. Create Cartesian matrix from the angles and radii
        3. Transform image to linear by interpolation
        4. Create and save transformed image

        """
        report_section = 'curved'
        self.report[report_section] = {}
        x0, y0, x1, y1 = edges

        ''' Calculate limiting angles and radii'''
        circle_fitfrac = self.params['circle_fitfrac']
        circ_xy = self.circ_xy
        circL_xy = self.circL_xy
        circR_xy = self.circR_xy

        # As deformation towards edges occur, use only central part for fitting
        fraction = 1 - circle_fitfrac
        cf_fraction = circ_xy[int(fraction * len(circ_xy)):]  # Create circle of only central part

        # Fit the fractionated circle: calculate optimized center of circle and determine radius of the circle
        cf = fit_circle.fit_circle(cf_fraction)
        (x_center, y_center), R = cf.circ_fit()

        # Calculate the limiting curvature of the circle at left side and right side in rad
        curve_left = np.arctan2(circL_xy[0][0] - x_center, circL_xy[0][1] - y_center)
        curve_right = np.arctan2(circR_xy[0][0] - x_center, circR_xy[0][1] - y_center)
        angles_curve = [curve_left, curve_right]

        # Calculate the limiting radii of the circle
        max_R = min([
            (x0 - x_center) / np.sin(angles_curve[0]),
            (x1 - x_center) / np.sin(angles_curve[1]),
            (y1 - y_center)])

        self.params['pt_curve_radii_px'] = [R, max_R]
        self.params['pt_curve_origin_px'] = [x_center, y_center, R]
        self.params['pt_curve_angles_deg'] = [c / np.pi * 180. for c in angles_curve]

        # Report the variables
        self.report[report_section]['curve_residue'] = ('float', cf.residue)  # residue of fitted curvature
        self.report[report_section]['curve_radmin_px'] = ('float', R)  # minimum radius of circle
        self.report[report_section]['curve_radmax_px'] = ('float', max_R)  # maximum radius of circle
        self.report[report_section]['curve_xc_px'] = ('float', x_center)  # x center of circle
        self.report[report_section]['curve_yc_px'] = ('float', y_center)  # y center of circle
        self.report[report_section]['curve_rc_px'] = ('float', R)  # Radius of circle
        self.report[report_section]['curve_angmin_deg'] = ('float', curve_left/np.pi*180)  # minimum angle of circle curvature
        self.report[report_section]['curve_angmax_deg'] = ('float', curve_right/np.pi*180)  # maximum angle of circle curvatre

        '''2. Create Cartesian matrix'''
        # Define vector containing all angle values between left limiting angle value and right limiting angle value
        angles = np.linspace(curve_left, curve_right, x1 - x0)

        # Define vector containing all radii values between circle radius and maximum radius value
        radii = np.linspace(R, max_R, int(0.5 + max_R - R))

        # Create Cartesian matrix from coordinate vectors (angles & radii)
        an, ra = np.meshgrid(angles, radii)

        '''3. Transform image to linear by interpolation'''
        # Define x coordinates and y coordinates at which input (im_segment) is evaluated.
        xi = x_center + ra * np.sin(an)  # Transform x values using x center as offset and meshgrid
        yi = y_center + ra * np.cos(an)  # Transform y values using y center as offset and meshgrid

        # Create array of coordinates to find for each point in the output, the corresponding coordinates in the input
        coordinates = np.array([yi, xi])

        # Map the input array to new coordinates by interpolation.
        image_transform = scind.map_coordinates(im_segment, coordinates)

        return image_transform


    def depth_of_penetration(self, data):
        """
        Function to analyse the depth of penetration of the reverberation pattern

        1. Average in horizontal direction to create vertical profile
            1b) Normalize the vector


        2. Retrieve the reverberation line positions
            2a) Smooth the obtained vector with box convolution
            2b) Only select the first 5 lines
            2c) Calculate the depth till the 5th line

        3. Create image with reverberation pattern and plotted vertical profile
        """

        report_section = 'DOP'
        self.report[report_section] = {}

        '''1. Average in horizontal direction to create vertical profile'''
        vertical_profile = np.average(data, axis=1)
        vertical_profile = helper.normalize(vertical_profile)

        '''2. Retrieve the reverberation line positions from the detrended vertical profile '''
        # Smoothen the profile to remove small peaks
        window_size = 5
        vertical_profile_smooth = helper.smooth(vertical_profile, window_size)

        # Find the peaks using the scipy.signal function 'find_peaks'
        peaks = find_peaks(vertical_profile_smooth)
        peaks = peaks[0]

        # Select only first 5 reverberation lines and determine depth of 5th reverberation line in pixels
        peaks = peaks[0:5]
        depth_px = peaks[-1]

        # # Plot the vertical profile and the detected peaks
        # x_values = np.array(list(range(len(vertical_profile_smooth))))
        # plt.plot(x_values, vertical_profile_smooth)
        # plt.scatter(peaks, vertical_profile_smooth[peaks])
        # plt.show()

        # Save and report the values
        self.peaks_idx = peaks
        self.vert_profile_smooth = vertical_profile_smooth
        self.vert_profile = vertical_profile

        self.report[report_section]['peaks_idx'] = peaks
        self.report[report_section]['peak_depth_px'] = depth_px

    def isolate_reverberation(self, edges, data, depth_cor):
        x0, y0, x1, y1 = edges
        image_us = data
        peaks = self.peaks_idx

        # Determine depth of reverberation pattern to isolate
        depth_pattern = peaks[-1] + depth_cor

        # Crop the image to the depth of the pattern
        image_pattern = image_us[:depth_pattern, :]

        if self.is_curved == False:
            # Save edges of transformed image
            self.params['reverb_depth'] = x0, y0, x1, y0+depth_pattern

        #@TODO calculate depth of reverberation lines back to curved image

        return image_pattern

    def uniformity(self, data):
        """
        Function to analyse the uniformity of the reverberation pattern

        1. Average in vertical direction to create horizontal profile
            1b) Normalize the vector
        2. Define threshold for weak and dead elements, and calculate line profiles for weak, dead and mean
        3. Count the number of pixel with response below the weak threshold, and the dead threshold.
            a) Separate analysis for all, the outer 10%, outer 10%-30%, and 30%-70%
            b) Also report the relative (to the overall mean) mean, min, max of all region
        4. Create image with reverberation pattern and plotted profile
        5. Extract uniformity parameters: u_cov, u_skew & u_low
        """

        report_section = "uniformity"
        self.report[report_section] = {}

        ''' 1. Average in vertical direction to create horizontal profile.'''
        horizontal_profile = np.average(data, axis=0)
        horizontal_profile = helper.normalize(horizontal_profile)

        '''2. Define threshold for weak and dead elements, and calculate line profiles for weak, dead and mean'''
        # Define threshold for weak and dead element values
        weak = self.params['f_weak']
        dead = self.params['f_dead']

        # Create line profiles for mean, weak and dead
        mean = np.average(horizontal_profile)
        weak = weak * mean
        dead = dead * mean

        '''3. Count the number of pixels with response below the weak threshold, and the dead threshold.'''
        # Find indices of pixels below the weak and dead thresholds
        weak_idx = sorted(np.where(horizontal_profile < weak)[0])
        dead_idx = sorted(np.where(horizontal_profile < dead)[0])

        # Calculate total number of elements and determine index from each 'element;
        elements = len(horizontal_profile)
        indices = list(range(elements))

        '''3a) Separate analysis for all, the outer 10%, outer 10%-30%, and 30%-70%'''
        pixels10 = max(0, min(int(elements * .1 + .5), elements - 1))
        pixels30 = max(0, min(int(elements * .3 + .5), elements - 1))

        # Create buckets to be analysed:
        buckets = [
            ('all', indices),                                                           # All indices
            ('outer10', indices[:pixels10] + indices[-pixels10:]),                      # Outer 20 percent of indices
            ('outer10_30', indices[pixels10:pixels30] + indices[-pixels30:-pixels10]),  # Intermediate 20 percent of indices
            ('inner10_90', indices[pixels10:-pixels10]),                                # Inner 80 percent of indices
            ('inner30_70', indices[pixels30:-pixels30]),                                # Inner 40 percent of indices
        ]

        # Analyse number of weak/dead elements and number of neighbours per bucket
        for lab, idx_bucket in buckets:
            # Calculate number of weak/dead elements in the bucket and number of neighboring elements
            weak_num, weak_neighbors = helper.count_elements_neighboring([i for i in weak_idx if i in idx_bucket])
            dead_num, dead_neighbors = helper.count_elements_neighboring([i for i in dead_idx if i in idx_bucket])

            # Report the calculated values
            self.report[report_section]['weak_{}'.format(lab)] = ('int', weak_num)
            self.report[report_section]['weakneighbors_{}'.format(lab)] = ('int', weak_neighbors)
            self.report[report_section]['dead_{}'.format(lab)] = ('int', dead_num)
            self.report[report_section]['deadneighbors_{}'.format(lab)] = ('int', dead_neighbors)

            '''b) Also report the relative (to the overall mean) mean, min, max of all region'''
            bucket_profile = horizontal_profile[idx_bucket]

            if not lab == 'all':
                self.report[report_section]['relmean_{}'.format(lab)] = ('float', np.average(bucket_profile) / mean)
                self.report[report_section]['relmin_{}'.format(lab)] = ('float', np.min(bucket_profile) / mean)
                self.report[report_section]['relmax_{}'.format(lab)] = ('float', np.max(bucket_profile) / mean)

        # Report other results
        self.report[report_section]['mean'] = ('float', mean)
        self.report[report_section]['relmin'] = ('float', np.min(horizontal_profile) / mean)
        self.report[report_section]['relmax'] = ('float', np.max(horizontal_profile) / mean)
        self.report[report_section]['f_weak'] = ('float', self.params['f_weak'])
        self.report[report_section]['f_dead'] = ('float', self.params['f_dead'])

        self.buckets = buckets
        self.hori_profile = horizontal_profile
        self.mean = mean
        self.weak = weak
        self.dead = dead
        self.pixels10 = pixels10
        self.pixels30 = pixels30

        # # Plot the vertical profile and the detected peaks
        # x_values = np.array(list(range(len(horizontal_profile))))
        # plt.plot(x_values, horizontal_profile)
        # plt.show()

    def run(self):

        '''Step 1: Isolate the ultrasound data as mask'''
        self.us_mask = self.isolate_us_image()

        '''Step 2: Perform depth of penetration analysis'''
        self.analysis_type = 'p'

        # Step 2a: Define the edges of the analysed ultrasound image and create masked image
        edges = self.params['us_x0y0x1y1']
        mask = self.us_mask
        im_segment = self.im * mask

        # Step 2b: check if image is curved, and if so transform image
        self.is_curved = self.check_if_curved(mask, edges)
        if self.is_curved == True:
            image_p = self.curve_transform(im_segment, edges, mask)

        # Step 2c: crop image to a box only containing the reverberation pattern if data is not curved
        if self.is_curved == False:
            image_p = helper.crop_image(im_segment, edges, mask)

        # Create and save image of cropped rectangular image
        im_save = Image.fromarray(image_p)
        im_save.save(f"{self.path_save_images}/{self.label}image_transform.png")

        # Step 2d: Take only middle 33% of the image for the sensitivity analysis
        image_center_crop = helper.crop_center_image(image_p)

        # Create and save image of center cropped image
        im_save = Image.fromarray(image_center_crop)
        im_save.save(f"{self.path_save_images}/{self.label}image_cropped_center.png")

        # Step 2e: analyse preprocessed data
        self.depth_of_penetration(image_center_crop)

        # Step 2f: Create visualization of depth of penetration
        visualize.penetration_visualization(self, image_p)

        '''Step 3: Perform uniformity analysis'''
        self.analysis_type = 'u'

        # Step 3a: Isolate reverberation pattern by taking five reverberation lines + 10 pixels; TO DO extract minima
        depth_cor = 10 #pixels
        image_u = self.isolate_reverberation(edges, image_p, depth_cor)

        # Step 3b: analyse preprocessed data
        self.uniformity(image_u)

        # Step 3c: create visualization of uniformity
        visualize.uniformity_visualization(self, image_u)

        '''Step 4: Create final report of the analysis'''

