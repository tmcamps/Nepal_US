import os
from get_project_root import root_path
import yaml
import pydicom
import numpy as np
from scipy import optimize
import scipy.ndimage as scind
import operator
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw


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
        path_file = os.path.join(analysis_root, file)

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


class fit_circle:
    def __init__(self, cf):
        self.x = [xy[0] for xy in cf]  # Determine x-values of circle
        self.y = [xy[1] for xy in cf]  # Determine y-values of circle
        self.residu = -1.

    def calc_radius(self, center_estimate):
        """ calculate the distance (=radius) of each 2D points from the center (xc, yc) """
        radii = np.sqrt((self.x - center_estimate[0]) ** 2 + (self.y - center_estimate[1]) ** 2)

        return radii

    def d_mean(self, center_estimate):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        # Calculate the distance of each 2D points from the center
        dist_center = self.calc_radius(center_estimate)

        return dist_center - dist_center.mean()

    def circ_fit(self):
        center_estimate = np.mean(self.x), np.mean(self.y)    # Estimate center of circle

        # Optimize center estimation
        center_optimize, ier = optimize.leastsq(self.d_mean, center_estimate)

        # Calculate distance of each 2D points from the optimized centre
        dist_cent = self.calc_radius(center_optimize)
        radius = dist_cent.mean()  # Determine radius of the circle

        # Calculate new optimized residue
        self.residue = np.sum((dist_cent - radius) ** 2)

        return center_optimize, radius

class analysis_helper:
    #def __init__(self):

    def remove_small_cluster(self, cca, labels, BW, qc):
        # Determine size of image
        h, w = np.shape(cca)

        # Determine size of minimum size of cluster using parameter clusters
        min_size = h * w * qc.params['cluster_fminsize']

        # Calculate the clustersizes present in the image
        cluster_size = scind.sum(BW, cca, range(labels + 1))

        # Check if minimal cluster size is correct, if too large, divide by 10
        while sum(cluster_size > min_size) < 2 and min_size > 100:
            min_size = int(min_size / 10)

        # Remove small clusters
        mask_size = cluster_size < min_size  # Determine the small clusters
        cca[mask_size[cca]] = 0  # Set small cluster to 0
        labels_uni = np.unique(cca)
        labels = len(labels_uni)
        cca_new = np.searchsorted(labels_uni, cca)  # Create new components array without small clusters

        # Define new components and labels with the small components removed
        return cca_new, labels

    def select_vertical_middle(self, cca):
        # Determine size of image
        h, w = np.shape(cca)

        # Search for clusters present in vertical middle area of image
        search_middle = cca[:, int(0.4 * w):int(0.6 * w)]

        labels_middle = []
        for ss in search_middle.ravel():
            if ss > 0 and not ss in labels_middle:  # Find label of components present in vertical middle
                labels_middle.append(ss)  # Append to label

        return labels_middle

    def crop_image(self, im_segment, edges, mask):
        """
        Function to crop the image to a box containing only data
        """

        x0, y0, x1, y1 = edges

        # # Define pixels to skip for left and right (hcor_px) and above and below (vcor_px)
        # dx = self.params['hcor_px']
        # dy = self.params['vcor_px']
        dx = 0
        dy = 0

        # Crop the image to the edges of the reverberation pattern
        crop_image = im_segment[y0 + dy:y1 - dy + 1, x0 + dx:x1 - dx + 1]

        return crop_image

    def crop_center_image(self, data):
        """
        Function to take only middle 33% for depth of penetration analysis
        """
        h, w = data.shape

        pixels_middle = max(0, min(int(w * .3 + .5), w - 1))
        im_sens = data[:, pixels_middle:-pixels_middle]

        return im_sens

    def check_if_curved(self, reverberation_mask, type = 'reverberation'):
        """
        1. Fit circle through top of reverberation image
        2. Determine if transducer is curved
        """
        report_section = 'curved'
        self.report[report_section] = {}

        ''' 1. Fit circle through top of reverberation image '''
        # Find outer edges of the image
        if type == 'reverberation':
            x0, y0, x1, y1 = self.params['reverb_x0y0x1y1']
        else:
            x0, y0, x1, y1 = self.params['trans_x0y0x1y1']

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
                if reverberation_mask[y, x]:
                    xl = x
                    if xl > -1:
                        circL_xy.append((xl, y))
                    break
            for x in range(center_x, x1):  # find right point
                if reverberation_mask[y, x]:
                    xr = x
                    if xr > -1:
                        circR_xy.append((xr, y))
                    break

            if xr - xl < 10:            # Stop when
                break

        # Create complete circle to fit image
        circ_xy = []
        circ_xy.extend(circL_xy)
        circ_xy.extend(circR_xy)
        circ_xy.sort(key=operator.itemgetter(1))

        ''' 2.  Determine if transducer is curved '''
        if len(circ_xy) < 11:
            # at least 10 point, else probably not a curved probe
            is_curved = False

        else:
            is_curved = True

        return is_curved

    def normalize(self, data):
        """
        Function to normalize data
        """
        return (data - data.min())/ (data.max() - data.min())

    def smooth(self, data, window_size):
        window = np.ones(window_size) / window_size
        data_smooth = np.convolve(data, window, mode='same')

        return data_smooth

    def convert_image_palette(self, data):
        # make a palette, mapping intensities to greyscale
        pal = np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] * \
              np.ones((3,), dtype=np.uint8)[np.newaxis, :]
        # but reserve the first for red for markings
        pal[0] = [255, 0, 0]

        temp = np.array(data)
        temp[data == 0] = 1   # Set lowest value to 1
        im = Image.fromarray(temp, mode='L')
        im.putpalette(pal)

        return im

    def count_elements_neighboring(self, indices):
        """
        count number of elements in list and number of neighboring elements
        """

        neighbors = 0
        if len(indices) > 0:
            indices = sorted(indices)
            for i in range(len(indices) - 1):
                if indices[i + 1] == indices[i] + 1:
                    neighbors += 1

        return len(indices), neighbors


class visualizations:
    def penetration_visualization(self, qc, im):

        # Set up parameters
        profile = qc.vert_profile_smooth
        depth_px = np.array(list(range(len(profile))))
        peaks_idx = qc.peaks_idx
        im = np.transpose(im, (1,0))

        ymax = np.max(profile) + 0.05
        xmax = np.max(depth_px)

        # Set up figure
        fig = plt.figure(dpi=1200)

        # Set up axis; ax0 for image and ax1 for the horizontal profile
        ax0 = plt.axes([0.10, 0.75, 0.85, 0.20])
        ax1 = plt.axes([0.10, 0.10, 0.85, 0.65])

        # Show uniformity image on top and fill whole image
        ax0.imshow(im, cmap='gray', aspect='auto')
        ax0.grid(False)
        ax0.axis('off')

        # Make the tick labels invisible of the image
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)

        # Show profile below
        ax1.plot(depth_px, profile,
                 color='xkcd:slate blue',
                 label="Vertical intensity profile",
                 zorder=-1)
        ax1.scatter(peaks_idx[0:4], profile[peaks_idx[0:4]],
                    marker='o',
                    color='xkcd:pale red',
                    label="Reverberation lines")
        ax1.scatter(peaks_idx[-1], profile[peaks_idx[-1]],
                    marker='x',
                    color='xkcd:red orange',
                    label="Fifth reverberation line")

        ax1.plot([], [], ' ', label="Depth = %.2f px" %peaks_idx[-1])

        # Set the x and y labels
        ax1.set_xlabel("position [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(False)
        ax1.legend()

        # Set limits for x and yaxis
        ax0.set_xlim(left=0, right=xmax)
        ax1.set_ylim(bottom=0, top=ymax)
        ax1.set_xlim(left=0, right=xmax)

        # Save image
        fig.savefig(f"{qc.path_save_images}/{qc.label}DOP.png", bbox_inches='tight')

    def uniformity_visualization(self, qc, im):
        # Set up parameters
        profile = qc.hori_profile
        width_px = np.array(list(range(len(profile))))

        mean = qc.mean
        weak = qc.weak
        dead = qc.dead

        buckets = qc.buckets
        pixels10 = qc.pixels10
        pixels30 = qc.pixels30

        ymax = np.max(profile) + 0.05
        xmax = np.max(width_px)

        # Set up figure
        fig = plt.figure(dpi=1200)

        # Set up axis; ax0 for image and ax1 for the horizontal profile
        ax0 = plt.axes([0.10, 0.75, 0.85, 0.20])
        ax1 = plt.axes([0.10, 0.10, 0.85, 0.65])

        # Show uniformity image on top and fill whole image
        ax0.imshow(im, cmap='gray', aspect='auto')
        ax0.grid(False)
        ax0.axis('off')

        # Plot the weak and dead areas buckets
        ax1.axhspan(0, dead,
                   facecolor='xkcd:pale red', alpha=0.2)  # Dead area
        ax1.axhspan(dead, weak,
                   facecolor='xkcd:pumpkin', alpha=0.2)  # Weak area
        ax1.axhspan(weak, 1,
                    facecolor='xkcd:dull green', alpha=0.2)  # remaining area

        # Extract weak and dead indices
        weak_idx = sorted(np.where((profile < weak)&(profile>dead))[0])
        dead_idx = sorted(np.where(profile < dead)[0])

        # Plot the elemenets
        ax1.scatter(width_px[dead_idx], profile[dead_idx],
                    marker='x',
                    color='xkcd:pale red',
                    label='dead elements',
                    s=10)
        ax1.scatter(width_px[weak_idx], profile[weak_idx],
                    marker='o',
                    color='xkcd:pumpkin',
                    label='weak elements',
                    s=10)

        # Show profile
        ax1.plot(width_px, profile,
                 color='xkcd:slate blue',
                 label="Horizontal intensity profile",
                 zorder=-1)

        # Make the tick labels invisible of the image
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)

        # Plot the weak, dead and mean limits and plot the buckets
        ax1.plot([width_px[0], width_px[-1]], [mean, mean],
                linestyle=':',
                linewidth=1,
                color='xkcd:dull green')

        # Plot the outer buckets
        ax1.axvspan(width_px[0], width_px[pixels10],
                   facecolor='black', alpha=0.2)  # Outer left 10 percent
        ax1.axvspan(width_px[-1], width_px[-pixels10],
                   facecolor='black', alpha=0.2)  # Outer right 10 percent
        ax1.axvspan(width_px[pixels10], width_px[pixels30],
                   facecolor='black', alpha=0.1)  # Outer left 10-30 percent
        ax1.axvspan(width_px[-pixels30], width_px[-pixels10],
                   facecolor='black', alpha=0.1)  # Outer right 10-30 percent

        # Set x and y labels
        ax1.set_ylim(bottom=0, top=ymax)
        ax1.set_xlabel("position [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(False)
        ax1.legend(loc='upper right')

        # Offset for xlimit to see the first and last lines
        xoffset = 5
        ax1.set_xlim(left=-xoffset, right=xmax + xoffset)
        ax0.set_xlim(left=-xoffset, right=xmax + xoffset)

        # Save image
        fig.savefig(f"{qc.path_save_images}/{qc.label}uniformity1.png", bbox_inches='tight')



   # def full_overview(self,qc):
# rectrois = []
# polyrois = []
# if self.is_curved == True:
#     curve_roi = []
#     ang0, ang1 = self.params['pt_curve_angles_deg']
#     r0, r1 = self.params['pt_curve_radii_px']
#     xc, yc, rc = self.params['pt_curve_origin_px']  # [xc,yc,Rc]
#
#     for ang in np.linspace(ang0, ang1, num=x1 - x0, endpoint=True):
#         x = xc + r0 * np.sin(np.pi / 180. * ang)
#         y = yc + r0 * np.cos(np.pi / 180. * ang)
#         curve_roi.append((x, y))
#     for ang in np.linspace(ang1, ang0, num=x1 - x0, endpoint=True):
#         x = xc + r1 * np.sin(np.pi / 180. * ang)
#         y = yc + r1 * np.cos(np.pi / 180. * ang)
#         curve_roi.append((x, y))
#     polyrois.append(curve_roi)

