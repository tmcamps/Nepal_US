import numpy as np
import scipy.ndimage as scind
import operator
from PIL import Image

class analysis_helper:
    def remove_small_cluster(self, BW):
        BW = BW.astype('int')
        kernel = scind.generate_binary_structure(2,1)

        # Perform closing to fill holes in ultrasound region where signal is low
        BW_closing = scind.binary_closing(BW, structure=kernel).astype(np.int)

        # Select only largest component = ultrasound region; remove small clusters
        cca, labels = scind.label(BW_closing)
        cluster_size = scind.sum(BW_closing, cca, range(labels + 1))
        loc = np.argmax(cluster_size)
        BW_closing = cca == loc

        return BW_closing

    def remove_small_cluster_old(self, cca, labels, BW, qc):
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

    def crop_image(self, im_segment, edges):
        """
        Function to crop the image to a box containing only data
        """

        x0, y0, x1, y1 = edges

        # # Define pixels to skip for left and right (hcor_px) and above and below (vcor_px)
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

        # If defined also crop a few pixels from the upper part of the image
        dy = 0
        im_sens = data[dy:, pixels_middle:-pixels_middle]

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

    def normalize_mean(self, data):
        """
        Function to normalize data by average; add 1 to center around 1
        """
        norm = (data - data.mean()) / (data.max() - data.min())

        return norm + 1

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