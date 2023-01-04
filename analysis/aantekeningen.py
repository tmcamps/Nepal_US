#%%
# '''2. Calculate max. DOP in pixels'''
# # Determine average background value as noise
# background = im[np.round(im.shape[0] / 2).astype(int):, :]
# background_norm = helper.normalize(background)
# background_value = np.median(background_norm.ravel())
# background_line = np.ones(vertical_profile_norm.shape[0]) * background_value
#
# # Find indices where background intersects with the profile
# idx = np.argwhere(np.diff(np.sign(vertical_profile - background_line))).flatten()
#
# # Select index of first intersection, and make sure it is not located in the first part (10 pixels) of the image
# skip_start = idx > 10
# first_idx = next((i for i, j in enumerate(skip_start) if j), None)
#
# # Extract pixel value corresponding to max DOP
# DOP_pixels = idx[first_idx]
# self.DOP_pixels = DOP_pixels
 #%%
'''1. Average in horizontal direction to create vertical profile'''
vertical_profile = np.average(im_crop, axis=1)
vertical_profile_norm = helper.normalize(vertical_profile)

# Determine average background value as noise

background = im_crop[np.round(im_crop.shape[0] / 2).astype(int):, :]
background_norm = helper.normalize(background)
background_value = np.median(background_norm.ravel())
background_line = np.ones(vertical_profile_norm.shape[0]) * background_value

# Find indices where background intersects with the profile
idx = np.argwhere(np.diff(np.sign(vertical_profile_norm - background_line))).flatten()

x = np.array(list(range(len(vertical_profile_norm))))
plt.plot(x, vertical_profile_norm)
plt.scatter(idx, vertical_profile_norm[idx])
plt.show()


# #%%
#
# profile_DOP = vertical_profile_norm[:79]
# detrend = np.zeros(profile_DOP.shape)
# smooth_factor = 3
# profile_smooth = helper.smooth(profile_DOP, smooth_factor)
#
# for t in range(1, profile_DOP.shape[0]):
#     detrend[t] = profile_smooth[t] - profile_smooth[t - 1]
#
# detrend[0] = detrend[1]
# locs = argrelextrema(detrend, np.less)  # find extremas
# locs = locs[0]
#
# positions = locs.copy()
# positions = positions

#%%
vertical_db =  10.0 * np.log10(vertical_profile)

#%%
from scipy.signal import find_peaks
smooth = helper.smooth(vertical_profile_norm,5)
peaks = find_peaks(smooth)
peaks = peaks[0]
peaks = peaks[0:5]

#%%
x = np.array(list(range(len(vertical_profile_norm))))
plt.plot(x, smooth)

plt.scatter(peaks, smooth[peaks])
plt.show()

#%% full image
snr_sd = np.std(im_norm, axis=1)
snr_mean = np.mean(im_norm, axis=1)
im_snr = snr_mean / snr_sd

x_snr = np.array(list(range(len(im))))
plt.plot(x_snr, im_snr)

im_snr_smooth = helper.smooth(im_snr, 3)
plt.plot(x_snr, im_snr_smooth)

plt.show()

#%%
pattern = im_crop
pattern_sd = np.std(pattern, axis = 1)

noise_sd = np.std(im_crop[200:500,:])

SNR = 10.0 * np.log10((pattern_sd**2)/(noise_sd**2))

x_snr = np.array(list(range(len(SNR))))
plt.plot(x_snr, SNR)
plt.show()
#%%
pattern = im_crop[0:100,:]
pattern_sd = np.std(pattern, axis = 1)
pattern_mean = np.mean(pattern, axis = 1)
pattern_snr = pattern_mean / pattern_sd
































def dicom_info(self, info='dicom'):
    # Different from ImageJ version; tags "0008","0104" and "0054","0220"
    #  appear to be part of sequences. This gives problems (cannot be found
    #  or returning whole sequence blocks)
    # Possibly this can be solved by using if(type(value) == type(dicom.sequence.Sequence()))
    #  but I don't see the relevance of these tags anymore, so set them to NO

    import string
    printable = set(string.printable)

    try:
        manufacturer = str(self.readDICOMtag("0008,0070")).lower()
        if "siemens" in manufacturer:
            manufacturer = "Siemens"
        elif "philips" in manufacturer:
            manufacturer = "Philips"
        else:
            print("Manufacturer '{}' not implemented. Treated as 'Philips'.".format(manufacturer))
            manufacturer = "Philips"
    except:
        print("Could not determine manufacturer. Assuming 'Philips'.")

    if manufacturer == "Philips":
        if info == "dicom":
            dicomfields = {
                'string': [
                    ["0008,0012", "Instance Date"],
                    ["0008,0013", "Instance Time"],
                    ["0008,0060", "Modality"],
                    ["0008,0070", "Manufacturer"],
                    ["0008,1090", "Manufacturer Model Name"],
                    ["0008,1010", "Station Name"],
                    ["0008,1030", "Study Description"],
                    ["0008,0068", "Presentation Intent Type"],
                    ["0018,1000", "Device Serial Number"],
                    ["0018,1020", "Software Version(s)"],
                    ["0018,1030", "Protocol Name"],
                    ["0018,5010", "Transducer Data"],
                    ["0018,5020", "Processing Function"],
                    ["0028,2110", "Lossy Image Compression"],
                    ["2050,0020", "Presentation LUT Shape"],
                ],
                'float': [
                    ["0028,0002", "Samples per Pixel"],
                    ["0028,0101", "Bits Stored"],
                    ["0018,6011, 0018,6024", "Physical Units X Direction"],
                    ["0018,6011, 0018,602c", "Physical Delta X"],
                ]  # Philips
            }
        elif info == "id":
            dicomfields = {
                'string': [
                    ["0008,1010", "Station Name"],
                    ["0018,5010", "Transducer"],
                    ["0008,0012", "InstanceDate"],
                    ["0008,0013", "InstanceTime"]
                ]
            }

        elif info == "probe":
            dicomfields = {
                'string': [
                    ["0018,5010", "Transducer"],
                ]
            }

    elif manufacturer == "Siemens":
        if info == "dicom":
            dicomfields = {
                'string': [
                    ["0008,0023", "Image Date"],
                    ["0008,0033", "Image Time"],
                    ["0008,0060", "Modality"],
                    ["0008,0070", "Manufacturer"],
                    ["0008,1090", "Manufacturer Model Name"],
                    ["0008,1010", "Station Name"],
                    ["0018,1000", "Device Serial Number"],
                    ["0018,1020", "Software Version(s)"],
                    ["0018,5010", "Transducer Data"],
                ],
                'float': [
                    ["0028,0002", "Samples per Pixel"],
                    ["0028,0101", "Bits Stored"],
                    ["0018,6011, 0018,6024", "Physical Units X Direction"],
                    ["0018,6011, 0018,602c", "Physical Delta X"],
                    ["0018,5022", "Mechanical Index"],
                    ["0018,5024", "Thermal Index"],
                    ["0018,5026", "Cranial Thermal Index"],
                    ["0018,5027", "Soft Tissue Thermal Index"],
                    ["0019,1003", "FrameRate"],
                    ["0019,1021", "DynamicRange"],
                ]  # Siemens
            }


        elif info == "id":
            dicomfields = {
                'string': [
                    ["0008,1010", "Station Name"],
                    ["0018,5010", "Transducer"],
                    ["0008,0023", "ImageDate"],
                    ["0008,0033", "ImageTime"],
                ]
            }

        elif info == "probe":
            dicomfields = {
                'string': [
                    ["0018,5010", "Transducer"],
                ]
            }

    results = {}
    for dtype in dicomfields.keys():
        if not dtype in results.keys():
            results[dtype] = []
        for df in dicomfields[dtype]:
            key = df[0]
            value = ""
            try:
                value = str(self.readDICOMtag(key)).replace('&', '')
                value = ''.join(list(filter(lambda x: x in printable, value)))
            except:
                value = ""

            if dtype in ['string']:
                results[dtype].append((df[1], value))
            elif dtype in ['int']:
                results[dtype].append((df[1], int(value)))
            elif dtype in ['float']:
                results[dtype].append((df[1], float(value)))

    return results


def imageID(self, probeonly=False):
    """
    find a identifyable suffix
    """
    if not self.label in [None, ""]:
        return self.label

    # make an identifier for this image
    if probeonly:
        di = self.dicom_info(info='probe')
    else:
        di = self.dicom_info(info='id')

    # construct label from tag values
    label = '_'.join(v for k, v in di['string'])

    # sanitize label
    forbidden = '[,]\'" '
    label2 = ''
    for la in label:
        if la in forbidden:
            continue
        else:
            label2 += la
    label2 = label2.replace('UNUSED', '')  # cleaning
    label2.replace('/', '-')

    self.label = label2
    return label2
