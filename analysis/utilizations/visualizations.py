import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
import matplotlib as mpl

class visualizations:
    def penetration_visualization(self, qc, im):

        # Set up parameters
        profile = qc.vert_profile_smooth
        error = qc.error_ver
        depth_px = np.array(list(range(len(profile))))
        peaks_idx = qc.peaks_idx
        num_reverb_lines = qc.params['num_reverb_lines']
        im = np.transpose(im, (1,0))

        ymax = np.max(profile) + 0.05
        xmax = np.max(depth_px)

        # Set up figure
        fig = plt.figure(dpi=1200)
        fig.suptitle(qc.label + 'DOP', fontsize=10)

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
                 zorder=1)
        ax1.fill_between(depth_px, profile+error, profile-error,
                         color='gray',
                         alpha=0.2)
        if peaks_idx[0] == 0 and peaks_idx.shape[0] == 1:
            ax1.plot([], [], ' ', label="No reverberation lines were found")

        else:
            ax1.scatter(peaks_idx[0:num_reverb_lines-1], profile[peaks_idx[0:num_reverb_lines-1]],
                        marker='o',
                        color='xkcd:pale red',
                        label="Reverberation lines",
                        zorder=-1)
            ax1.scatter(peaks_idx[-1], profile[peaks_idx[-1]],
                        marker='x',
                        color='xkcd:red orange',
                        label= 'Last reverberation line',
                        zorder=-1)

            # Plot peaks as line to indicate location in ultrasound image
            ax1.vlines(x=peaks_idx, ymin=profile[peaks_idx], ymax=ymax,
                       color='xkcd:slate blue',
                       zorder=-1)

            ax1.plot([], [], ' ', label="Depth = %.2f px" % peaks_idx[-1])

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
        plt.close()

    def uniformity_visualization(self, qc, im):
        # Set up parameters
        profile = qc.hori_profile
        width_px = np.array(list(range(len(profile))))

        mean = qc.mean
        weak = qc.weak
        dead = qc.dead

        pixels10 = qc.pixels10
        pixels30 = qc.pixels30

        xmax = np.max(width_px)
        ymax = np.max(profile)

        # Set up figure
        fig = plt.figure(dpi=1200)
        fig.suptitle(qc.label + 'uniformity', fontsize=10)

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
        ax1.axhspan(weak, ymax,
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

        ax1.vlines(x=width_px[dead_idx], ymin=profile[dead_idx], ymax=ymax,
                   color='xkcd:pale red',
                   zorder=-1,
                   alpha=0.6,
                   linewidth=2)

        ax1.scatter(width_px[weak_idx], profile[weak_idx],
                    marker='o',
                    color='xkcd:pumpkin',
                    label='weak elements',
                    s=10)

        ax1.vlines(x=width_px[weak_idx], ymin=profile[weak_idx], ymax=ymax,
                   color='xkcd:pale red',
                   zorder=-1,
                   alpha=0.2,
                   linewidth=0.4)

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
        ax1.set_ylim(bottom=0, top=2)
        ax1.set_xlabel("position [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(False)
        ax1.legend(loc='lower right')

        # Offset for xlimit to see the first and last lines
        xoffset = 5
        ax1.set_xlim(left=-xoffset, right=xmax + xoffset)
        ax0.set_xlim(left=-xoffset, right=xmax + xoffset)

        # Save image
        fig.savefig(f"{qc.path_save_images}/{qc.label}uniformity1.png", bbox_inches='tight')

        plt.close()

    def overview_plot(self, qc, im):
        # Set up figure
        fig = plt.figure(dpi=1200)
        fig.set_figwidth(10)
        fig.set_figheight(10)
        fig.suptitle(qc.label + 'Report', fontsize=16)

        # Set up axis;
        ax0 = plt.axes([0.10, 0.55, 0.7, 0.4])  # image
        ax1 = plt.axes([0.10, 0.1, 0.7, 0.45])  # horizontal profile
        ax2 = plt.axes([0.8, 0.55, 0.2, 0.4])  # vertical profile

        # ax0 = plt.axes([0.10, 0.65, 0.7, 0.3])  # image
        # ax1 = plt.axes([0.10, 0.1, 0.7, 0.55])  # horizontal profile
        # ax2 = plt.axes([0.8, 0.65, 0.2, 0.3])  # vertical profile

        '''1. Plot horizontal profile'''
        # Set up parameters
        hori_profile = qc.hori_profile
        width_px = np.array(list(range(len(hori_profile))))

        mean = qc.mean
        weak = qc.weak
        dead = qc.dead

        pixels10 = qc.pixels10
        pixels30 = qc.pixels30

        xmax_hor = np.max(width_px)
        ymax_hor = np.max(hori_profile)

        # Plot the weak and dead areas buckets
        ax1.axhspan(0, dead,
                    facecolor='xkcd:pale red', alpha=0.2)  # Dead area
        ax1.axhspan(dead, weak,
                    facecolor='xkcd:pumpkin', alpha=0.2)  # Weak area
        ax1.axhspan(weak, ymax_hor,
                    facecolor='xkcd:dull green', alpha=0.2)  # remaining area

        # Extract weak and dead indices
        weak_idx = sorted(np.where((hori_profile < weak) & (hori_profile > dead))[0])
        dead_idx = sorted(np.where(hori_profile < dead)[0])

        # Plot the elements
        ax1.scatter(width_px[dead_idx], hori_profile[dead_idx],
                    marker='x',
                    color='xkcd:pale red',
                    label='dead elements',
                    s=10)
        ax1.vlines(x=width_px[dead_idx], ymin=hori_profile[dead_idx], ymax=ymax_hor,
                   color='xkcd:pale red',
                   zorder=-1,
                   alpha=0.6,
                   linewidth=2)

        ax1.scatter(width_px[weak_idx], hori_profile[weak_idx],
                    marker='o',
                    color='xkcd:pumpkin',
                    label='weak elements',
                    s=10)

        ax1.vlines(x=width_px[weak_idx], ymin=hori_profile[weak_idx], ymax=ymax_hor,
                   color='xkcd:pale red',
                   zorder=-1,
                   alpha=0.2,
                   linewidth=0.4)

        # Show profile
        ax1.plot(width_px, hori_profile,
                 color='xkcd:slate blue',
                 label="Horizontal intensity profile",
                 zorder=-1)

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
        ax1.set_ylim(bottom=0, top=ymax_hor)
        ax1.set_xlabel("width [px]")
        ax1.set_ylabel("signal [a.u.]")
        ax1.grid(False)
        ax1.legend(loc='lower right', fontsize='small')

        # Offset for xlimit to see the first and last lines
        xoffset = 5
        ax1.set_xlim(left=-xoffset, right=xmax_hor + xoffset)

        '''2. Plot vertical profile'''
        # Set up parameters
        x0, y0, x1, y1 = qc.params['reverb_depth']
        verti_profile = qc.vert_profile_smooth
        depth_px = np.array(list(range(len(verti_profile))))
        peaks_idx = qc.peaks_idx
        num_reverb_lines = qc.params['num_reverb_lines']

        ymax_ver = np.max(verti_profile) + 0.05
        xmax_ver = y1-y0

        # Show profile below
        ax2.plot(verti_profile, depth_px,
                 color='xkcd:slate blue',
                 label="Vertical intensity profile",
                 zorder=1)
        if peaks_idx[0] == 0 and peaks_idx.shape[0] == 1:
            ax2.plot([], [], ' ', label="No reverberation lines were found")

        else:
            ax2.scatter(verti_profile[peaks_idx[0:num_reverb_lines-1]], peaks_idx[0:num_reverb_lines-1],
                        marker='o',
                        color='xkcd:pale red',
                        label="Reverberation lines",
                        zorder=-1)
            ax2.scatter(verti_profile[peaks_idx[-1]], peaks_idx[-1],
                        marker='x',
                        color='xkcd:red orange',
                        label="Last reverberation line",
                        zorder=-1)

            # Plot peaks as line to indicate location in ultrasound image
            ax2.hlines(y=peaks_idx, xmin=0, xmax=verti_profile[peaks_idx],
                       color='xkcd:slate blue',
                       zorder=-1)

            ax2.plot([], [], ' ', label="Depth = %.2f px" % peaks_idx[-1])

        # Set the x and y labels
        ax2.set_xlabel("signal [a.u.]")
        ax2.yaxis.tick_right()
        ax2.grid(False)
        ax2.legend(loc='lower center')

        # Set limits for x and yaxis
        ax2.set_ylim(bottom=xmax_ver, top=0)
        ax2.set_xlim(left=0, right=ymax_ver)

        ''' 3. Plot image '''
        ax0.imshow(im, cmap='gray', aspect='auto')
        ax0.grid(False)
        ax0.axis('off')

        # Make the tick labels invisible of the image
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)

        # Offset for xlimit to see the first and last lines
        ax0.set_xlim(left=-xoffset, right=xmax_hor + xoffset)

        '''Plot ultrasound parameters, if available'''
        if qc.params['box_param_x0y0x1y1'] is not None:
            ax3 = plt.axes([0.85, 0.2, 0.1, 0.2])  # vertical profile
            x0, y0,x1,y1 = qc.params['box_param_x0y0x1y1']
            ax3.imshow(qc.im[y0:y1,x0:x1], cmap='gray', aspect='auto')
            ax3.grid(False)
            ax3.axis('off')

            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)

        fig.savefig(f"{qc.path_save_overview}/{qc.label}overview.png", bbox_inches='tight')
        plt.close()

    def draw_ROI(self, qc, image):

        # Set up palette
        pal = np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] * \
              np.ones((3,), dtype=np.uint8)[np.newaxis, :]
        # but reserve the first for red for markings
        pal[0] = [255, 0, 0]

        # Create 8bit image for drawing
        temp = np.array(image)
        temp[image == 0] = 1  # Set lowest value to 1
        im = Image.fromarray(temp, mode='L')
        im.putpalette(pal)

        # Add box around reverberation pattern
        rectrois = []
        polyrois = []

        if qc.is_curved == False:
            x0, y0, x1, y1 = qc.params['reverb_depth']
            rectrois.append([(x0, y0), (x1, y1)])

            for r in rectrois:
                draw = ImageDraw.Draw(im)
                (x0, y0), (x1, y1) = r
                draw.rectangle(((x0, y0), (x1, y1)), outline=0)

        if qc.is_curved == True:
            curve_roi = []

            x0, y0, x1, y1 = qc.params['us_x0y0x1y1']
            ang0, ang1 = qc.params['pt_curve_angles_deg']
            r0, r1 = qc.params['pt_curve_radii_px']
            xc, yc, rc = qc.params['pt_curve_origin_px']  # [xc,yc,Rc]

            for ang in np.linspace(ang0, ang1, num=x1 - x0, endpoint=True):
                x = xc + r0 * np.sin(np.pi / 180. * ang)
                y = yc + r0 * np.cos(np.pi / 180. * ang)
                curve_roi.append((x, y))
            for ang in np.linspace(ang1, ang0, num=x1 - x0, endpoint=True):
                x = xc + r1 * np.sin(np.pi / 180. * ang)
                y = yc + r1 * np.cos(np.pi / 180. * ang)
                curve_roi.append((x, y))

            polyrois.append(curve_roi)

            draw = ImageDraw.Draw(im)
            for r in polyrois:
                roi = []
                for x, y in r:
                    roi.append((int(x + .5), int(y + .5)))
                draw.polygon(roi, outline=0)

        im.save(f"{qc.path_save_images}/{qc.label}_ROI.png")









