import numpy as np
import matplotlib.pyplot as plt

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