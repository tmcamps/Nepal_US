import numpy as np
from scipy import optimize

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