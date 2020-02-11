import cv2

# function for counting the number of fingers in a segmented hand
def count(diff_binary, largest_contour):
    # find the convex hull of the segmented hand (Google convex hull images if you don't know what it is)
    convex_hull = cv2.convexHull(largest_contour)

    # find the most extreme points of the convex hull
    extreme_top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    extreme_left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    extreme_right = tuple(convex_hull[convex_hull[: :, 0].argmax()][0])
