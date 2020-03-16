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

    # find the center of the palm
    centreX = int((extreme_left[0] + extreme_right[0]) / 2)
    centreY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the distances between the centre and the extreme points
    distances = pairwise.euclidean_distances([(centreX, centreY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]

    # get the maximum distance
    max_distance = distances[distances.argmax()]

    # calculate the radius of the hand circle with 80% of the max distance
    radius = int(0.8 * max_distance)

    # get the circumference of the circle
    circumference = (2 * np.pi * radius)

    # get the region of interest which has the palm and fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    # get the contours inside the circle
    (_, contours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    finger_count = 0

    for contour in contours:
        # get the dimensions of a box around the contour
        (x, y, width, height) = cv2.boundingRect(contour)

        # increment the count of fingers if the contour region is not the wrist and
        # the number of points along the contour does not exceed 25% of the circular region
        if((centreY + (centreY * 0.25)) > (y + height) and ((circumference * 0.25) > c.shape[0]):
            finger_count += 1

return finger_count
