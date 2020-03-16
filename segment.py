# imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

# global variables
background = None

# function used to find the average image in the background
def find_average(current_frame, weight):
    global background
    #initial function call
    if background is None:
        background = current_frame.copy().astype("float")
        return

    # compute a weighted average and update the 'background' image
    # changes in the weight variable result in different levels of responsiveness
    # to changes in the image, i.e. the background variable changes more with higher weight
    cv2.accumulateWeighted(current_frame, background, weight)

# function used to segment the region of the hand in the image from the background
def segment(current_frame, threshold=25):
    global background

    # calculates the absolute difference between the current frame and the background
    diff = cv2.absdiff(background.astype("uint8"), current_frame)

    # compares diff with the threshold, if it is above the threshold, it is set to 1
    # if it is below the threshold, it is set to 0
    diff_binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # gets the contours of the thresholded image
    #(_, contours, _) = cv2.findContours(diff_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(diff_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # return None if no contours were detected
        return
    else:
        # get the maximum size contour which is the hand
        largest_contour = max(contours, key=cv2.contourArea)
        return (diff_binary, largest_contour)

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
    circular_roi = np.zeros(diff_binary.shape[:2], dtype="uint8")

    # get the contours inside the circle
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    finger_count = 0

    for contour in contours:
        # get the dimensions of a box around the contour
        (x, y, width, height) = cv2.boundingRect(contour)

        # increment the count of fingers if the contour region is not the wrist and
        # the number of points along the contour does not exceed 25% of the circular region
        if((centreY + (centreY * 0.25)) > (y + height) and ((circumference * 0.25) > c.shape[0])):
            finger_count += 1

    return finger_count

# Main function
if __name__ == "__main__":
    # initialise weight used in calculating running average
    weight = 0.5

    # get reference to the webcam
    camera = cv2.VideoCapture(0)

    # set co-ordinates for the region of interest in the frame
    # i.e. where the hand make the gesture
    top, right, bottom, left = 10, 350, 225, 590

    # number of frames captured
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (_, current_frame) = camera.read()

        # resize the frame
        current_frame = imutils.resize(current_frame, width=700)

        # flip the frame so that it is not the mirror view
        current_frame = cv2.flip(current_frame, 1)

        # clone the frame
        clone = current_frame.copy()

        # get the height and width of the frame
        (height, width) = current_frame.shape[:2]

        # get the region of interest
        reg_of_interest = current_frame[top:bottom, right:left]

        # convert reg_of_interest to grayscale and blur it
        gray_frame = cv2.cvtColor(reg_of_interest, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        if num_frames < 30:
            # use first 30 frames to calculate the background
            find_average(gray_frame, weight)
        else:
            # once background has been calculated, we can now segment the hand region
            region = segment(gray_frame)

            if region is not None:
                # if hand region successfully found, unpack it
                (diff_from_background, largest_contour) = region

                # draw the hand on the cloned frame
                cv2.drawContours(clone, [largest_contour + (right, top)], -1, (0, 0, 255))

                # count number of fingers
                num_fingers = count(diff_from_background, largest_contour)

                cv2.putText(clone, str(num_fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # find the convex hull of the segmented hand (Google convex hull images if you don't know what it is)
                convex_hull = cv2.convexHull(largest_contour,returnPoints = False)

                defects = cv2.convexityDefects(largest_contour, convex_hull)

                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(largest_contour[s][0])
                    end = tuple(largest_contour[e][0])
                    far = tuple(largest_contour[f][0])
                    cv2.line(clone,start,end,[0,255,0],2)
                    cv2.circle(clone,far,5,[0,0,255],-1)

                # show the segmented region
                cv2.imshow("Difference from Background", diff_from_background)

        # draw the region to place the hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1

        # show the frame
        cv2.imshow("Video Feed", clone)

        # give user chance to quit the program. Mask used to get last 8 bits
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

    # free up memory when program is finished
    camera.release()
    cv2.destroyAllWindows()
