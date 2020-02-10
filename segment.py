# imports
import cv2
import imutils
import numpy as np

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
