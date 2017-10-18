import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys
from moviepy.editor import VideoFileClip
from collections import deque
import imageio


# find chess board corners
def corr_coef(path, file, nx=9, ny=6):
    fname = path + file
    img = cv2.imread(fname)
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # finds the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return ret, corners


# unwarp perspective of an image
def corners_unwarp(img, src, dest, mtx, dist):
    # Undistorts using mtx and dist
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Converts to grayscale
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # uses cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # Minv - inverted matrix
    Minv = cv2.getPerspectiveTransform(dest, src)
    img_size = (gray.shape[1], gray.shape[0])
    # uses cv2.warpPerspective() to warp the image to a top-down view
    warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# finds calibration parameters for camera
def cal_param(path, nx=9, ny=6):
    # prepares object points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2d points in image plane

    # prepares object points, like (0,0,0),(1,0,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates

    # makes a list of calibration images
    edge_found = 0
    edge_not_found = 0
    for file in os.listdir(path):
        fname = path + file
        img = cv2.imread(fname)
        ret, corners = corr_coef(path, file, nx=9, ny=6)
        # If found, append corners to the lists
        if ret == True:
            edge_found += 1
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            edge_not_found += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calibration parameters for camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


# calculates more or less equal number of rows and cols for plots with multiple sublots
def nrows_ncols(n):
    if (int(np.sqrt(n))) ** 2 == n:
        n_col = n_row = np.sqrt(n)
    else:
        n_col = n // int(np.sqrt(n))
        n_row = n // n_col + n % n_col
    return n_row, n_col


# plots multiple images on the same plt.figure(), camera was not calibrated yet, this function is not neccessary for
# final pipeline
# all images will be ploted in pairs: original images from "files" and images with applied func(*args) on it
def multiple_plot_uncalibrated(path, files, func, title1='Original image', title2='Modified', *args):
    fig = plt.figure()
    n = len(files)  # number of pictures to show
    n_row, n_col = nrows_ncols(
        2 * n)  # multiplies by two, because it should show original image + image with applied func(*args) on it
    for i in range(n):
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        img = mpimg.imread(path + files[i])
        plt.imshow(img)
        a.set_title(title1, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.set_title(title2, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        dst = func(img, *args)
    plt.imshow(dst)
    fig.savefig('output_images/' + title2 + '.jpg')
    return fig


# plots multiple images on the same plt.figure()
# all images will be ploted in pairs: original images from "files" and images with applied func(*args) on it
def multiple_plot(path, files, func, title1='Original image', title2='Modified', *args):
    ret, mtx, dist, rvecs, tvecs = cal_param('camera_cal/')
    fig = plt.figure()
    n = len(files)  # number of pictures to show
    n_row, n_col = nrows_ncols(
        2 * n)  # multiplies by two, because it will show original + image with applied func() on it
    for i in range(n):  # images calibrationXX, where XX = [2,4] are the most distorted from camera images
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        img = mpimg.imread(path + files[i])
        plt.imshow(img)
        a.set_title(title1, fontsize=10)
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.set_title(title2, fontsize=10)
        a.set_xticks([])
        a.set_yticks([])
        img = cv2.undistort(img, mtx, dist, None, mtx)
        dst = func(img, *args)
        a.set_xticks([])
        a.set_yticks([])
        plt.imshow(dst)
        fig.savefig('output_images/' + title2 + '.jpg')
    return fig


# checks on several images how the algoritm for calibrate camera + unwarp works
def undistort(path, files, mtx, dist, title2, nx=9, ny=6):
    fig = plt.figure()
    n = len(files)
    n_row, n_col = nrows_ncols(2 * n)
    for i in range(2):
        ret, corners = corr_coef(path, files[i], nx=9, ny=6)
        img = cv2.imread(path + files[i])
        # defines 4 source points src = np.float32([[,],[,],[,],[,]])
        src = np.float32([corners[0][0], corners[nx - 1][0], corners[nx * ny - 1][0], corners[nx * (ny - 1)][0]])
        # defines 4 destination points dst = np.float32([[,],[,],[,],[,]])
        x = img.shape[1]
        y = img.shape[0]
        dst = np.float32([[x // nx, y // ny], [x // nx * (nx - 1), y // ny], [x // nx * (nx - 1), y // ny * (ny - 1)],
                          [x // nx, y // ny * (ny - 1)]])
        unwarped = corners_unwarp(img, src, dst, mtx, dist)
        a = fig.add_subplot(n_row, n_col, 2 * i + 1)
        a.imshow(img)
        a.set_title('Original Image')
        a = fig.add_subplot(n_row, n_col, 2 * i + 2)
        a.imshow(unwarped)
        a.set_title(title2)
    fig.savefig('output_images/' + title2 + '.jpg')


# applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, mag_thresh=(0, 255)):
    # applies the following steps to img
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # takes the derivative in x or y given orient = 'x' or 'y'
    # takes the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel))
    # scales to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # returns this mask as binary_output image
    return binary_output


# defines a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh_func(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # applies the following steps to img
    # converts to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # takes the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    # calculates the magnitude
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # scales to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)
    # create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(mag)
    binary_output[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
    # return this mask as your binary_output image
    return binary_output


# deletes everything in images besides the retion of interest
def region_of_interest(img, vertices=0):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defines a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on the image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2), vertices=0):
    # applies the following steps to img
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # takes the absolute value of the x and y gradients
    ##abs_x = np.absolute(sobelx)
    ##abs_y = np.absolute(sobely)
    # np.arctan2(abs_sobely, abs_sobelx) calculates the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # creates a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # returns this mask as your binary_output image
    binary_output = region_of_interest(binary_output, vertices)
    return binary_output


# combimes abs_sobel_thresh() with dir_threshold() and masks everything besides the region of interest
def combined_sober(img, sobel_kernel=3, mag_thresh=(0, 255), thresh=(0, np.pi / 2), blur_kernel=0.0, vertices=0):
    # if blur_kernel!=(0,0):
    #     img = cv2.GaussianBlur(img, blur_kernel, 0)
    gradx = abs_sobel_thresh(img, 'x', sobel_kernel, mag_thresh)
    grady = abs_sobel_thresh(img, 'y', sobel_kernel, mag_thresh)
    mag_binary = mag_thresh_func(img, sobel_kernel, mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel, thresh, vertices)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # if blur_kernel!=(0,0):
    #     combined = cv2.GaussianBlur(combined, blur_kernel, 0)
    combined = region_of_interest(combined, vertices)
    return combined


# creates bitmap image from saturation channel of HLS images representation
def saturation_extraction(img, thresh_s=(90, 255), vertices=0):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh_s[0]) & (S <= thresh_s[1])] = 1
    binary = region_of_interest(binary, vertices)

    return binary


# This returns a stack of the two binary images, whose components are sobel transformation of an image and
# saturation channel from HLS with applied thresholds
def stack_sobel_saturation(img, sobel_kernel=3, mag_thresh=(20, 100), s_thresh=(170, 255), output='color'):
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # Threshold x gradient
    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    if output == 'color':
        return color_binary
    else:
        return combined_binary


# transforms image to bird eye view
def transform_image(img, src=np.float32([[200, 680], [590, 451], [690, 451], [1042, 666]]),
                    dest=np.float32([[180, 720], [200, 0], [1000, 0], [1000, 720]])):
    # uses cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dest)
    # uses cv2.warpPerspective() to warp your image to a top-down view
    Minv = cv2.getPerspectiveTransform(dest, src)
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return unwarped


from collections import deque

frame_buffer_left = deque()  # buffer for polynomial coefficients for left lane to avoid outbursts
frame_buffer_right = deque()  # buffer for polynomial coefficients for right lane to avoid outbursts
frame_curvature_left = deque()  # buffer for curvature for left lane to avoid outbursts
frame_curvature_right = deque()  # buffer for curvature for right lane to avoid outbursts


#mask for yellow lines
def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # lower = np.array([20,60,60])
    # upper = np.array([38,174, 250])
    lower = np.array([20, 60, 60])
    upper = np.array([38,174, 250])

    mask = cv2.inRange(hsv, lower, upper)

    return mask


#mask for white lines
def select_white(image):
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)

    return mask


def yellow_white_sobel(img):
    mask_yellow = select_yellow(img)#mask for yellow lines
    mask_white = select_white(img)#mask for white lines
    img_sobel = stack_sobel_saturation(img, output='black')
    mask = np.zeros_like(img)
    mask[(mask_yellow > 0) | (mask_white > 0) | (img_sobel > 0)]=1
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations = 2)
    # if output == 'mask':
    #     return mask
    # else:
    img_masked = img*mask
    return img_masked


# finds lines
def finding_lines(img, sobel_kernel=3, mag_thresh=(35, 100), thresh_s=(170, 255), output=0):
    # output=0 - outputs plt.fig with a plot with found on it lines
    # output=1 - returns polynomial coefficients for single images (not video frames), frame_buffers are
    #  not used during caclulation of coefficients
    # output=2 - outputs polynom's coefficients for static image
    # output=3  function skippes sliding windows). it means we know already where to look
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    global frame_curvature_left  # collections.deque() for several last values of curvature
    global frame_curvature_right  # collections.deque() for several last values of curvature

    img = transform_image(img)
    binary_warped = stack_sobel_saturation(img, sobel_kernel, mag_thresh, thresh_s, 'bw')
    binary_warped=np.zeros_like(binary_warped)
    img_masked = yellow_white_sobel(img)
    binary_warped[(img_masked[:,:,0] > 0) | (img_masked[:,:,1] > 0) | (img_masked[:,:,2] > 0)] = 1

    # takes a histogram of the bottom half of the 'binary_warped' image
    histogram = np.sum(binary_warped[3 * int(binary_warped.shape[0] / 4):, :], axis=0)
    # create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # finds the peak of the left and right halves of the histogram
    # these will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 50
    # sets height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # set the width of the windows +/- margin
    margin = 50
    # set minimum number of pixels found to recenter window
    minpix = 25
    # creates empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # steps through the windows one by one
    for window in range(nwindows):
        # identifies window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # draws the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # appends these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # if you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # concatenates the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # extracts left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # fits a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # generates x and y values for plotting
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if output == 0:  # image with wide polylinomial drawn on lanes
        return out_img
    elif output == 1:  # return polynomial coefficients for single images (not video frames), frame_buffers are
        # not used during caclulation of coefficients
        return left_fit, right_fit
    elif output == 2:  # return polynomial coefficients for single images from videos, frame_buffers are
        # used during caclulation of coefficients
        nframes = 2  # number of frames for frame_buffers to use
        margin = (0.5, 0.35, 4, 40)  ## deviation margin between last frame and last {n}frames
                    #  {curvature_left, curvature_right, \
                    #  {curvature_left/curvature_rigth or curvature_rigth/curvature_left,
                    #  offset between old line and new line(if offest bigger than {40} values from previous
                    #  video frame used)}} \

        curvature = calc_curvature(img)
        if len(frame_buffer_left) == 0:
            for i in range(nframes):
                frame_buffer_left.append(left_fit)
                frame_buffer_right.append(right_fit)
                frame_curvature_left.append(curvature[0])
                frame_curvature_right.append(curvature[1])
        ploty = binary_warped.shape[0]#y of lowest polynomial line on the screen, \
                                      # left_fitx and left_fitx_previous appropriate x values for ploty \
                                      # in current and previous video frame
        left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
        left_fitx_previous = frame_buffer_left[-1][0] * (ploty ** 2) + frame_buffer_left[-1][1] * ploty + frame_buffer_left[-1][2]
        if (1 + margin[0] > curvature[0] / frame_curvature_left[-1] > 1 - margin[0]) and (curvature[0]>1000 or \
                1/margin[2] < curvature[0] / curvature[1] < margin[2]) and abs(left_fitx-left_fitx_previous)<margin[3]:
            frame_buffer_left.popleft()
            frame_buffer_left.append(left_fit)
            frame_curvature_left.popleft()
            frame_curvature_left.append(curvature[0])
        else:
            left_fit = frame_buffer_left[-1]
        right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
        right_fitx_previous = frame_buffer_right[-1][0] * (ploty ** 2) + frame_buffer_right[-1][1] * ploty + frame_buffer_right[-1][2]
        if (1 + margin[1] > curvature[1] / frame_curvature_right[-1] > 1 - margin[1]) and (curvature[1]>1000 or \
                    1/margin[2] < curvature[1] / curvature[0] < margin[2]) and abs(right_fitx-right_fitx_previous)<margin[3]:
            frame_buffer_right.popleft()
            frame_buffer_right.append(right_fit)
            frame_curvature_right.popleft()
            frame_curvature_right.append(curvature[0])

        else:
            right_fit = frame_buffer_right[-1]
        return left_fit, right_fit

    elif output == 3:
        return finding_lines_continue(binary_warped, left_fit, right_fit)

    elif output == 3:
        return finding_lines_continue(binary_warped, left_fit, right_fit)


# function used, if skipped function finding_lines(aka sliding windows). it means we know already where to look
# for line's starting points
def finding_lines_continue(binary_warped, left_fit, right_fit):
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    nframes = 10  # number of frames for frame_buffers to use
    average_left = []  # contains values to compare with coefficients of left_fit
    average_right = []  # contains values to compare with coefficients of right_fit
    margin = 1  # deviation margin between last frame and last {n}frames
    if len(frame_buffer_left) == 0:
        for i in range(nframes):
            frame_buffer_left.append(left_fit)
            frame_buffer_right.append(right_fit)
    else:
        for n in range(3):
            average_left.append(left_fit[n] / sum([i[n] for i in frame_buffer_left]) * nframes)
            average_right.append(left_fit[n] / sum([i[n] for i in frame_buffer_right]) * nframes)
    left = [1 for i in average_left if i < (1 - margin) or i > (1 + margin)]
    right = [1 for i in average_right if i < (1 - margin) or i > (1 + margin)]
    frame_buffer_left.append(left_fit)
    frame_buffer_right.append(right_fit)

    if len(left) == 0:
        frame_buffer_left.popleft()
        frame_buffer_left.append(left_fit)
    else:
        left_fit = frame_buffer_left[-1]

    if len(right) == 0:
        frame_buffer_right.popleft()
        frame_buffer_right.append(right_fit)
    else:
        right_fit = frame_buffer_right[-1]

    # we have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))
    # extracts left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # fits a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # generates x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # creates an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # generates a polygon to illustrate the search window area
    # and recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    # draws the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result


# calculates curvature
def calc_curvature(path, files=0, sobel_kernel=3, mag_thresh=(35, 100), thresh_s=(170, 255)):
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    if files == 0:
        img = path
        left_fit, right_fit = finding_lines(img, sobel_kernel, mag_thresh, thresh_s, output=1)
        # left_fit, right_fit = frame_buffer_left[-1], frame_buffer_right[-1]
        y_eval = np.max(ploty)
        # For each y position generates random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2] for y in ploty])
        rightx = np.array([right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2] for y in ploty])
        # defines conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # fits new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # calculates the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                             (1 + (
                                 2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                     1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        # following lines of code calculate car center deviation from the middle point between to lanes
        # let's assume the car has 2m width
        y = 720
        leftx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
        rightx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
        lane_middle = (leftx + rightx) / 2
        center_deviation = abs((720 - lane_middle) / 720 * 2)
        return left_curverad, right_curverad, center_deviation
    else:
        n = len(files)
        for i in range(n):  # images calibrationXX, where XX = [2,4] are the most distorted from camera images
            # img = cv2.imread(path + files[i])
            img = mpimg.imread(path + files[i])
            frame_buffer_left = deque()
            frame_buffer_right = deque()
            left_fit, right_fit = finding_lines(img, sobel_kernel, mag_thresh, thresh_s, output=1)
            y_eval = np.max(ploty)
            # for each y position generates random x position within +/-50 pix
            # of the line base position in each case (x=200 for left, and x=900 for right)
            leftx = np.array([left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2] for y in ploty])
            rightx = np.array([right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2] for y in ploty])
            # defines conversions in x and y from pixels space to meters
            ym_per_pix = 30 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
            # fits new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
            # calculates the new radii of curvature
            left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            right_curverad = (
                                 (1 + (
                                     2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                                         1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
            # now our radius of curvature is in meters
            # following lines of code calculate car center deviation from the middle point between to lanes
            # let's assume the car has 2m width
            y = 720
            leftx = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
            rightx = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
            lane_middle = (leftx + rightx) / 2
            center_deviation = abs((720 - lane_middle) / 720 * 2)
            print(left_curverad, 'm', right_curverad, 'm', center_deviation, 'm')


# draw text with a curvature on the image
def draw_text(img, text1, text2, text3):
    text4 = 'curvl:' + str(text1) + 'm ' + 'curvr:' + str(text2) + 'm'
    text5 = 'center:' + str(text3) + 'm'
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text4, (200, 50), font, 1, (255, 255, 255), 2)
    img = cv2.putText(img, text5, (200, 80), font, 1, (255, 255, 255), 2)
    return img


# draws green zone on images where car can drive
def draw_lane_zone(img, output=0, src=np.float32([[200, 680], [590, 451], [690, 451], [1042, 666]]),
                   dst=np.float32([[180, 720], [200, 0], [1000, 0], [1000, 720]])):
    # Create an image to draw the lines on
    warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_fit, right_fit = finding_lines(img, output=output)

    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
    index = [n for n, (t, p) in enumerate(zip(left_fitx, right_fitx)) if t >= p]  # finds the point where left and right
    # polynomial intersect
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # draws the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # warps the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    left_curverad, right_curverad, center = calc_curvature(img)
    result = draw_text(result, left_curverad, right_curverad, center)
    return result


# process to apply to still images (*or single video frames)
def process_still_image(image):
    # print('frame processed')
    img = draw_lane_zone(image, output=2)
    return img


def process_video(videofilein, videofileout):
    # clears and initializes frame buffers
    global frame_buffer_left  # collections.deque() for several last values of polynomial coefficients
    global frame_buffer_right  # collections.deque() for several last values of polynomial coefficients
    global frame_curvature_left  # collections.deque() for several last values of curvature
    global frame_curvature_right  # collections.deque() for several last values of curvature
    frame_buffer_left = deque()
    frame_buffer_right = deque()
    frame_curvature_left = deque()
    frame_curvature_right = deque()
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,1)
    clip1 = VideoFileClip(videofilein)#.subclip(20,30)
    write_clip = clip1.fl_image(process_still_image)  # NOTE: this function expects color images!!
    write_clip.write_videofile(videofileout, audio=False)


# main function with the pipeline
def main(argv):
    imageio.plugins.ffmpeg.download()
    if len(argv) == 2:
        path = str(argv[1]) + '/'
    else:
        path = 'camera_cal/'
    nx = 9  # the number of inside corners in x
    ny = 6  # enter the number of inside corners in y
    # calculates calibration parameters for camera
    ret, mtx, dist, rvecs, tvecs = cal_param(path)
    # most distorted images by camera lens:
    files = ['calibration1.jpg', 'calibration2.jpg']
    ret, corners = corr_coef(path, files[1], nx=9, ny=6)
    # draws original image with found corners on it
    fig1 = multiple_plot_uncalibrated(path, [files[1]], cv2.drawChessboardCorners, 'Original image',
                                      'cv2.drawChessboardCorners', (nx, ny), corners, ret)
    # draws original and undistored images
    files = ['calibration1.jpg']
    fig2 = multiple_plot_uncalibrated(path, files, cv2.undistort, 'Original image', 'cv2.undistort', \
                                      mtx, dist, None, mtx)
    # check on 2 files whethers chosen algoritm for camera calibration and "unwarping" images works
    # 'unwarping' transformation is most visible on files: 'calibration2.jpg', 'calibration9.jpg'
    files = ['calibration2.jpg', 'calibration9.jpg']
    fig3 = undistort(path, files, mtx, dist, 'Calibration+Unwarping', nx=9, ny=6)
    # folder with pictures of roads
    path = 'test_images/'
    files = os.listdir(path)
    # choose a Sobel kernel size
    sobel_kernel = 17  # larger odd number smoothes gradient measurements
    mag_thresh = (35, 100)
    thresh = (0.8, 1.570 / 1.5)
    blur_kernel = (3, 3)  # kernel for Gaussian blur
    # applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    #                      mag_thresh)
    # function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    fig5 = multiple_plot(path, files, mag_thresh_func, 'Original image', 'absSobel+gradient', sobel_kernel, mag_thresh)
    delta_y = 60  # offset of the region_of_interest from the middle of image
    delta_x = 10  # offset of the region_of_interest from the middle of image
    x_offset = 0  # offset of the region_of_interest's bottom corners (criticals for the challenge)
    y_offset = 50  # offset of the region_of_interest's bottom corners (critical for the challenge)
    x = 1280
    y = 720
    vertices = np.array([[(x_offset, y - y_offset), (x // 2 - delta_x, y // 2 + delta_y), \
                          (x // 2 + delta_x, y // 2 + delta_y), (x - x_offset, y - y_offset)]], dtype=np.int32)
    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold
    # Only keeps the region of the image defined by the polygon
    # formed from `vertices`. The rest of the image is set to black.
    fig6 = multiple_plot(path, files, dir_threshold, 'Original image', 'SobelXY+gradient', sobel_kernel, thresh,
                         vertices)
    # combimes abs_sobel_thresh() with dir_threshold() and masks everything besides the region of interest
    fig7 = multiple_plot(path, files, combined_sober, 'Original image', 'combinedSober', sobel_kernel, \
                         mag_thresh, thresh, blur_kernel, vertices)

    # # with the following commented out lines of code I tried to find best parameters for the model
    #
    # # for kernel in range(3,15,4):
    # #     for m_thresh in range(1,76,25):
    # #         for m_thresh1 in range(250,101,-50):
    # #             for thresh1 in range(0,10,3):
    # #                 for thresh2 in range(10,17,3):
    # #                     print(sobel_kernel,mag_thresh, thresh, blur_kernel)
    # #                     mag_thresh = (m_thresh,m_thresh1)
    # #                     thresh = (thresh1/10, thresh2/10)
    # #                     sobel_kernel=kernel
    # #                     blur_kernel=(0,0)
    # #                     fig8 = multiple_plot(path, files, combined_sober, 'Original image', 'combined Sober', sobel_kernel,
    # #                                          mag_thresh, thresh, blur_kernel, vertices)
    # #                     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # #                     plt.xticks([])
    # #                     plt.yticks([])
    # #                     fig8.savefig('output/'+str(kernel)+'_'+str(m_thresh)+'_'+str(0)+'_'+str(thresh1)+'_'+str(thresh2)+'.jpg')
    # #                     plt.close(fig8)
    # #
    # #
    # # for kernel in range(3,15,4):
    # #     for blur in range(1, 7, 2):
    # #         for m_thresh in range(1,100,25):
    # #             for m_thresh1 in range(250, 101, -50):
    # #                 for thresh1 in range(0,10,3):
    # #                     for thresh2 in range(10,17,3):
    # #                         print(sobel_kernel,mag_thresh, thresh, blur_kernel)
    # #                         mag_thresh = (m_thresh,m_thresh1)
    # #                         thresh = (thresh1/10, thresh2/10)
    # #                         sobel_kernel=kernel
    # #                         blur_kernel=(blur,blur)
    # #                         fig8 = multiple_plot(path, files, combined_sober, 'Original image', 'combined Sober', sobel_kernel,
    # #                                              mag_thresh, thresh, blur_kernel, vertices)
    # #                         plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # #                         plt.xticks([])
    # #                         plt.yticks([])
    # #                         fig8.savefig('output/'+str(kernel)+'_'+str(m_thresh)+'_'+str(blur)+'_'+str(thresh1)+'_'+str(thresh2)+'.jpg')
    # #                         plt.close(fig8)
    #

    thresh_s = (170, 255)
    # creates bitmap image from saturation channel of HLS images representation
    fig8 = multiple_plot(path, files, saturation_extraction, 'Original Image', 'Saturation', thresh_s, vertices)
    # returns a stack of the two binary images, whose components are sobel transformation of an image and
    # saturation channel from HLS with applied thresholds
    fig9 = multiple_plot(path, files, stack_sobel_saturation, 'Original Image', 'StackedThresholds', sobel_kernel, \
                         mag_thresh, thresh_s, 'color')
    # test of unwarping with manually calculated/selected regions (just to check algorithm)
    src = np.float32([[200, 680], [588, 451], [688, 451], [1042, 666]])
    dst = np.float32([[180, 720], [200, 0], [1000, 0], [1000, 720]])
    # transforms image to bird eye view
    fig10 = multiple_plot(path, files, transform_image, 'Original Image', 'unwarped image', src, dst)
    # findes line on an image
    fig11 = multiple_plot(path, files, finding_lines, 'Original Image', 'ImageWithPolylines', sobel_kernel, mag_thresh,
                          thresh_s, 3)

    # calculates curvature for chosen samples
    calc_curvature(path, files, sobel_kernel, mag_thresh, thresh_s)

    # draw line zone on sample images
    fig12 = multiple_plot(path, files, draw_lane_zone, 'Original Image', 'ZoneBetweenLanes', 1)
    # plt.show()

    #combines masks for white color, yellow(saturation channels from HLS) and Sobel
    fig13 = multiple_plot(path, files, yellow_white_sobel, 'Original Image', 'YellowWhiteSobel')

    # following 4 lines generate big image with zone where car can go
    img = mpimg.imread('test_images' + '/test4.jpg')
    img = draw_lane_zone(img, output=1)
    import matplotlib
    matplotlib.image.imsave('output_images/' + 'ZoneBetweenLinesBig' + '.png', img)
    #
    # following steps apply the pipeline to video streams
    videofilein = 'project_video.mp4'
    videofileout = 'output_video/project_video_out.mp4'
    process_video(videofilein, videofileout)
    # #
    # videofilein = 'challenge_video.mp4'
    # videofileout = 'output_video/challenge_video_out.mp4'
    # process_video(videofilein, videofileout)
    #
    # videofilein = 'harder_challenge_video.mp4'
    # videofileout = 'output_video/harder_challenge_video_out.mp4'
    # process_video(videofilein, videofileout)


if __name__ == '__main__':
    main(sys.argv)
