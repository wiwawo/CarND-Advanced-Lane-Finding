﻿## Advanced Lane Finding
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/ZoneBetweenLinesBig.png" width="480" alt="lane lines" />
</p>

In this project, the goal is to write a software pipeline to identify the lane boundaries in a video.

The steps of this project are the following:

* Computation of the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Application of a distortion correction to raw images.
* Use of color transforms, gradients, etc., to create a thresholded binary image.
* Application of a perspective transform to rectify binary image ("birds-eye view").
* Detection of lane pixels and fit to find the lane boundary.
* Determination of the curvature of the lane and vehicle position with respect to center.
* Warping of the detected lane boundaries back onto the original image.
* Output on visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Pipeline
---
Since there is mostly processing of visual information in the project, I will demonstrate each step of the pipeline with a picture. To see exact code for every transformation, please check the file [carnd_advanced_lines.py](carnd_advanced_lines.py). To be sure, that the chosen algorithms perform well on test images, all test images will be processed and visualised.

---
#### Camera calibration
---
Camera will be calibrated using images of a chessboard from different angles made using the camera I want to calibrate. For this was used the function from cv2 library cv2.findChessboardCorners.
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/cv2.drawChessboardCorners.jpg" width="480" alt="lane lines" />
</p>
After finding corners of chessboards in the images, I calculated disortion coefficients for the camera using cv2.calibrateCamera and applied them to calibrate camera/undistort the images. The result of calibration is:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/cv2.undistort.jpg" width="480" alt="lane lines" />
</p>
After undistorting image, the images were unwarped to see, whether calibration was effective:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/Calibration+Unwarping.jpg" width="480" alt="lane lines" />
</p>

#### Transformation of images using gradients and thresholds

---
To localize  lines on images, Sobel function np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)) was used. This function allows in cv2 select which lines you want to keep in the images, vertical, horizontal or both. This step of the pipeline lets eliminate part of the images without any lines on it (like sky).
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/abs_sobel_thresh.jpg" width="480" alt="lane lines" />
</p>

Followoing function applies Sobel x and y then computes the magnitude of the gradient and applies a threshold:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/absSobel+gradient.jpg" width="480" alt="lane lines" />
</p>
Combining Sobel x+y with gradient and threshold, following result was achieved:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/SobelXY+gradient.jpg" width="480" alt="lane lines" />
</p>
Combining functions from previous steps and deleted everything outside the region of interest, I got following outputs:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/combinedSober.jpg" width="480" alt="lane lines" />
</p>
Working just with gradients and Sobel function showed results, but it was sensitive to changes in color saturation, therefore was made decision to try HLS color space. Here you see the saturation channel from HLS:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/Saturation.jpg" width="480" alt="lane lines" />
</p>
Combining Sobel transformations with the saturation channel from HLS can potentially show better result, than those 2 methods separately:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/StackedThresholds.jpg" width="480" alt="lane lines" />
</p>
After transforming image to the bird-eye view and fitting 2nd order polynomial to the lanes, following was achieved:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/unwarped image.jpg" width="480" alt="lane lines" />
</p>
Here is more pictures with more details, how similar found polynomial are to the lines on images:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/ImageWithPolylines.jpg" width="480" alt="lane lines" />
</p>
After combining results from all the steps from above, it was possible to select lane area (green on the following images), where car can drive. Of course, the warping of the detected lane boundaries back onto the original image was done:
<p align='center'>
<img src="https://github.com/wiwawo/CarND-Advanced-Lane-Finding/blob/master/output_images/ZoneBetweenLanes.jpg" width="480" alt="lane lines" />
</p>
Curvatures of left and right lanes and car deviation from the middle point between lanes for 8 test images are:

    1219.5933787 m 16267.9628753 m 0.305998966581 m
    3182.33821341 m 6460.54903772 m 0.292650466693 m
    399.987575789 m 3.86073595145 m 0.524978911501 m
    360.41727769 m 208.322489641 m 0.16397388477 m
    1290.85485352 m 436.613826062 m 0.228975554875 m
    572.082057411 m 259.376784967 m 0.115939750357 m
    170.248655436 m 1341.35938086 m 0.328530350628 m
    596.257467939 m 769.784082996 m 0.162777691917 m
These values seems to close to real values.

Having pipeline, that works on the individual frames, it was not difficult to apply the pipeline on individual frames of videos. The pipeline did work well only on project_video.mp4.

#### Possible improvements

* use of processing visual information from several cameras simultaneously can significantly improve the performance of the system;
* doesn't matter what algorithm will be used to process visual information received from cameras, it will be still susceptible to garbage/snow etc. on roads or extremely bright light. Using additional sensors (lidars, radars etc.) with the developed in this project pipeline will improve performance of the system.

