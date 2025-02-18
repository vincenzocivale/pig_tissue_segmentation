import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def extract_smoothed_contours(image, smooth_factor=5, threshold1=50, threshold2=150):
  """
  Extracts and smooths contours based on the image gradient.

  Parameters:
    - image: grayscale image (numpy array).
    - smooth_factor: smoothing factor for contour approximation.
    - threshold1, threshold2: thresholds for edge detection with Canny.

  Returns:
    - smooth_contours: list of NumPy arrays with smoothed contours.
  """
  # Compute the gradient with the Canny filter
  edges = cv2.Canny(image, threshold1, threshold2)

  # Find contours on the edge map
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  smooth_contours = []

  for cnt in contours:
    cnt = cnt.squeeze()  # Remove unnecessary dimensions

    if len(cnt) >= 5:  # At least 5 points are needed for smoothing
      # Interpolation with Spline
      tck, u = splprep(cnt.T, s=smooth_factor)
      u_new = np.linspace(u.min(), u.max(), len(cnt) * 2)  # More detail
      smoothed = np.array(splev(u_new, tck)).T.astype(np.int32)

      smooth_contours.append(smoothed)

  return smooth_contours

def draw_dashed_contours(image, contours, color=(0, 255, 0), dash_length=5, space_length=5):
  """
  Draws dashed contours on an image.

  Parameters:
    - image: image on which to draw the contours.
    - contours: list of NumPy arrays with the contours.
    - color: color of the dashes (BGR, default green).
    - dash_length: length of the solid dash.
    - space_length: length of the space between dashes.

  Returns:
    - img_with_dashed_contours: image with dashed contours.
  """
  img_with_dashed_contours = image.copy()

  for contour in contours:
    contour = contour.squeeze()  # Remove unnecessary dimensions (avoids issues with multi-dimensional arrays)
    for i in range(0, len(contour) - 1, dash_length + space_length):
      start_point = tuple(map(int, contour[i]))  # Convert to tuple of integers
      end_index = min(i + dash_length, len(contour) - 1)
      end_point = tuple(map(int, contour[end_index]))  # Convert to tuple of integers
      cv2.line(img_with_dashed_contours, start_point, end_point, color, 1)

  return img_with_dashed_contours
