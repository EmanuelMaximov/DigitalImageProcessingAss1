import cv2
import os
import numpy as np
import math
import sys

# Read Images
image, original_image = [], []
relative_path = "image200.jpg"

# Lists to store the bounding box coordinates
top_left_corner, bottom_right_corner = [], []
top_left_corner_button, bottom_right_corner_button, parab_point_button = False, False, False
parab_first_point, parab_second_point = (), ()


# ------------------------------Cubic Interpolation------------------------------
# def getBicPixelChannel(img, x, y, channel):
#     if (x < img.shape[1]) and (y < img.shape[0]):
#         return img[y, x, channel]
#     return 0
#
#
# def Bicubic(img, rate):
#     new_w = int(math.ceil(float(img.shape[1]) * rate))
#     new_h = int(math.ceil(float(img.shape[0]) * rate))
#
#     new_img = np.zeros((new_w, new_h, 3))
#
#     x_rate = float(img.shape[1]) / new_img.shape[1]
#     y_rate = float(img.shape[0]) / new_img.shape[0]
#
#     C = np.zeros(5)
#
#     for hi in range(new_img.shape[0]):
#         for wi in range(new_img.shape[1]):
#
#             x_int = int(wi * x_rate)
#             y_int = int(hi * y_rate)
#
#             dx = x_rate * wi - x_int
#             dy = y_rate * hi - y_int
#
#             for channel in range(new_img.shape[2]):
#                 for jj in range(0, 4):
#                     o_y = y_int - 1 + jj
#                     a0 = getBicPixelChannel(img, x_int, o_y, channel)
#                     d0 = getBicPixelChannel(img, x_int - 1, o_y, channel) - a0
#                     d2 = getBicPixelChannel(img, x_int + 1, o_y, channel) - a0
#                     d3 = getBicPixelChannel(img, x_int + 2, o_y, channel) - a0
#
#                     a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#                     a2 = 1. / 2 * d0 + 1. / 2 * d2
#                     a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#                     C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx
#
#                 d0 = C[0] - C[1]
#                 d2 = C[2] - C[1]
#                 d3 = C[3] - C[1]
#                 a0 = C[1]
#                 a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#                 a2 = 1. / 2 * d0 + 1. / 2 * d2
#                 a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#                 new_img[hi, wi, channel] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
#
#     return new_img


# ------------------------------Nearest neighbor Interpolation------------------------------
def nearest_neighbor_interpolation(img, new_size):
    """Vectorized Nearest Neighbor Interpolation"""
    old_size = img.shape
    row_ratio, col_ratio = np.array(new_size) / np.array(old_size)

    # row wise interpolation
    row_idx = (np.ceil(range(1, 1 + int(old_size[0] * row_ratio)) / row_ratio) - 1).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1] * col_ratio)) / col_ratio) - 1).astype(int)

    final_matrix = img[row_idx, :][:, col_idx]
    return final_matrix


# ------------------------------Bilinear Interpolation------------------------------
def bilinear_pixel(image, y, x):
    height = image.shape[0]
    width = image.shape[1]

    x1 = max(min(math.floor(x), width - 1), 0)
    y1 = max(min(math.floor(y), height - 1), 0)
    x2 = max(min(math.ceil(x), width - 1), 0)
    y2 = max(min(math.ceil(y), height - 1), 0)

    a = float(image[y1, x1])
    b = float(image[y2, x1])
    c = float(image[y1, x2])
    d = float(image[y2, x2])

    dx = x - x1
    dy = y - y1

    new_pixel = a * (1 - dx) * (1 - dy)
    new_pixel += b * dy * (1 - dx)
    new_pixel += c * dx * (1 - dy)
    new_pixel += d * dx * dy
    return round(new_pixel)


def bilinear_interpolation(image, new_height, new_width):
    new_image = np.zeros((new_height, new_width),
                         image.dtype)  # new_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    orig_height = image.shape[0]
    orig_width = image.shape[1]

    # Compute center column and center row
    x_orig_center = (orig_width - 1) / 2
    y_orig_center = (orig_height - 1) / 2

    # Compute center of resized image
    x_scaled_center = (new_width - 1) / 2
    y_scaled_center = (new_height - 1) / 2

    # Compute the scale in both axes
    scale_x = orig_width / new_width;
    scale_y = orig_height / new_height;

    for y in range(new_height):
        for x in range(new_width):
            x_ = (x - x_scaled_center) * scale_x + x_orig_center
            y_ = (y - y_scaled_center) * scale_y + y_orig_center

            new_image[y, x] = bilinear_pixel(image, y_, x_)

    return new_image


def resize(img, new_height, new_width, interpolation=None):
    if interpolation == 'nearest_neighbor':
        return nearest_neighbor_interpolation(img, (new_height, new_width))

    elif interpolation == 'bilinear':
        return bilinear_interpolation(img, new_height, new_width)

    elif interpolation == 'cubic':
        return cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_CUBIC)


def draw_rectangle():
    global parab_first_point, parab_second_point
    # Draw the rectangle
    cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0, 0, 0), 2, 8)

    # Compute Medial Line
    distance = bottom_right_corner[0][0] - top_left_corner[0][0]
    parab_first_point = tuple(int(item) for item in (top_left_corner[0][0] + (distance / 2), top_left_corner[0][1]))
    parab_second_point = tuple(
        int(item) for item in (bottom_right_corner[0][0] - (distance / 2), bottom_right_corner[0][1]))

    # Draw Medial line
    cv2.line(image, parab_first_point, parab_second_point, (0, 0, 0), 2)
    cv2.imshow("Assignment 1", image)


def draw_parabola(parab_clicked_point_x, parab_clicked_point_y):
    global parab_point_button
    pts = np.array([[parab_first_point[0], parab_first_point[1]],
                    [parab_clicked_point_x, parab_clicked_point_y],
                    [parab_second_point[0], parab_second_point[1]]], np.int32)

    # side parabola coeffs
    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)
    poly = np.poly1d(coeffs)
    yarr = np.arange(parab_first_point[1], parab_second_point[1])
    xarr = poly(yarr)
    parab_pts = np.array([xarr, yarr], dtype=np.int32).T

    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    vertex = (round((((4 * a * c) - (b * b)) / (4 * a))), round((-b / (2 * a))))

    # Ensure parabola won't be drawn outside the rectangle
    if top_left_corner[0][0] < vertex[0] < bottom_right_corner[0][0]:
        # Draw horizontal line through parabola vertex for reference
        cv2.line(image, (parab_first_point[0], round(0.5 * (parab_first_point[1] + parab_second_point[1]))), vertex,
                 (255, 255, 255), 2)
        # Draw Parabola
        cv2.polylines(image, [parab_pts], False, (255, 0, 0), 3)
        cv2.imshow("Assignment 1", image)
        return vertex
    else:
        parab_point_button = False
        return None


def draw_half_circle(clicked_point_x, clicked_point_y):
    point1 = np.array((parab_first_point[0], 0.5 * (parab_first_point[1] + parab_second_point[1])))
    point2 = np.array((clicked_point_x, clicked_point_y))
    center = (parab_first_point[0], round(0.5 * (parab_first_point[1] + parab_second_point[1])))
    radius = round(np.linalg.norm(point1 - point2))
    axes = (radius, radius)
    angle = 270
    startAngle = 0
    endAngle = 180

    cv2.ellipse(image, center, axes, angle, startAngle, endAngle, (0, 0, 0), 3)
    cv2.imshow("Assignment 1", image)


def compute_cropped_area(vertex):
    global top_left_corner, bottom_right_corner, parab_first_point, parab_second_point, original_image
    # Example: cropped = img[start_row:end_row, start_col:end_col]
    cropped_left_half = original_image[top_left_corner[0][1]:bottom_right_corner[0][1],
                        top_left_corner[0][0]:parab_first_point[0]]
    cropped_right_half = original_image[top_left_corner[0][1]:bottom_right_corner[0][1],
                         parab_first_point[0]:bottom_right_corner[0][0]]

    new_width_left = vertex[0] - top_left_corner[0][0]
    new_width_right = bottom_right_corner[0][0] - vertex[0]
    new_height = parab_second_point[1] - parab_first_point[1]

    # Nearest Neighbor
    resized_left_half_NN = resize(cropped_left_half, new_height, new_width_left, interpolation='nearest_neighbor')
    resized_right_half_NN = resize(cropped_right_half, new_height, new_width_right, interpolation='nearest_neighbor')
    new_image_NN = np.concatenate((resized_left_half_NN, resized_right_half_NN), axis=1)

    # Bilinear
    resized_left_half_bilinear = resize(cropped_left_half, new_height, new_width_left, interpolation='bilinear')
    resized_right_half_bilinear = resize(cropped_right_half, new_height, new_width_right, interpolation='bilinear')
    new_image_biliniar = np.concatenate((resized_left_half_bilinear, resized_right_half_bilinear), axis=1)

    # Cubic
    # resized_left_half_BC = resize(cropped_left_half, new_width_left, new_height, interpolation='cubic')
    # resized_right_half_BC = resize(cropped_right_half, new_height, new_width_right, interpolation='cubic')
    # new_image_BC = np.concatenate((resized_left_half_BC, resized_right_half_BC), axis=1)

    cv2.imshow("Bilinear", new_image_biliniar)
    cv2.imshow("Nearest Neighbor", new_image_NN)
    # cv2.imshow("Cubic", new_image_BC)


# function which will be called on mouse input
def click_handler(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, \
        top_left_corner_button, bottom_right_corner_button, \
        parab_point_button

    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN and not top_left_corner_button:
        top_left_corner = [(x, y)]
        top_left_corner_button = True

    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP and not bottom_right_corner_button:
        bottom_right_corner = [(x, y)]
        bottom_right_corner_button = True
        draw_rectangle()

    elif action == cv2.EVENT_LBUTTONUP and bottom_right_corner_button and (not parab_point_button):
        parab_point_button = True
        vertex = draw_parabola(x, y)
        if vertex is not None:
            compute_cropped_area(vertex)


def load_image():
    global image, relative_path, original_image
    image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)


def display_window():
    global image
    cv2.imshow("Assignment 1", image)
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('Assignment 1', click_handler)
    # First Parameter is for holding screen for specified milliseconds
    # It should be positive integer. If 0 pass an parameter, then it will
    # hold the screen until user close it.
    cv2.waitKey(0)
    # It is for removing/deleting created GUI window from screen and memory
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check if the file exists
    # if os.path.exists(sys.argv):
    if os.path.exists(relative_path):
        load_image()
        display_window()
    else:
        print("False pathname")
