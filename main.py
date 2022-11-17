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
poly = None


# ------------------------------Cubic Interpolation------------------------------
# def cubic_pixel_val(original_img, x, y):
#     rows, cols = original_img.shape
#
#     x_ratio = float(cols / scaled_weight)
#     y_ratio = float(rows / scaled_height)
#
#     C = np.zeros(5)
#     x_int = int(j * x_ratio)
#     y_int = int(i * y_ratio)
#
#     dx = x_ratio * j - x_int
#     dy = y_ratio * i - y_int
#
#     for jj in range(0, 4):
#         o_y = y_int - 1 + jj
#         a0 = getBicPixel(image, x_int, o_y)
#         d0 = getBicPixel(image, x_int - 1, o_y) - a0
#         d2 = getBicPixel(image, x_int + 1, o_y) - a0
#         d3 = getBicPixel(image, x_int + 2, o_y) - a0
#
#         a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#         a2 = 1. / 2 * d0 + 1. / 2 * d2
#         a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#         C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx
#
#     d0 = C[0] - C[1]
#     d2 = C[2] - C[1]
#     d3 = C[3] - C[1]
#     a0 = C[1]
#     a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#     a2 = 1. / 2 * d0 + 1. / 2 * d2
#     a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#     val = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
#     return val

def cubicEquationSolver(d, a):
    d = abs(d)
    if 0.0 <= d <= 1.0:
        score = (a + 2.0) * pow(d, 3.0) - ((a + 3.0) * pow(d, 2.0)) + 1.0
        return score
    elif 1 < d <= 2:
        score = a * pow(d, 3.0) - 5.0 * a * pow(d, 2.0) + 8.0 * a * d - 4.0 * a
        return score
    else:
        return 0.0


def Cubic_Interpolation(img, x, y):
    newX = x
    newY = y
    dx = abs(newX - round(newX))
    dy = abs(newY - round(newY))
    sumCubicGrayValue = 0
    # if math.floor(newX) - 1  < 0 or math.floor(newX) + 2  > src.cols - 1 or math.floor(newY) < 0 or math.floor(newY)  > src.rows - 1:
    #     dst[dstPixel] = 0
    # else:
    for cNeighbor in range(-1, 3):
        for rNeighbor in range(-1, 3):
            CaX = cubicEquationSolver(rNeighbor + dx, -0.5)
            CaY = cubicEquationSolver(cNeighbor + dy, -0.5)
            sumCubicGrayValue = sumCubicGrayValue + img[(round(newX) + rNeighbor, cNeighbor + round(newY))] * CaX * CaY

    if sumCubicGrayValue > 255:
        sumCubicGrayValue = 255
    elif sumCubicGrayValue < 0:
        sumCubicGrayValue = 0
    return sumCubicGrayValue


def getBicPixel(img, x, y):
    if (x < img.shape[1]) and (y < img.shape[0]):
        return img[y, x] & 0xFF

    return 0


#
# def cubic_interpolation(img, scale_factor):
#     rows, cols = img.shape
#     scaled_height = int(math.ceil(float(rows * scale_factor[0])))
#     scaled_weight = int(math.ceil(float(cols * scale_factor[1])))
#
#     scaled_image = np.zeros((scaled_weight, scaled_height), np.uint8)
#
#     x_ratio = float(cols / scaled_weight)
#     y_ratio = float(rows / scaled_height)
#
#     C = np.zeros(5)
#
#     for i in range(0, scaled_height):
#         for j in range(0, scaled_weight):
#
#             x_int = int(j * x_ratio)
#             y_int = int(i * y_ratio)
#
#             dx = x_ratio * j - x_int
#             dy = y_ratio * i - y_int
#
#             for jj in range(0, 4):
#                 o_y = y_int - 1 + jj
#                 a0 = getBicPixel(image, x_int, o_y)
#                 d0 = getBicPixel(image, x_int - 1, o_y) - a0
#                 d2 = getBicPixel(image, x_int + 1, o_y) - a0
#                 d3 = getBicPixel(image, x_int + 2, o_y) - a0
#
#                 a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#                 a2 = 1. / 2 * d0 + 1. / 2 * d2
#                 a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#                 C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx
#
#             d0 = C[0] - C[1]
#             d2 = C[2] - C[1]
#             d3 = C[3] - C[1]
#             a0 = C[1]
#             a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
#             a2 = 1. / 2 * d0 + 1. / 2 * d2
#             a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
#             scaled_image[j, i] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
#     return scaled_image


# ------------------------------Nearest neighbor Interpolation------------------------------
def nearest_neighbor_interpolation(img, x, y):
    return img[round(x), round(y)]


# ------------------------------Bilinear Interpolation------------------------------
def bilinear_interpolation(original_img, x, y):
    old_h, old_w = original_img.shape[:2]
    # calculate the coordinate values for 4 (2x2) surrounding pixels.
    x_floor = math.floor(x)
    y_floor = math.floor(y)
    ##ensure that its value remains in the range (0 to old_h-1) and (0 to old_w-1)
    x_ceil = min(old_h - 1, math.ceil(x))
    y_ceil = min(old_w - 1, math.ceil(y))

    if (x_ceil == x_floor) and (y_ceil == y_floor):
        q = original_img[int(x), int(y)]
    elif (x_ceil == x_floor):
        q1 = original_img[int(x), int(y_floor)]
        q2 = original_img[int(x), int(y_ceil)]
        # relative distance of q1 from (x,y) +relative distance of q2 from (x,y)
        q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
    elif (y_ceil == y_floor):
        q1 = original_img[int(x_floor), int(y)]
        q2 = original_img[int(x_ceil), int(y)]
        # relative distance of q1 from (x,y) +relative distance of q2 from (x,y)
        q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
    else:
        v1 = original_img[x_floor, y_floor]
        v2 = original_img[x_ceil, y_floor]
        v3 = original_img[x_floor, y_ceil]
        v4 = original_img[x_ceil, y_ceil]

        q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
        q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
        q = q1 * (y_ceil - y) + q2 * (y - y_floor)

    return q


def get_interpolation(interpolation=None):
    if interpolation == 'nearest_neighbor':
        return nearest_neighbor_interpolation

    elif interpolation == 'bilinear':
        return bilinear_interpolation

    # elif interpolation == 'cubic':
    #     # return cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
    #     return cubic_interpolation(img, (1.2, 1.2))


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
    global parab_point_button, poly, parab_first_point, parab_second_point
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


def transformation(img):
    col, row = img.shape[:2]
    new_image = np.zeros((col, row), image.dtype)
    rect_half_width = 0.5 * (bottom_right_corner[0][0] - top_left_corner[0][0])
    rect_half_x = round((bottom_right_corner[0][0] + top_left_corner[0][0]) / 2)

    for y in range(top_left_corner[0][1], bottom_right_corner[0][1]):
        # parabola x value in row y = > parabola(y)=x
        parab_x = poly(y)
        relative_left_parab_x = parab_x - top_left_corner[0][0]
        relative_right_parab_x = parab_x - rect_half_x
        for x in range(top_left_corner[0][0], bottom_right_corner[0][0]):
            if x <= rect_half_x:
                relative_x = x - top_left_corner[0][0]
                # linear scale using x value
                # if x==  top_left_corner[0][0] then scale=0 (0%)
                # if x==        rect_half_width then scale=1 (100%)
                x_scale = relative_x / rect_half_width
                delta = round(relative_left_parab_x * x_scale)
                new_image[y - col, top_left_corner[0][0] + delta - row] = img[y, x]
            else:
                relative_x = (x - rect_half_x)
                x_scale = 1 - (relative_x / rect_half_width)
                delta = round((rect_half_width - relative_right_parab_x) * (x_scale))
                new_image[y - col, bottom_right_corner[0][0] - delta - row] = img[y, x]
                # j=bottom_right_corner[0][0] - (rect_half_width - relative_right_parab_x) * (1 - ((x - rect_half_x) / rect_half_width))

    cv2.imshow("Assignment 1", new_image)
    cv2.waitKey(0)


def inverse_transformation(img):
    global top_left_corner, bottom_right_corner
    width, height = img.shape[:2]
    scaled_image = np.zeros((width, height), image.dtype)
    rect_half_width = 0.5 * (bottom_right_corner[0][0] - top_left_corner[0][0])
    rect_half_x = (bottom_right_corner[0][0] + top_left_corner[0][0]) / 2

    for j in range(height):
        for i in range(width):
            parab_x = poly(i)
            relative_left_parab_x = parab_x - top_left_corner[0][0]
            relative_right_parab_x = parab_x - rect_half_x
            # if the pixel in rect range copmute
            if top_left_corner[0][0] <= j <= bottom_right_corner[0][0] \
                    and top_left_corner[0][1] <= i <= bottom_right_corner[0][1]:
                # x vals up to rect medial line
                if j <= parab_x:

                    # compute (y,x) pixel from original image
                    x = (((j - top_left_corner[0][0]) / relative_left_parab_x) * rect_half_width) + top_left_corner[0][
                        0]
                    y = i
                    scaled_image[i, j] = Cubic_Interpolation(img, y, x)
                # x vals from rect medial line
                else:
                    x = ((((-j + bottom_right_corner[0][0]) / (
                            rect_half_width - relative_right_parab_x)) - 1) * rect_half_width * (-1)) + rect_half_x
                    y = i
                    scaled_image[i, j] = Cubic_Interpolation(img, y, x)

            # if the pixel is not in rect range take the value from original image
            else:
                scaled_image[i, j] = img[i, j]
    cv2.imshow("Bilinear Interpolation", scaled_image)
    cv2.waitKey(0)


# def deformat():
#     # nearest_neighbor = inverse_transformation(original_image, interpolation="nearest_neighbor")
#     bilinear = inverse_transformation(original_image)
#     # cv2.imshow("Nearest Neighbor Interpolation", nearest_neighbor)
#     cv2.imshow("Bilinear Interpolation", bilinear)
#     cv2.waitKey(0)


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
        # if vertex is not None:
        #     deformat()


def load_image():
    global image, relative_path, original_image, checking_image
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
        inverse_transformation(original_image)
    else:
        print("False pathname")
