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
parabola_func = None
vertex = None


# ------------------------------Cubic Interpolation------------------------------
def cubic_equation_solver(d, a):
    d = abs(d)
    if 0.0 <= d <= 1.0:
        score = (a + 2.0) * pow(d, 3.0) - ((a + 3.0) * pow(d, 2.0)) + 1.0
        return score
    elif 1 < d <= 2:
        score = a * pow(d, 3.0) - 5.0 * a * pow(d, 2.0) + 8.0 * a * d - 4.0 * a
        return score
    else:
        return 0.0


def cubic_interpolation(img, x, y):
    newX = x
    newY = y
    dx = abs(newX - round(newX))
    dy = abs(newY - round(newY))
    sum_value = 0
    for c_neighbor in range(-1, 3):
        for r_neighbor in range(-1, 3):
            CaX = cubic_equation_solver(r_neighbor + dx, -0.5)
            CaY = cubic_equation_solver(c_neighbor + dy, -0.5)
            sum_value = sum_value + img[(round(newX) + r_neighbor, c_neighbor + round(newY))] * CaX * CaY

    if sum_value > 255:
        sum_value = 255
    elif sum_value < 0:
        sum_value = 0
    return sum_value


# ------------------------------Nearest neighbor Interpolation------------------------------
def nearest_neighbor_interpolation(img, x, y):
    return img[round(x), round(y)]


# ------------------------------Bilinear Interpolation------------------------------
def bilinear_interpolation(original_img, x, y):
    old_h, old_w = original_img.shape[:2]
    # calculate the coordinate values for 4 (2x2) surrounding pixels.
    x_floor = math.floor(x)
    y_floor = math.floor(y)
    # ensure that its value remains in the range (0 to old_h-1) and (0 to old_w-1)
    x_ceil = min(old_h - 1, math.ceil(x))
    y_ceil = min(old_w - 1, math.ceil(y))

    if (x_ceil == x_floor) and (y_ceil == y_floor):
        q = original_img[int(x), int(y)]
    elif x_ceil == x_floor:
        q1 = original_img[int(x), int(y_floor)]
        q2 = original_img[int(x), int(y_ceil)]
        # relative distance of q1 from (x,y) +relative distance of q2 from (x,y)
        q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor))
    elif y_ceil == y_floor:
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


def draw_rectangle():
    global parab_first_point, parab_second_point
    # Draw the rectangle
    cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (255, 255, 255), 2, 8)
    cv2.rectangle(original_image, top_left_corner[0], bottom_right_corner[0], (255, 255, 255), 2, 8)

    # Compute Medial Line
    distance = bottom_right_corner[0][0] - top_left_corner[0][0]
    parab_first_point = tuple(int(item) for item in (top_left_corner[0][0] + (distance / 2), top_left_corner[0][1]))
    parab_second_point = tuple(
        int(item) for item in (bottom_right_corner[0][0] - (distance / 2), bottom_right_corner[0][1]))

    # Draw Medial line
    cv2.line(image, parab_first_point, parab_second_point, (0, 0, 0), 2)
    cv2.line(original_image, parab_first_point, parab_second_point, (0, 0, 0), 2)
    cv2.imshow("Assignment 1", image)


def draw_parabola(parab_clicked_point_x, parab_clicked_point_y):
    global parab_point_button, parabola_func, parab_first_point, parab_second_point
    pts = np.array([[parab_first_point[0], parab_first_point[1]],
                    [parab_clicked_point_x, parab_clicked_point_y],
                    [parab_second_point[0], parab_second_point[1]]], np.int32)

    # side parabola coeffs
    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 2)

    parabola_func = np.poly1d(coeffs)

    yarr = np.arange(parab_first_point[1], parab_second_point[1])
    xarr = parabola_func(yarr)

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
        cv2.polylines(image, [parab_pts], False, (255, 0, 0), 2)
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
        parab_x = parabola_func(y)
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
    return new_image


def inverse_transformation(img):
    global top_left_corner, bottom_right_corner
    width, height = img.shape[:2]
    scaled_image_nn = np.zeros((width, height), image.dtype)
    scaled_image_b = np.zeros((width, height), image.dtype)
    scaled_image_c = np.zeros((width, height), image.dtype)
    rect_half_width = 0.5 * (bottom_right_corner[0][0] - top_left_corner[0][0])
    rect_half_x = (bottom_right_corner[0][0] + top_left_corner[0][0]) / 2

    for j in range(height):
        for i in range(width):
            parab_x = parabola_func(i)
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
                    scaled_image_nn[i, j] = nearest_neighbor_interpolation(img, y, x)
                    scaled_image_b[i, j] = bilinear_interpolation(img, y, x)
                    scaled_image_c[i, j] = cubic_interpolation(img, y, x)
                # x vals from rect medial line
                else:
                    x = ((((-j + bottom_right_corner[0][0]) / (
                            rect_half_width - relative_right_parab_x)) - 1) * rect_half_width * (-1)) + rect_half_x
                    y = i
                    scaled_image_nn[i, j] = nearest_neighbor_interpolation(img, y, x)
                    scaled_image_b[i, j] = bilinear_interpolation(img, y, x)
                    scaled_image_c[i, j] = cubic_interpolation(img, y, x)

            # if the pixel is not in rect range take the value from original image
            else:
                scaled_image_nn[i, j] = img[i, j]
                scaled_image_b[i, j] = img[i, j]
                scaled_image_c[i, j] = img[i, j]
    cv2.imshow("Nearest Neighbor Interpolation", scaled_image_nn)
    cv2.imshow("Bilinear Interpolation", scaled_image_b)
    cv2.imshow("Cubic Interpolation", scaled_image_c)
    cv2.waitKey(0)


# function which will be called on mouse input
def click_handler(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, \
        top_left_corner_button, bottom_right_corner_button, \
        parab_point_button, vertex, original_image

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


def load_image():
    global image, relative_path, original_image, checking_image
    image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)


def display_window():
    global image
    cv2.imshow("Assignment 1", image)
    cv2.setMouseCallback('Assignment 1', click_handler)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check if the file exists
    # if os.path.exists(sys.argv):
    if os.path.exists(relative_path):
        load_image()
        display_window()
        if vertex is not None:
            inverse_transformation(original_image)
    else:
        print("False pathname")
