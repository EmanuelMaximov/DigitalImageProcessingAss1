import cv2
import os
import numpy as np
import math
import sys

# Read Images
image, original_image = [], []
checking_image = []
relative_path = "image200.jpg"

# Lists to store the bounding box coordinates
top_left_corner, bottom_right_corner = [], []
top_left_corner_button, bottom_right_corner_button, parab_point_button = False, False, False
parab_first_point, parab_second_point = (), ()
parab_coeffs = (0, 0, 0)
poly = None
half_rect = None
parab_pts = None


# ------------------------------Cubic Interpolation------------------------------
def getBicPixel(img, x, y):
    if (x < img.shape[1]) and (y < img.shape[0]):
        return img[y, x] & 0xFF

    return 0


def cubic_interpolation(img, scale_factor):
    rows, cols = img.shape
    scaled_height = int(math.ceil(float(rows * scale_factor[0])))
    scaled_weight = int(math.ceil(float(cols * scale_factor[1])))

    scaled_image = np.zeros((scaled_weight, scaled_height), np.uint8)

    x_ratio = float(cols / scaled_weight)
    y_ratio = float(rows / scaled_height)

    C = np.zeros(5)

    for i in range(0, scaled_height):
        for j in range(0, scaled_weight):

            x_int = int(j * x_ratio)
            y_int = int(i * y_ratio)

            dx = x_ratio * j - x_int
            dy = y_ratio * i - y_int

            for jj in range(0, 4):
                o_y = y_int - 1 + jj
                a0 = getBicPixel(image, x_int, o_y)
                d0 = getBicPixel(image, x_int - 1, o_y) - a0
                d2 = getBicPixel(image, x_int + 1, o_y) - a0
                d3 = getBicPixel(image, x_int + 2, o_y) - a0

                a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
                a2 = 1. / 2 * d0 + 1. / 2 * d2
                a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
                C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx

            d0 = C[0] - C[1]
            d2 = C[2] - C[1]
            d3 = C[3] - C[1]
            a0 = C[1]
            a1 = -1. / 3 * d0 + d2 - 1. / 6 * d3
            a2 = 1. / 2 * d0 + 1. / 2 * d2
            a3 = -1. / 6 * d0 - 1. / 2 * d2 + 1. / 6 * d3
            scaled_image[j, i] = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
    return scaled_image


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

def bilinear_my_way(original_img, new_h, new_w):
    # get dimensions of original image
    # if the image is GrayScale the is no shape[2], if COLOR and in RGB then shape[2]=3
    old_h, old_w = original_img.shape[:2]
    # create an array of the desired shape and initialize with black pixels
    resized = np.zeros((new_h, new_w))
    # Calculate width and height scaling factors
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0

    for i in range(new_h):
        for j in range(new_w):
            # map the coordinates back to the original image = (x,y)=(i,j)*scale
            x = i * h_scale_factor
            y = j * w_scale_factor
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

            resized[i, j] = q
    return resized.astype(np.uint8)


def bl_resize(original_img, new_h, new_w):
    # get dimensions of original image
    old_h, old_w, c = original_img.shape
    # create an array of the desired shape.
    # We will fill-in the values later.
    resized = np.zeros((new_h, new_w, c))
    # Calculate horizontal and vertical scaling factor
    w_scale_factor = (old_w) / (new_w) if new_h != 0 else 0
    h_scale_factor = (old_h) / (new_h) if new_w != 0 else 0
    for i in range(new_h):
        for j in range(new_w):
            # map the coordinates back to the original image
            x = i * h_scale_factor
            y = j * w_scale_factor
            # calculate the coordinate values for 4 surrounding pixels.
            x_floor = math.floor(x)
            x_ceil = min(old_h - 1, math.ceil(x))
            y_floor = math.floor(y)
            y_ceil = min(old_w - 1, math.ceil(y))

            if (x_ceil == x_floor) and (y_ceil == y_floor):
                q = original_img[int(x), int(y), :]
            elif (x_ceil == x_floor):
                q1 = original_img[int(x), int(y_floor), :]
                q2 = original_img[int(x), int(y_ceil), :]
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)
            elif (y_ceil == y_floor):
                q1 = original_img[int(x_floor), int(y), :]
                q2 = original_img[int(x_ceil), int(y), :]
                q = (q1 * (x_ceil - x)) + (q2 * (x - x_floor))
            else:
                v1 = original_img[x_floor, y_floor, :]
                v2 = original_img[x_ceil, y_floor, :]
                v3 = original_img[x_floor, y_ceil, :]
                v4 = original_img[x_ceil, y_ceil, :]

                q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
                q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
                q = q1 * (y_ceil - y) + q2 * (y - y_floor)

            resized[i, j, :] = q
    return resized.astype(np.uint8)


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
        # return cv2.resize(img, (new_height, new_width), interpolation=cv2.INTER_CUBIC)
        return cubic_interpolation(img, (1.2, 1.2))


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
    global parab_point_button, parab_coeffs, poly, parab_pts
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
    parab_coeffs = (a, b, c)
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
    global top_left_corner, bottom_right_corner, parab_first_point, parab_second_point, original_image, parab_coeffs
    # Example: cropped = img[start_row:end_row, start_col:end_col]
    cropped_left_half = original_image[top_left_corner[0][1]:bottom_right_corner[0][1],
                        top_left_corner[0][0]:parab_first_point[0]]
    cropped_right_half = original_image[top_left_corner[0][1]:bottom_right_corner[0][1],
                         parab_first_point[0]:bottom_right_corner[0][0]]

    new_width_left = vertex[0] - top_left_corner[0][0]
    new_width_right = bottom_right_corner[0][0] - vertex[0]
    new_height = parab_second_point[1] - parab_first_point[1]
    #
    # # Nearest Neighbor
    # resized_left_half_NN = resize(cropped_left_half, new_height, new_width_left, interpolation='nearest_neighbor')
    # resized_right_half_NN = resize(cropped_right_half, new_height, new_width_right, interpolation='nearest_neighbor')
    # new_image_NN = np.concatenate((resized_left_half_NN, resized_right_half_NN), axis=1)
    #
    # # new_image_NN = removecolor(image, parab_coeffs)
    #
    # Bilinear

    # rect = original_image[top_left_corner[0][1]:bottom_right_corner[0][1],
    #        top_left_corner[0][0]:bottom_right_corner[0][0]]
    # rect = resize(rect, rect.shape[0], rect.shape[1], interpolation='bilinear')

    # resized_left_half_bilinear = resize(cropped_left_half, new_height, new_width_left, interpolation='bilinear')
    # resized_right_half_bilinear = resize(cropped_right_half, new_height, new_width_right, interpolation='bilinear')
    # new_image_biliniar = np.concatenate((resized_left_half_bilinear, resized_right_half_bilinear), axis=1)

    # Cubic
    # resized_left_half_BC = resize(checking_image, new_width_left, new_height, interpolation='cubic')
    # resized_right_half_BC = resize(cropped_right_half, new_height, new_width_right, interpolation='cubic')
    # new_image_BC = np.concatenate((resized_left_half_BC, resized_right_half_BC), axis=1)

    # cv2.imshow("Bilinear", rect)
    # cv2.imshow("Nearest Neighbor", new_image_NN)
    # cv2.imshow("Cubic", resized_left_half_BC)


def new_deformat(img):
    col, row = img.shape[:2]
    scaled_image = np.zeros((col, row), image.dtype)
    rect_half_width = 0.5 * (bottom_right_corner[0][0] - top_left_corner[0][0])
    rect_half_x = round((bottom_right_corner[0][0] + top_left_corner[0][0]) / 2)

    for y in range(top_left_corner[0][1], bottom_right_corner[0][1]):
        parab_x = poly(y)
        relative_left_parab_x = parab_x - top_left_corner[0][0]
        relative_right_parab_x = parab_x - rect_half_x
        for x in range(top_left_corner[0][0], bottom_right_corner[0][0]):
            if x <= rect_half_x:
                relative_x = x - top_left_corner[0][0]
                x_scale = relative_x / rect_half_width
                delta = round(relative_left_parab_x * x_scale)
                scaled_image[y - col, top_left_corner[0][0] + delta - row] = img[y, x]
            else:
                relative_x = (x - rect_half_x)
                x_scale = 1 - (relative_x / rect_half_width)
                #
                delta = round((rect_half_width - relative_right_parab_x) * (x_scale))
                scaled_image[y - col, bottom_right_corner[0][0] - delta - row] = img[y, x]
    cv2.imshow("Assignment 2", scaled_image)
    cv2.waitKey(0)


def deformat(img):
    global poly
    # poly(y)=x

    concated_image = None
    for y in range(top_left_corner[0][1], bottom_right_corner[0][1]):
        eq = round(poly(y))
        cropped_left = img[y:y + 1,
                       top_left_corner[0][0]:parab_first_point[0]]
        cropped_right = img[y:y + 1,
                        parab_first_point[0]:bottom_right_corner[0][0]]

        new_width_left = eq - top_left_corner[0][0]
        new_width_right = bottom_right_corner[0][0] - eq
        new_height = 1

        cropped_left = resize(cropped_left, new_height, new_width_left, interpolation='bilinear')
        cropped_right = resize(cropped_right, new_height, new_width_right, interpolation='bilinear')
        # print("wanted length: ", bottom_right_corner[0][0] - top_left_corner[0][0])
        # print("summed length: ", cropped_left.shape[1] + cropped_right.shape[1])
        # print("cropped left: ", "width:", cropped_left.shape[1], "height:", cropped_left.shape[0])
        # # print(cropped_left)
        # print("cropped right: ", "width:", cropped_right.shape[1], "height:", cropped_right.shape[0])
        # # print(cropped_right)
        full = np.concatenate((cropped_left, cropped_right), axis=1)
        if full.shape[1] < bottom_right_corner[0][0] - top_left_corner[0][0]:
            full = np.concatenate((full, [[100]]), axis=1)
        # elif full.shape[1] > bottom_right_corner[0][0] - top_left_corner[0][0]:
        #     full = full[0:1, 0: full.shape[1] - 2]
        if concated_image is None:
            concated_image = full
        else:
            concated_image = np.concatenate((concated_image, full), axis=0)

    left = img[0:img.shape[0],
           0:  top_left_corner[0][0]]
    right = img[0:img.shape[0],
            bottom_right_corner[0][0]:  img.shape[1]]
    top = img[0:top_left_corner[0][1],
          top_left_corner[0][0]: bottom_right_corner[0][0]]
    bottom = img[bottom_right_corner[0][1]:img.shape[0],
             top_left_corner[0][0]: bottom_right_corner[0][0]]
    middle = np.concatenate((top, concated_image), axis=0)
    middle = np.concatenate((middle, bottom), axis=0)
    full = np.concatenate((left, middle), axis=1)
    full = np.concatenate((full, right), axis=1)
    cv2.polylines(full, [parab_pts], False, (255, 0, 0), 3)
    cv2.imshow("new", full)
    cv2.waitKey(0)
    return full


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
    global image, relative_path, original_image, checking_image
    image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(relative_path, cv2.IMREAD_GRAYSCALE)
    checking_image = cv2.imread(relative_path, cv2.IMREAD_COLOR)


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
        # x = bilinear_my_way(image, 500, 500)
        # y = bilinear_interpolation(image, 500, 500)
        # cv2.imshow("My way", x)
        # cv2.imshow("other", y)
        # cv2.waitKey(0)
        new_deformat(original_image)
        # print(x)

        # x = cubic_interpolation(image, (1, 1.2))

    else:
        print("False pathname")
