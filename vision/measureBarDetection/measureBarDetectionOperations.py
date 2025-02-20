import math

import cv2
import numpy as np

from constants import RED


def sobelFilter(grayImage, dx=1, dy=0):
    return cv2.Sobel(grayImage, cv2.CV_8U, dx=dx, dy=dy, ksize=3)


def getVerticalLines(edges, threshold, minLineLength, maxLineGap, angle_tolerance=10):
    angle = np.pi / 180
    lines = cv2.HoughLinesP(edges, 1, angle, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Compute angle in degrees
            dx = x2 - x1
            dy = y2 - y1
            line_angle = np.degrees(np.arctan2(abs(dy), abs(dx)))

            # Check if the line is approximately vertical
            if 90 - angle_tolerance <= line_angle <= 90 + angle_tolerance:
                vertical_lines.append(line)

    return vertical_lines


def filterVerticalEdges(image):
    kernel = np.ones((3, 1), np.uint8)

    # this dilation fills vertical line gaps
    image = cv2.dilate(image, kernel, iterations=1)

    kernel = np.ones((100, 1), np.uint8)

    # exclude short lines
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((1, 3), np.uint8)
    # widening lines eases detection
    image = cv2.dilate(image, kernel, iterations=1)

    return image


def mergeLines(lines):

    tolerance = 250  # proximity in pixels

    # list of midpoints
    midpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        length = abs(y2 - y1)
        midpoints.append((midpoint_x, midpoint_y, line, length))

    # sort midpoints by their x-coordinate (to help with the initial grouping)
    midpoints.sort(key=lambda x: x[0])

    # create group by proximity of midpoints
    groups = []

    for i in range(len(midpoints)):

        curr_midp_x, curr_midp_y, curr_line, curr_length = midpoints[i]

        # find an existing group for the current line
        group_found = False
        for group in groups:
            # compute group's centroid
            group_midp_x = sum(line[0] for line in group) / len(group)
            group_midp_y = sum(line[1] for line in group) / len(group)

            # compare distance between current line midpoint and group centroid
            distance = math.sqrt((curr_midp_x - group_midp_x) ** 2 + (curr_midp_y - group_midp_y) ** 2)

            if distance <= tolerance:
                # current group close enough -> add to current group
                group.append((curr_midp_x, curr_midp_y, curr_line, curr_length))
                group_found = True
                break
        # no group found -> create new group
        if not group_found:
            groups.append([(curr_midp_x, curr_midp_y, curr_line, curr_length)])

    # find the longest line in each group
    longest_lines = []
    for group in groups:
        longest_line = max(group, key=lambda x: x[3])  # x[3] is length
        longest_lines.append(longest_line[2])  # longest_line[2] contains the line
    return longest_lines


def drawLines(lines, image, color=(0, 0, 255), thickness=2):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image
