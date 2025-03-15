import numpy as np


def filterClosePoints(points, min_distance=5):
    # x, y, opt: w, opt:h
    # sometimes points might include the height and width of the template with which that specific point was detected
    # this function while still outputting all the data in the point does not work with the width and the height
    # it is just information meant to be kept

    filtered_points = []
    for current_point in points:
        x1, y1 = current_point[:2]  # extract (x, y) and ignore w and h if present
        too_close = False

        # search in filtered points if there is one that is too close to the current point
        for existing_point in filtered_points:
            x2, y2 = existing_point[:2]  # compare only x and y
            distance = np.linalg.norm([x1 - x2, y1 - y2])

            if distance <= min_distance:
                too_close = True
                break

        if not too_close:  # if there were no close points, add the current point
            filtered_points.append(current_point)

    return filtered_points
