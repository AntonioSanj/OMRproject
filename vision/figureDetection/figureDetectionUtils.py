import numpy as np


def filterClosePoints(points, min_distance=5):
    filtered_points = []  # List to store selected points

    for current_point in points:
        too_close = False

        # search in filtered points if there is one that is too close to the current point
        for existing_point in filtered_points:
            distance = np.linalg.norm(np.array(current_point) - np.array(existing_point))
            if distance <= min_distance:
                too_close = True
                break

        if not too_close:  # if there were no close points, add the current point
            filtered_points.append(current_point)

    return filtered_points
