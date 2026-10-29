import math

import numpy as np


ECCENTRICITY_SQUARED = 0.006694380022900787
SEMIMAJOR_AXIS = 6378137.0


def authalic_integral(latitude):
    sine = np.sin(np.deg2rad(latitude))
    eccentricity = math.sqrt(ECCENTRICITY_SQUARED)
    return sine / (2 * (1 - ECCENTRICITY_SQUARED * sine * sine)) + np.arctanh(
        eccentricity * sine
    ) / (2 * eccentricity)


def surface_area(mask, order=16):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    positions = (nodes + 1) / 2
    weights = weights / 2
    scale = SEMIMAJOR_AXIS ** 2 * (1 - ECCENTRICITY_SQUARED) * math.pi / 180
    polygons = [mask] if mask.geom_type == 'Polygon' else list(mask.geoms)
    result = 0.0
    for polygon in polygons:
        for index, ring in enumerate([polygon.exterior, *polygon.interiors]):
            coordinates = np.asarray(ring.coords)
            total = 0.0
            for begin in range(0, len(coordinates) - 1, 65536):
                start = coordinates[begin:begin + 65536]
                end = coordinates[begin + 1:begin + 65537]
                start = start[:len(end)]
                mean = np.zeros(len(start))
                for position, weight in zip(positions, weights, strict=True):
                    mean += weight * authalic_integral(
                        start[:, 1] + position * (end[:, 1] - start[:, 1])
                    )
                total += float(np.sum((end[:, 0] - start[:, 0]) * mean))
            result += (1 if index == 0 else -1) * abs(total) * scale
    return result


def rectangle_surface_area(bounds):
    west, south, east, north = bounds
    scale = SEMIMAJOR_AXIS ** 2 * (1 - ECCENTRICITY_SQUARED) * math.pi / 180
    return float((east - west) * (authalic_integral(north) - authalic_integral(south)) * scale)
