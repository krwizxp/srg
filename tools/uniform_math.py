import math
import numpy as np

DEGREES = 0.017453292519943295
ECCENTRICITY_SQUARED = 0.006694380022900787
SURFACE_SCALE = 12309072078.888908
SIN_CENTER = 0.5877852522924731
COS_CENTER = 0.8090169943749475
ENVELOPE_FACTOR = 1.0000000000009095


def density(latitude):
    delta = (latitude - 36.0) * DEGREES
    square = delta * delta
    sine_delta = delta * (1.0 + square * (-0.16666666666666666 + square * (0.008333333333333333 + square * (-0.0001984126984126984 + square * (2.7557319223985893e-6 + square * -2.505210838544172e-8)))))
    cosine_delta = 1.0 + square * (-0.5 + square * (0.041666666666666664 + square * (-0.001388888888888889 + square * (2.48015873015873e-5 + square * (-2.755731922398589e-7 + square * 2.08767569878681e-9)))))
    sine = SIN_CENTER * cosine_delta + COS_CENTER * sine_delta
    cosine = COS_CENTER * cosine_delta - SIN_CENTER * sine_delta
    denominator = 1.0 - ECCENTRICITY_SQUARED * sine * sine
    return cosine / (denominator * denominator)


def product_error(a, b, product):
    c = 134217729.0 * a
    ah = c - (c - a)
    al = a - ah
    c = 134217729.0 * b
    bh = c - (c - b)
    bl = b - bh
    return ((ah * bh - product) + ah * bl + al * bh) + al * bl


def determinant(a, b, c, d):
    left = a * d
    right = b * c
    expansion = []
    for value in [product_error(a, d, left), left, -product_error(b, c, right), -right]:
        result = []
        for entry in expansion:
            total = value + entry
            virtual = total - value
            error = (value - (total - virtual)) + (entry - virtual)
            if error != 0.0:
                result.append(error)
            value = total
        if value != 0.0 or not result:
            result.append(value)
        expansion = result
    return sum(expansion)


def proposal_weights(points):
    a = points[:, 1, 1] - points[:, 0, 1]
    b = points[:, 1, 0] - points[:, 0, 0]
    c = points[:, 2, 1] - points[:, 0, 1]
    d = points[:, 2, 0] - points[:, 0, 0]
    left = a * d
    right = b * c
    det = (left - right) + (product_error(a, d, left) - product_error(b, c, right))
    uncertain = np.abs(det) <= (np.abs(left) + np.abs(right)) * 1e-10
    for i in np.flatnonzero(uncertain):
        det[i] = determinant(float(a[i]), float(b[i]), float(c[i]), float(d[i]))
    area = np.abs(det) * 0.5
    envelope = area * SURFACE_SCALE * density(points[:, :, 0].min(axis=1)) * ENVELOPE_FACTOR * 1000000.0
    if not np.all(np.isfinite(envelope)) or not np.all(envelope > 0):
        raise ValueError('Invalid triangle proposal envelope')
    return np.ceil(envelope).astype(np.uint64) + 2
