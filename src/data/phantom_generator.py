"""
Phantom image generation utilities for KTC2023 EIT training data.

Generates random 256x256 phantom images with 3 classes:
  0: Background
  1: Resistive inclusion
  2: Conductive inclusion

Extracted from: KTC2023_SubmissionFiles/ktc_training/src/dataset/SimDataset.py
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import rotate


def create_phantoms(min_inclusions=1, max_inclusions=4, max_iter=80,
                    distance_between=25, p=None):
    """Generate a random 256x256 phantom image with values {0, 1, 2}.

    Creates random inclusions (polygons, circles, rectangles) inside
    a circular domain mask. Each inclusion is randomly assigned class
    1 (resistive) or 2 (conductive).

    Args:
        min_inclusions: Minimum number of inclusions to place.
        max_inclusions: Maximum number of inclusions to place.
        max_iter: Maximum placement attempts before giving up.
        distance_between: Minimum pixel distance between inclusion centers.
        p: Probability weights for [polygon, circle, rectangle] types.

    Returns:
        np.ndarray: 256x256 array with integer values in {0, 1, 2}.
    """
    if p is None:
        p = [0.7, 0.15, 0.15]

    rectangle_dict = {
        'min_width': 25, 'max_width': 50,
        'min_height': 40, 'max_height': 120,
    }

    pixwidth = 0.23 / 256
    pixcenter_x = np.linspace(
        -0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing='ij')

    I = np.zeros((256, 256))
    im = Image.fromarray(np.uint8(I))
    draw = ImageDraw.Draw(im)

    num_forms = np.random.randint(min_inclusions, max_inclusions)
    circle_list = []
    iteration = 0

    while len(circle_list) < num_forms:
        object_type = np.random.choice(['polygon', 'circle', 'rectangle'], p=p)

        if object_type == 'rectangle':
            lower_x = 50 + np.random.randint(-24, 24)
            lower_y = 50 + np.random.randint(-24, 24)
            width = np.random.randint(
                rectangle_dict['min_width'], rectangle_dict['max_width'])
            height = np.random.randint(
                rectangle_dict['min_height'], rectangle_dict['max_height'])
            center_x = lower_x + width / 2
            center_y = lower_y + height / 2
            avg_radius = max(width / 2, height / 2)
        else:
            avg_radius = np.random.randint(25, 50)
            center_x = 128 + np.random.randint(-54, 54)
            center_y = 128 + np.random.randint(-54, 54)

        # Collision detection
        collide = False
        for x, y, r in circle_list:
            d = (center_x - x) ** 2 + (center_y - y) ** 2
            if d < (avg_radius + r + distance_between) ** 2:
                collide = True
                break

        if not collide:
            fill_val = 1 if np.random.rand() < 0.5 else 2

            if object_type == 'rectangle':
                draw.rectangle(
                    [lower_x, lower_y, lower_x + width, lower_y + height],
                    fill=fill_val)
            elif object_type == 'circle':
                draw.ellipse(
                    (center_x - avg_radius, center_y - avg_radius,
                     center_x + avg_radius, center_y + avg_radius),
                    fill=fill_val)
            elif object_type == 'polygon':
                num_vertices = np.random.randint(5, 9)
                vertices = generate_polygon(
                    center=(center_x, center_y),
                    avg_radius=avg_radius,
                    irregularity=0.4,
                    spikiness=0.3,
                    num_vertices=num_vertices)
                draw.polygon(vertices, fill=fill_val)

            circle_list.append((center_x, center_y, avg_radius))

        iteration += 1
        if iteration > max_iter:
            break

    sigma_pix = np.array(np.asarray(im))
    sigma_pix[X ** 2 + Y ** 2 > 0.098 ** 2] = 0.0
    angle = np.random.randint(0, 180)
    sigma_pix = np.round(
        rotate(sigma_pix, angle, mode='constant', cval=0.0,
               reshape=False, order=0))

    return sigma_pix


def generate_polygon(center, avg_radius, irregularity, spikiness,
                     num_vertices):
    """Generate a random polygon with the given parameters.

    Reference: https://stackoverflow.com/questions/8997099

    Args:
        center: (x, y) center of the polygon.
        avg_radius: Average distance of vertices from center.
        irregularity: Variance of angular spacing (0-1).
        spikiness: Variance of radial distance (0-1).
        num_vertices: Number of polygon vertices.

    Returns:
        List of (x, y) tuples for the polygon vertices.
    """
    if irregularity < 0 or irregularity > 1:
        raise ValueError('Irregularity must be between 0 and 1.')
    if spikiness < 0 or spikiness > 1:
        raise ValueError('Spikiness must be between 0 and 1.')

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = _random_angle_steps(num_vertices, irregularity)

    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = _clip(random.gauss(avg_radius, spikiness), 0,
                       2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points


def _random_angle_steps(steps, irregularity):
    """Generate random angular divisions for polygon vertices."""
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for _ in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def _clip(value, lower, upper):
    """Clip value to [lower, upper]."""
    return min(upper, max(value, lower))
