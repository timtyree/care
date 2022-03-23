# geometry.py
# Programmer: Tim Tyree
# Date: 3.23.2022
import numpy as np

def translate_then_rotate(x,y,x0,y0,theta):
    """x,y are numpy.ndarray instances of equal shape.
    x0, y0, and theta are floats denoting the new origin and the angle to rotate in radians.

    Example Usage:
x_translated_rotated,y_translated_rotated=translate_then_rotate(x,y,x0,y0,theta)
    """
    x_translated=x-x0
    y_translated=y-y0
    x_translated_rotated=np.cos(theta)*x_translated-np.sin(theta)*y_translated
    y_translated_rotated=np.cos(theta)*y_translated+np.sin(theta)*x_translated
    return x_translated_rotated,y_translated_rotated
