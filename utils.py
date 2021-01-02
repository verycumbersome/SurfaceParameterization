import os
import pdb
import math
import glfw
import numpy as np
import OpenGL
import pprint

import mpmath

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from ctypes import *

from numpy import linalg as LA

import math_utils
import shapes

pp = pprint.PrettyPrinter(indent=4)

def calc_normals(quad_mat, i):
    """Calculate normals for a quad"""
    Vu = quad_mat[(i + 1) % 4] - quad_mat[i]
    Vv = quad_mat[(i + 2) % 4] - quad_mat[i]
    normal_x = Vu[1] * Vv[2] - Vu[2] * Vv[1]
    normal_y = Vu[2] * Vv[0] - Vu[0] * Vv[2]
    normal_z = Vu[0] * Vv[1] - Vu[1] * Vv[0]

    return normal_x, normal_y, normal_z


def read_ply(filename):
    """
    Read and parse a PLY file
    Output: PLY(vertices, faces) - vertices and faces parsed from file
    """
    with open(filename, "r") as ply_file:
        header = ply_file.read().split("end_header")
        header, data = header[0], header[1][1:].split("\n")

        n_verts = [x for x in header.split("\n") if "element vertex" in x][0]
        n_faces = [x for x in header.split("\n") if "element face" in x][0]

        n_verts = int(n_verts.split()[-1])
        n_faces = int(n_faces.split()[-1])

        vertices = []
        for i in range(n_verts):
            v_info = data.pop(0).split()

            rgb = {"r": 1.0, "g": 1.0, "b": 1.0}

            v = shapes.vertex(
                    i,
                    float(v_info[0]),    # x
                    float(v_info[1]),    # y
                    float(v_info[2]),    # z
                    rgb=rgb,             # rgb
                )
            vertices.append(v)

        # for v in vertices[-30:]:
            # v.rgb = {"r": 1.0, "g": 1.0, "b": 1.0}

        faces = []
        for i in range(n_faces):
            f_info = data.pop(0).split()

            # Adds face to poly
            if (int(f_info[0]) == 3):  # if the poly is a triangle
                f = shapes.face(
                        int(f_info[0]),
                        [
                            vertices[int(f_info[1])],  # v1
                            vertices[int(f_info[2])],  # v2
                            vertices[int(f_info[3])],  # v3
                        ]
                    )
            elif (int(f_info[0]) == 4):  # if the poly is a quad
                f = shapes.face(
                        int(f_info[0]),
                        [
                            vertices[int(f_info[1])],  # v1
                            vertices[int(f_info[2])],  # v2
                            vertices[int(f_info[3])],  # v3
                            vertices[int(f_info[4])],  # v4
                        ]
                    )
            faces.append(f)

        return(shapes.poly(vertices, faces))


def render_shape(z_res, xy_res, shape_func, multiplier=1):
    """
    Function to render OpenGL shape given function of form T(u, v)
    Input:  xy_res -> Resolutions of the xy for shape
            z_res -> Resolutions of the z for shape
    Output: renders OpenGl shape
    """
    # Get step size given subdivision
    z_step = (1 / z_res)
    xy_step = (1 / xy_res)

    # Get inputs for cylinder function
    z_res = round(z_res / 2) * multiplier
    # z_range = [x * z_step for x in range(-z_res, z_res)]
    xy_range = [x * xy_step for x in range(0, xy_res)]
    z_range = [x * xy_step for x in range(0, z_res)]

    vertices = []
    faces = []
    triangle_pairs = []

    W = [[0 for i in range(z_res)] for j in range(xy_res)]

    for index_u, u in enumerate(z_range):
        for index_v, v in enumerate(xy_range):
            # Matrix for storing traingle strip vertex coordinates
            # [1][2] - Top left | Top Right
            # [0][3] - Bottom left | Bottom right
            quad_idx = [
                (u, v), (u, v + z_step),
                (u + xy_step, v + z_step), (u + xy_step, v),
            ]
            quad_mat = np.array([
                shape_func(*quad_idx[0]),
                shape_func(*quad_idx[1]),
                shape_func(*quad_idx[2]),
                shape_func(*quad_idx[3]),
            ])

            # Add triangle_pairs to array to calculate dirichlet
            # Triangle 1: | Triangle 2:
            #   [1][ ]    |   [1][2]
            #   [0][3]    |   [ ][3]
            for i in range(2):
                triangle_pairs.append({
                    "2D": (quad_idx[1], quad_idx[2 * i], quad_idx[3]),
                    "3D": (quad_mat[1], quad_mat[2 * i], quad_mat[3]),
                })

                # Set default color for poly
                rgb = {"r": 1.0, "g": 1.0, "b": 1.0}

                # For the last vertex added to triangle pair
                for v in triangle_pairs[-1]["3D"]:
                    vert = shapes.vertex(
                            len(vertices),  # Index of vert
                            *v,             # xyz
                            rgb=rgb,        # rgb
                        )
                    vertices.append(vert)

            # 3 denotes the amount of vertices for each face
            tri1 = shapes.face(3, vertices[-6:-3])
            tri2 = shapes.face(3, vertices[-3:])
            faces.append(tri1)
            faces.append(tri2)


    return(shapes.poly(vertices, faces, triangle_pairs=triangle_pairs))
