import os
import pdb
import time
import math
import pprint
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
# from numpy.linalg import eig, eigh, eigvals
import numpy.linalg as LA
import mpmath

import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from OpenGL.raw.GL.ARB.vertex_array_object import glGenVertexArrays, \
                                                          glBindVertexArray

import topology
import utils

pp = pprint.PrettyPrinter(indent=4)


def plane(length):
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, (length / 2))
    glVertex3f((-length / 2), (-length / 2), 0.0)
    glTexCoord2d(1, 0)
    glVertex3f((-length / 2), (length / 2), 0.0)
    glTexCoord2d(0, 0)
    glVertex3f((length / 2), (length / 2), 0.0)
    glTexCoord2d(0, 1)
    glVertex3f((length / 2), (-length / 2), 0.0)

    glEnd()


def cylinder(u, v):
    """ Function that returns 3d coordinates for a cylinder from 2d input.

    Input:
        (u, v) -> (height[0-1], theta[0-1])

    Ouput:
        (x, y, z) coordinate
    """
    radius = 4.0
    height = 10.0
    v = math.radians(v * 360.0)

    return(
        radius * math.cos(v),
        radius * math.sin(v),
        u * height
    )


def vase(u, v):
    """ Function that returns 3d coordinates for a vase from 2d input.

    Input:
        (u, v) -> (height[0-1], theta[0-1])

    Ouput:
        (x, y, z) coordinate
    """
    radius = 4.0

    v = math.radians(v * 360.0)
    u_rad = math.radians(u * 360.0)

    return(
        radius * math.cos(u_rad) * math.cos(v),
        radius * math.cos(u_rad) * math.sin(v),
        (2 * radius * u) - radius
    )


def sphere(u, v, radius=2.0):
    """ Function that returns 3d coordinates for a cylinder from 2d input.

    Arguments:
        (u, v) -> (height[0-1], theta[0-1])

    Returns:
        (x, y, z) coordinate
    """

    v = math.radians(v * 360.0)
    d = math.sqrt(radius ** 2 - u ** 2)

    return(
        radius * d * math.sin(v),
        radius * d * math.cos(v),
        u * radius
    )


@dataclass
class vertex:
    """Vertex class to help store information about 3D vertices.

    Note:
        Only works for 3D vertices and is mainly used for loading PLY files

    Args:
        x(float): X coordinate for the 3d vertex
        y(float): Y coordinate for the 3d vertex
        z(float): Z coordintate for the 3d vertex

    Attributes:
        x(float): X coordinate for the 3d vertex
        y(float): Y coordinate for the 3d vertex
        z(float): Z coordintate for the 3d vertex
        rgb(dict): dict of {"r":0, "g":0, "b":0}
    """
    id: int
    # XYZ coordinates for the vertex
    x: float
    y: float
    z: float

    # Size 3 tuple of rgb: rgb[0] = red_val, etc..
    rgb: tuple

    # Stores xyz in np array
    coords: np.ndarray = np.zeros(shape=3)

    def __post_init__(self):
        self.coords = np.array([self.x, self.y, self.z])

    def __getitem__(self, index):
        return({
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
        }[index])


@dataclass
class edge:
    v1: vertex
    v2: vertex


@dataclass
class face:
    """Face class to help store information about face properties.

    Note:
        This stores vertices as a list of Vertex objects

    Args:
        count(int): The number of vertices in the face
        vertices(list): The list of vertices for the face

    Attributes:
        count(int): The number of vertices in the face
        vertices(list): The list of vertices for the face
        edges(list): The list of edges for the face
    """
    # Number of vertices and list of all vertices for the face
    count: int
    vertices: list

    edges: list = field(repr=False, default_factory=list)

    def __post_init__(self):
        self.calc_normal()

    def calc_normal(self):
        if (self.count == 3):
            pass

        v1 = self.vertices[1].coords - self.vertices[0].coords  # vertex 0-1
        v2 = self.vertices[2].coords - self.vertices[0].coords  # vertex 0-2

        self.normal = np.cross(v1, v2)

    def render_face(self):
        # If face is a triangle
        if (self.count == 3):
            glPointSize(5.0)
            # glBegin(GL_POINTS)
            glBegin(GL_TRIANGLE_STRIP)
            for v in self.vertices:
                glColor3f(v.rgb["r"], v.rgb["g"], v.rgb["b"])
                # glTexCoord2f(0.0, 0.0)
                # glNormal3f(*self.normal);
                glVertex3fv((v.x, v.y, v.z))
            glEnd()

        # If face is a quad
        if (self.count == 4):
            glPointSize(5.0)
            glBegin(GL_POINTS)
            # glBegin(GL_QUAD_STRIP)
            for v in self.vertices:
                glColor3f(v.rgb["r"], v.rgb["g"], v.rgb["b"])
                glNormal3f(*self.normal)
                glVertex3fv((v.x, v.y, v.z))
            glEnd()


@dataclass
class poly:
    """ The poly class store information about an entire PLYfile object

    Note:
        This stores vertices as a list of Vertex objects

    Args:
        vertices(list): The list of vertices for the polygon
        faces(list): The list of faces for the polygon

    Attributes:
        vertices(list): The list of vertices for the polygon
        faces(list): The list of faces for the polygon
        num_vertices(int): The number of vertices for the polygon
        num_faces(int): The number of faces for the polygon
        edges(list): The list of edges for the polygon
        edge_graph(dict): An edge graph for each vertex showing connection
    """
    vertices: list
    faces: list

    num_faces: int = 0
    num_vertices: int = 0

    edges: list = field(repr=False, default_factory=list)
    triangle_pairs: list = field(repr=False, default_factory=list)
    edge_graph: dict = field(repr=False, default_factory=dict)

    def __post_init__(self):
        self.num_faces = len(self.faces)
        self.num_vertices = len(self.vertices)
        self.get_scale()
        self.get_edges()
        self.get_boundary()

        # pp.pprint(self.edge_graph)

        if(self.triangle_pairs):
            self.calc_dirichlet()

    def calc_dirichlet(self):
        for i, vi in enumerate(self.vertices):  # For all verts in poly
            for vj in self.edge_graph[vi.id]:   # for all neightbors of vert
                j = self.vertices.index(vj)
                num_edges = len(self.edge_graph[vi.id])

                # print(self.edge_graph[vi.id])
                # print(num_edges)

                # tri1 = self.triangle_pairs[i]["2D"]
                # tri2 = self.triangle_pairs[(j - 1) % num_edges]["2D"]
                # tri3 = self.triangle_pairs[j]["2D"]
                # tri4 = self.triangle_pairs[(j + 1) % num_edges]["2D"]
                # print(tri1)
                # print(tri2)
                # print(tri3)
                # print(tri4)
                # print()


        # W = []
        # for f in self.faces:
            # vi1 = f.vertices[0].coords
            # vi = f.vertices[1].coords

            # vj = f.vertices[2].coords
            # vj1 = f.vertices[3].coords

            # fvi = np.array(self.triangle_pairs[-1]["2D"])[0]
            # fvj = np.array(self.triangle_pairs[-2]["2D"])[1]

            # aij = topology.calc_theta(vj - vj1, vi - vj1)
            # bij = topology.calc_theta(vi - vi1, vj - vi1)

            # wij = mpmath.cot(aij) + mpmath.cot(bij)

            # W.append(wij * (fvi - fvj))

        # print(W)

    def get_boundary(self):
        # TODO get vertices where number of connections == 3
        rgb = {"r":0.0,"g":0.0,"b":0.0,}
        self.boundary = []
        self.faces[0].vertices[1].rgb = rgb
        for i, f in enumerate(self.faces):
            for j, v in enumerate(f.vertices):
                if len(self.edge_graph[v.id]) == 3:
                    self.faces[i].vertices[j].rgb = rgb
                    self.boundary.append(v)

    def get_edges(self):
        """Returns a graph of all edges with vertices as the key"""
        self.edge_graph = {k: [] for k in range(self.num_vertices)}

        # Loops through each poly face to connect all vertices as edges
        for f in self.faces:
            for i, v in enumerate(f.vertices):
                # Connect current vertex in face to next indexed vertex in face
                self.edge_graph[v.id].append(f.vertices[(i + 1) % f.count])
                self.edge_graph[f.vertices[(i + 1) % f.count].id].append(v)

                # Get the edge as a set and add to respective face
                edge = (v, f.vertices[(i + 1) % f.count])
                self.edges.append(edge)
                f.edges.append(edge)

                # pdb.set_trace()


    def get_scale(self):
        """Gets the bounds and OpenGL scale factor for poly"""
        self.bounds = [
            max(self.vertices, key=lambda v: v.x).x,
            max(self.vertices, key=lambda v: v.y).y,
            max(self.vertices, key=lambda v: v.z).z,
            min(self.vertices, key=lambda v: v.x).x,
            min(self.vertices, key=lambda v: v.y).y,
            min(self.vertices, key=lambda v: v.z).z,
        ]
        self.scale = max(self.bounds) - min(self.bounds)

        return(self.scale)

    def render_mesh(self):
        for f in self.faces:
            f.render_face()
