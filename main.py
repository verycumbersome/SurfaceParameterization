import os
import pdb
import numpy as np
import glfw
import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from PIL import Image

# Import opengl shapes from file
import shapes
import utils

WINDOW_W = 1500
WINDOW_H = 1500

poly_index = 3
display_mode = 5


# Load all poly files
path = "./ply_files/"
# poly_list = [utils.read_ply(path + x) for x in os.listdir(path)]
poly_list = [utils.read_ply(path + "sphere.ply")]

# Shape functions
cube = utils.render_shape(4, 4, shapes.cylinder)
cylinder = utils.render_shape(10, 10, shapes.cylinder)
sphere = utils.render_shape(10, 10, shapes.sphere, 4)
vase = utils.render_shape(10, 10, shapes.vase)


def lighting():
    brightness = 20
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glLightfv(GL_LIGHT0, GL_POSITION, [500, 100, -100, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [brightness, brightness, brightness, 1])
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    return


def read_texture(filename):
    _, file_extension = os.path.splitext(filename)
    img = Image.open(filename)
    img_data = np.array(list(img.getdata()), np.int8)
    texture_id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    if (file_extension == ".jpg"):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img_data)

    if (file_extension == ".png"):
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.size[0], img.size[1], 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    return texture_id


def init():
    glClearColor(0.2, 0.2, 0.2, 0.0)
    glShadeModel(GL_FLAT)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)

    # Enable vertex buffers
    glEnableClientState(GL_VERTEX_ARRAY)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # read_texture("textures/texture2.jpg")


def keyboard(bkey, x, y):
    global display_mode
    global poly_index

    key = bkey.decode("utf-8")
    if key == chr(27):
        os._exit(0)

    if key == "1":
        display_mode = 1

    if key == "2":
        display_mode = 2

    if key == "3":
        display_mode = 3

    if key == "4":
        display_mode = 4

    if key == "5":
        display_mode = 5

    if key == "6":
        display_mode = 6

    if key == "n":
        poly_index += 1

    if key == "p":
        poly_index -= 1

    if key == "q":
        os._exit(0)


def display():
    # Init and reset frame
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    # Set Projection window view
    glViewport(0, 0, WINDOW_H, WINDOW_W)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (WINDOW_H/WINDOW_W), 0.1, 50.0)
    glLoadIdentity()

    # Set poly
    poly = poly_list[poly_index % len(poly_list)]

    scale = poly.get_scale() if display_mode == 1 else 10

    glOrtho(-scale, scale, -scale, scale, -100000.0, 100000.0)

    # Set Model view
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_CULL_FACE)

    if display_mode == 1:
        glRotatef(1, 1, 1, 1)
        poly.render_mesh()
    else:
        glDisable(GL_CULL_FACE)
    if display_mode == 2:
        glRotatef(0.01, 1, 1, 1)
        shapes.plane(5)
    if display_mode == 3:
        glRotatef(0.2, 1, 1, 1)
        cube.render_mesh()
    if display_mode == 4:
        glRotatef(1, 1, 1, 1)
        cylinder.render_mesh()
    if display_mode == 5:
        glRotatef(1, 1, 1, 1)
        sphere.render_mesh()
    if display_mode == 6:
        glRotatef(1, 1, 1, 1)
        vase.render_mesh()
    glFlush()


def main():
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(WINDOW_H, WINDOW_W)
    glutInitWindowPosition(100, 100)
    wind = glutCreateWindow("Surface Parameterization")

    # lighting()

    init()

    glutDisplayFunc(display)
    glutIdleFunc(display)

    glutKeyboardFunc(keyboard)
    glutMainLoop()


if __name__ == "__main__":
    main()
