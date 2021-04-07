from OpenGL.raw.GLUT import glutSolidTeapot
from pylab import *
from numpy import *
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image
from pygame.locals import *
import pickle

width, height = 1000, 747


# 安装OpenGL
# pip install PyOpenGL-3.1.5-cp36-cp36m-win_amd64.whl
# 安装pygame
# pip install pygame

def set_projection_from_camera(K):
    """
    从照相机标定矩阵中获取视图
    :param K:
    :return:
    """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    fx = K[0, 0]
    fy = K[1, 1]
    fovy = 2 * arctan(0.5 * height / fy) * 180 / pi
    aspect = (width * fy) / (height * fx)

    # 定义近的和远的剪裁平面
    near = 0.1
    far = 100.0

    # 设定透视
    gluPerspective(fovy, aspect, near, far)
    glViewport(0, 0, width, height)


def set_modelview_from_camera(Rt):
    """
    从照相机姿态中获得模拟视图矩阵
    :param Rt:
    :return:
    """
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 围绕x轴将茶壶旋转90度，使z轴向上
    Rx = array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # 获得旋转的最佳逼近
    R = Rt[:, :3]
    U, S, V = linalg.svd(R)
    R = dot(U, V)
    R[0, :] = -R[0, :]  # 改变x轴的符号

    # 获得平移量
    t = Rt[:, 3]

    # 获得4x4的模拟试图矩阵
    M = eye(4)
    M[:3, :3] = dot(R, Rx)
    M[:3, 3] = t

    # 转置并压平以获取列序数值
    M = M.T
    m = M.flatten()

    # 将模拟试图矩阵替换为新的矩阵
    glLoadMatrixf(m)


def draw_background(imname):
    """
    使用四边形绘制背景图像
    :param imname:
    :return:
    """
    # 载入背景图像（应该是.bmp格式），转换为OpenGL纹理
    bg_image = pygame.image.load(imname).convert()
    bg_data = pygame.image.tostring(bg_image, "RGBX", 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 绑定纹理
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, glGenTextures(1))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # 创建四方形填充整个窗口
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glEnd()

    # 清除纹理
    glDeleteTextures(1)


def draw_teapot(size):
    """
    在原点处绘制红色茶壶
    :param size:
    :return:
    """
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_DEPTH_TEST)
    # 清除所有的像素
    glClear(GL_DEPTH_BUFFER_BIT)

    # 绘制红色茶壶
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0, 0, 0, 0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.5, 0.0, 0.0, 0.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.7, 0.6, 0.6, 0.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0)
    glutSolidTeapot(size)


def setup():
    """
    设置窗口和 pygame 环境
    :return:
    """
    pygame.init()
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)
    pygame.display.set_caption('OpenGL AR demo')


# 载入照相机数据
with open('ar_camera.pkl', 'rb') as f:
    K = pickle.load(f)
    Rt = pickle.load(open('ar_camera.pkl', 'rb'))
setup()
draw_background('../resource/picture/book_perspective.bmp')
set_projection_from_camera(K)
set_modelview_from_camera(Rt)
# 到这个地方会出现闪退
# 编译运行会报错：freeglut ERROR: Function <glutSolidTeapot> called without first calling 'glutInit'.）
# 这是因为freeglut和glut共存且定义了相同的方法，存在动态链接库重叠问题。
# 解决方案：
# 进入安装OpenGL相关路径 例如：D:\anaconda3\envs\TF_2C\Lib\site-packages\OpenGL\DLLS
# 删掉freeglut64.vc14.dll和gle64.vc14.dll，留下glut64.vc14.dll
# 参考链接：https://blog.csdn.net/weixin_43837871/article/details/89057837
draw_teapot(0.5)

while True:
    event = pygame.event.poll()
    if event.type in (QUIT, KEYDOWN):
        break
pygame.display.flip()
