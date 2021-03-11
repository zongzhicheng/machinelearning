from PIL import Image
from pylab import *
from numpy import *
from numpy import random
from scipy.ndimage import filters


# ROF模型去噪
def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    m, n = im.shape  # 噪声图像的大小
    # 初始化
    U = U_init
    Px = im  # 对偶域的x分量
    Py = im  # 对偶域的y分量
    error = 1
    while (error > tolerance):
        Uold = U

        # 原始变量的梯度
        GradUx = roll(U, -1, axis=1) - U  # 变量U梯度的x分量
        GradUy = roll(U, -1, axis=0) - U  # 变量U梯度的y分量

        # 更新对偶变量
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew  # 更新x分量（对偶）
        Py = PyNew / NormNew  # 更新y分量（对偶）

        # 更新原始变量
        RxPx = roll(Px, 1, axis=1)  # 对x分量进行向右x轴平移
        RyPy = roll(Py, 1, axis=0)  # 对y分量进行向右y轴平移

        DivP = (Px - RxPx) + (Py - RyPy)  # 对偶域的散度
        U = im + tv_weight * DivP  # 更新原始变量

        # 更新误差
        error = linalg.norm(U - Uold) / sqrt(n * m)

        return U, im - U  # 去噪后的图像和纹理残余


if __name__ == '__main__':
    # 使用噪声创建合成图像
    # im = zeros((500, 500))
    # im[100:400, 100:400] = 128
    # im[200:300, 200:300] = 255
    # im += 30 * random.standard_normal((500, 500))
    #
    # U, T = denoise(im, im)
    # G = filters.gaussian_filter(im, 10)
    #
    # imshow(im)  # 原始噪声图像
    # show()
    # imshow(G)  # 高斯模糊的图像
    # show()
    # imshow(U)  # ROF模型去噪的图像
    # show()

    # 实际图像使用ROF模型去噪
    real_im = array(Image.open('../resource/picture/empire.jpg').convert('L'))
    real_U, real_T = denoise(real_im, real_im)
    figure()
    gray()
    imshow(real_U)
    axis('equal')
    axis('off')
    show()
