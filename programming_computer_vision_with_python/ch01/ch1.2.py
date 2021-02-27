from PIL import Image
from pylab import *


def example1():
    # 读取图像到数组中
    im = array(Image.open('../resource/picture/empire.jpg'))

    # 绘制图像
    imshow(im)

    # 一些点
    x = [100, 100, 400, 400]
    y = [200, 500, 200, 500]

    # 使用红色星状标记绘制点
    plot(x, y, 'r*')

    # 绘制连接前两个点的线
    plot(x[:2], y[:2])

    # 添加标题，显示绘制的图像
    title('Plotting:, "empir.jpg"')
    # 不显示坐标轴
    axis('off')

    show()


def example2():
    # 读取图像到数组中
    im = array(Image.open('../resource/picture/empire.jpg').convert('L'))

    # 新建一个图像
    figure()
    # 不使用颜色信息
    gray()
    # 在原点的左上角显示轮廓图像
    contour(im, origin='image')
    axis('equal')
    axis('off')

    figure()
    # 绘制该灰度图像的直方图
    hist(im.flatten(), 128)
    show()


def example3():
    im = array(Image.open('../resource/picture/empire.jpg'))
    imshow(im)
    print('Please click 3 points')
    x = ginput(3)
    print('you click:', x)
    show()


if __name__ == '__main__':
    # example1()
    example2()
    # example3()
