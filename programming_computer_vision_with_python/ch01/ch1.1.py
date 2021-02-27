from PIL import Image


# 示例
def example():
    pil_im = Image.open('../resource/picture/empire.jpg')
    pil_im.show()

    # 转换成灰度图像
    pil_im_L = Image.open('../resource/picture/empire.jpg').convert('L')
    pil_im_L.show()

    # 创建缩略图
    pil_im.thumbnail((128, 128))

    # 复制和粘贴图像区域
    box = (100, 100, 400, 400)
    region = pil_im.crop(box)
    region = region.transpose(Image.ROTATE_180)
    pil_im.paste(region, box)

    # 调整尺寸
    out = pil_im.resize((128, 128))
    out.show()

    # 旋转
    out = pil_im.rotate(45)
    out.show()


if __name__ == '__main__':
    example()
