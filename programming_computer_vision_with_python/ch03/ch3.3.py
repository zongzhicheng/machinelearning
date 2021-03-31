import programming_computer_vision_with_python.ch03.sift as sift


# 使用SIFT特征自动找到匹配对应
def main1():
    featname = ['Univ' + str(i + 1) + '.sift' for i in range(5)]
    imname = ['Univ' + str(i + 1) + '.jpg' for i in range(5)]
    l = {}
    d = {}
    for i in range(5):
        sift.process_image(imname[i], featname[i])
        l[i], d[i] = sift.read_features_from_file(featname[i])

    matches = {}
    for i in range(4):
        matches[i] = sift.match(d[i + 1], d[i])


if __name__ == '__main__':
    main1()
