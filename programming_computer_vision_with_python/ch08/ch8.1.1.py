from pylab import *
import pickle
import programming_computer_vision_with_python.ch08.knn as knn
from numpy import *
from PCV.tools import imtools

# 用pickle模块载入二维数据点
with open('points_normal.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

model = knn.KnnClassifier(labels, vstack((class_1, class_2)))

# 用pickle模块载入测试数据
with open('points_normal_test.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# 在测试数据集的第一个数据点上进行测试
print(model.classify(class_1[0]))


# 定义绘图函数
def classify(x, y, model=model):
    return array([model.classify([xx, yy]) for (xx, yy) in zip(x, y)])


# 绘制分解边界
imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])
show()
