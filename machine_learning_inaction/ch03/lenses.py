from machine_learning_inaction.ch03 import trees, treePlotter

# 使用决策树预测隐形眼镜类型
if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)
