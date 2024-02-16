import operator
from math import log
import matplotlib.pyplot as plt
import copy
import numpy as np
import random


# 输入数据集并初始化
def initdata(choice):
    # 输入1是鸢尾花数据集
    if choice == 1:
        iris_data = open(r'D:\学习资料\作业 论文\大四上\互联网+综合课设\决策树\数据集\iris\iris.data')
        init_data = [line.strip().split(',') for line in iris_data.readlines()]
        for i in init_data:
            for j in range(4):
                i[j] = float(i[j])

        train_iris = []
        test_iris = []
        flag = 3
        num_line = 1
        # 划分数据集，用shuffle打乱之后再划分会不稳定，所以采取顺序分配，每两个训练数据的下一个数据划分为测试数据
        for line in init_data:
            if num_line % flag != 0:
                train_iris.append(line)
                num_line += 1
            else:
                test_iris.append(line)
                num_line += 1

        return train_iris, test_iris

    # 输入2是贷款额度数据集
    elif choice == 2:
        adult_train = open(r'D:\学习资料\作业 论文\大四上\互联网+综合课设\决策树\数据集\adult\adult.data')
        init_data_train = [line.strip().split(',') for line in adult_train.readlines()]
        adult_test = open(r'D:\学习资料\作业 论文\大四上\互联网+综合课设\决策树\数据集\adult\adult.test')
        init_data_test = [line.strip().split(',') for line in adult_test.readlines()]
        for line in init_data_train:
            for j in [0, 2, 4, 10, 11, 12]:
                line[j] = float(line[j])
        for line in init_data_test:
            for j in [0, 2, 4, 10, 11, 12]:
                line[j] = float(line[j])

        return init_data_train, init_data_test

    else:
        print("错误，请输入正确的数字以选择数据集")


# 对连续数据进行清洗然后离散化
def discretize(choice, dataset):
    # 输入1是鸢尾花数据集，不用清洗只用离散化
    if choice == 1:
        for i in dataset:
            for j in range(4):
                if j == 0:
                    if 4 < i[j] <= 5:
                        i[j] = '<5'
                    elif 5 < i[j] <= 6:
                        i[j] = '5-6'
                    elif 6 < i[j] <= 7:
                        i[j] = '6-7'
                    else:
                        i[j] = '7<'
                if j == 1:
                    if i[j] <= 3:
                        i[j] = '<3'
                    elif 3 < i[j] <= 4:
                        i[j] = '3-4'
                    else:
                        i[j] = '4<'
                if j == 2:
                    if i[j] <= 2:
                        i[j] = '<2'
                    elif 2 < i[j] <= 3:
                        i[j] = '2-3'
                    elif 3 < i[j] <= 4:
                        i[j] = '3-4'
                    elif 4 < i[j] <= 5:
                        i[j] = '4-5'
                    else:
                        i[j] = '5<'
                if j == 3:
                    if i[j] <= 1:
                        i[j] = '<1'
                    elif 1 < i[j] <= 2:
                        i[j] = '1-2'
                    else:
                        i[j] = '2<'
        return dataset
    # 输入2是贷款额度数据集，需要处理缺失数据然后离散化
    elif choice == 2:
        for i in dataset:
            for j in range(14):
                # 缺失数据可以用下一个的来替换，尽管可以用更细致的方式预测，但是难度较高
                if i[j] == '?':
                    i[j] = i[j+1]
                # 年龄每十岁划分一次
                if j == 0:
                    if i[j] <= 20:
                        i[j] = '<20'
                    elif i[j] <= 30:
                        i[j] = '20-30'
                    elif i[j] <= 40:
                        i[j] = '30-40'
                    elif i[j] <= 50:
                        i[j] = '40-50'
                    elif i[j] <= 60:
                        i[j] = '50-60'
                    else:
                        i[j] = '60<'
                # "fnlwgt" 是美国人口普查数据中的一个字段，通常用于表示样本的权重或调查数据的加权值。具体来说：
                # "fnlwgt" 代表 "final weight"，也就是最终的样本权重每个样本中的个体都会被分配一个"fnlwgt"，这个值通常反映了这个个体在整个人口中的代表性或权重。
                if j == 2:
                    if i[j] <= 100000:
                        i[j] = '<10w'
                    elif i[j] <= 200000:
                        i[j] = '10w-20w'
                    elif i[j] <= 300000:
                        i[j] = '20w-30w'
                    elif i[j] <= 400000:
                        i[j] = '30w-40w'
                    else:
                        i[j] = '40w<'
                if j == 4:
                    if i[j] <= 9:
                        i[j] = '<9'
                    elif i[j] <= 14:
                        i[j] = '10-14'
                    else:
                        i[j] = '14<'
                if j == 10:
                    if i[j] == 0:
                        i[j] = '0'
                    else:
                        i[j] = '!0'
                if j == 11:
                    if i[j] == 0:
                        i[j] = '0'
                    else:
                        i[j] = '!0'
                if j == 12:
                    if i[j] <= 30:
                        i[j] = '<30'
                    elif i[j] <= 40:
                        i[j] = '30-40'
                    elif i[j] <= 50:
                        i[j] = '40-50'
                    elif i[j] <= 60:
                        i[j] = '60'
                    else:
                        i[j] = '60<'
        return dataset
    else:
        print("错误，请输入正确的数字以选择数据集")


# 计算信息熵
def informationentropy(dataset):
    num = len(dataset)
    labelCounts = {}
    for i in dataset:  # 遍历每个样本
        currentLabel = i[-1]  # 当前样本的类别
        if currentLabel not in labelCounts.keys():  # 生成类别字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 计算信息熵
    info = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num
        info = info - prob * log(prob, 2)

    return info


# 划分数据集，axis:按第几个属性划分，value:根据axis属性不同的value进行划分
def splitdataset(dataset, axis, value):
    spliteddataset = []
    for i in dataset:
        if i[axis] == value:
            reducedFeatVec = i[:axis]
            reducedFeatVec.extend(i[axis + 1:])
            spliteddataset.append(reducedFeatVec)

    return spliteddataset


# 选择信息增益最大的属性
def choosebestfeature(dataset):
    numfeatures = len(dataset[0]) - 1   # 属性的个数，减1是因为要减去标签
    baseentropy = informationentropy(dataset)   # 计算原本信息熵
    bestInfoGain = 0.0  # 初始化信息增益
    bestfeature = -1    # 初始化分裂属性

    for i in range(numfeatures):  # 对每个属性计算信息增益

        featList = [example[i] for example in dataset]  # 提取对应的列
        uniqueVals = set(featList)  # 该属性的取值集合，计算有多少不同的值
        newEntropy = 0.0

        for value in uniqueVals:  # 对每一种取值计算信息增益
            subDataSet = splitdataset(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * informationentropy(subDataSet)

        infoGain = baseentropy - newEntropy
        if (infoGain > bestInfoGain):  # 选择信息增益最大的属性
            bestInfoGain = infoGain
            bestfeature = i

    return bestfeature


# 通过排序返回出现次数最多的类
def majorclass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归构建决策树
def createtree(dataset, labels):
    classList = [example[-1] for example in dataset]  # 类向量

    if classList.count(classList[0]) == len(classList):  # 如果只有一个类，返回
        return classList[0]
    if len(dataset[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类
        return majorclass(classList)

    bestfeature = choosebestfeature(dataset)  # 最优划分属性的索引
    bestlabel = labels[bestfeature]  # 最优划分属性的标签

    ID3tree = {bestlabel: {}}

    del(labels[bestfeature])  # 已经选择的特征不再参与分类
    featValues = [example[bestfeature] for example in dataset]
    uniqueValue = set(featValues)  # 该属性所有可能取值，也就是节点的分支
    for value in uniqueValue:  # 对每个分支，递归构建树
        subLabels = labels[:]
        ID3tree[bestlabel][value] = createtree(splitdataset(dataset, bestfeature, value), subLabels)

    return ID3tree


# 根据建立好的决策树进行划分类别
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 跟节点对应的属性
    classLabel = None
    for key in secondDict.keys():  # 对每个分支循环
        if testVec[featIndex] == key:  # 测试样本进入某个分支
            if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:  # 如果是叶子， 返回结果
                classLabel = secondDict[key]
    return classLabel


# 测试决策树，返回、错误个数
def testtree(decisiontree, labels, testdata):
    '''
    真阴性率(TNR)：也称为特异性（Specificity），表示在所有实际负例中，模型成功地识别出了多少个负例。
    TNR计算方式为：TNR = TN / (TN + FP)，
    其中TN表示真阴性（模型正确地预测为负例的数量），FP表示假阳性（实际为负例但模型错误地预测为正例的数量）。
    TNR越高，表示模型在负例的分类上表现越好。

    真阳性率(TPR)：也称为灵敏度（Sensitivity）或召回率（Recall），表示在所有实际正例中，模型成功地识别出了多少个正例。
    TPR计算方式为：TPR = TP / (TP + FN)，
    其中TP表示真阳性（模型正确地预测为正例的数量），FN表示假阴性（实际为正例但模型错误地预测为负例的数量）。
    TPR越高，表示模型在正例的分类上表现越好。

    假阴性率(FNR)：表示模型未能正确识别的实际正例的比例。FNR计算方式为：FNR = FN / (TP + FN)。FNR越低，表示模型在识别正例方面表现越好。
    假阳性率(FPR)：表示模型错误地将负例分类为正例的比例。FPR计算方式为：FPR = FP / (TN + FP)。FPR越低，表示模型在负例分类方面表现越好。
    '''

    total = len(testdata)
    right = 0

    for i in testdata:
        classified_lable = classify(decisiontree, labels, i)
        if classified_lable == i[4]:
            right += 1

    accuracy = right/total
    return accuracy


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-", connectionstyle="arc3", shrinkA=0, shrinkB=16)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="top", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, s=txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # 设置决策节点和叶节点的边框形状、边距和透明度，以及箭头的形状
    decisionNode = dict(boxstyle="square,pad=0.5", fc="0.9")
    leafNode = dict(boxstyle="round4, pad=0.5", fc="0.9")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# 测试决策树正确率
def testing(myTree, data_test, labels):
    error=0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    return float(error)


# 测试投票节点正确率
def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    return float(error)


# 后剪枝(由于鸢尾花数据集无验证机不便预剪枝，且预剪枝可能会剪掉有用的子节点，故用后剪枝)
def postPruningTree(inputTree, dataset, datatest, labels):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    classList = [example[-1] for example in dataset]
    featkey = copy.deepcopy(firstStr)
    labelIndex = labels.index(featkey)
    temp_labels = copy.deepcopy(labels)
    del(labels[labelIndex])
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            if type(dataset[0][labelIndex]).__name__ == 'str':
                inputTree[firstStr][key] =postPruningTree(secondDict[key],
                                                           splitdataset(dataset, labelIndex, key),
                                                           splitdataset(datatest, labelIndex, key),
                                                           copy.deepcopy(labels))
            else:
                inputTree[firstStr][key] = postPruningTree(secondDict[key],
                                                           splitContinuousDataSet(dataset, labelIndex, featvalue, key),
                                                           splitContinuousDataSet(datatest, labelIndex, featvalue, key),
                                                           copy.deepcopy(labels))
    if testing(inputTree, datatest, temp_labels) <= testingMajor(majorclass(classList), datatest):
        return inputTree
    return majorclass(classList)


# 限制深度的决策树
def createdepthlimitedtree(dataset, labels, maxdepth, depth):
    classList = [example[-1] for example in dataset]  # 类向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类，返回
        return classList[0]
    if len(dataset[0]) == 1 or depth == maxdepth:  # 如果所有特征都被遍历完了，返回出现次数最多的类
        return majorclass(classList)

    bestfeature = choosebestfeature(dataset)  # 最优划分属性的索引
    bestlabel = labels[bestfeature]  # 最优划分属性的标签

    depthlimitedtree = {bestlabel: {}}

    del(labels[bestfeature])  # 已经选择的特征不再参与分类
    featValues = [example[bestfeature] for example in dataset]
    uniqueValue = set(featValues)  # 该属性所有可能取值，也就是节点的分支
    for value in uniqueValue:  # 对每个分支，递归构建树
        subLabels = labels[:]
        depthlimitedtree[bestlabel][value] = createdepthlimitedtree(splitdataset(dataset, bestfeature, value), subLabels, maxdepth, depth + 1)
    return depthlimitedtree


if __name__ == '__main__':
    choice_current = int(input("请输入选择什么数据集，iris数据集输入1，adult数据集输入2："))
    train_data, test_data = initdata(choice_current)
    train_data = discretize(choice_current, train_data)
    test_data = discretize(choice_current, test_data)
    print(train_data)

