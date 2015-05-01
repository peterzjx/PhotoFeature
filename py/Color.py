__author__ = 'peter'
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorsys
import os
import pickle
import csv

def getMain_colors(im):
    im = im.quantize(colors=8, kmeans=8).convert('RGB')
    totalpix = float(im.size[0] * im.size[1])
    colors = im.getcolors()
    colors.sort(key=lambda tup: tup[0], reverse=True)
    colors = [(x[0]/totalpix, x[1]) for x in colors]
    return colors


def getDist(data1, data2):
    # dist = ((data1[0] - data2[0])*255)**2
    dist = (data1[0] - data2[0])**2
    r, g, b = (float(x)/255 for x in data1[1])
    color1 = colorsys.rgb_to_hsv(r, g, b)
    r, g, b = (float(x)/255 for x in data2[1])
    # print r, g, b
    color2 = colorsys.rgb_to_hsv(r, g, b)
    # print color2
    for c in range(3):
        dist += (color1[c] - color2[c])**2
    return dist


def getMetric(rawdata1, rawdata2):
    data1 = rawdata1[:]
    data2 = rawdata2[:]
    res = []
    for color1 in data1:
        dist = []
        for i, color2 in enumerate(data2):
            dist.append((i, getDist(color1, color2)))
        dist.sort(key=lambda tup: tup[1])
        chosen = data2.pop(dist[0][0])
        res.append(((color1, chosen), dist[0][1]))
    return res


# def getSimilarity(im1, im2):
#     data1 = getMain_colors(im1)
#     data2 = getMain_colors(im2)
#     result = getMetric(data1, data2)
#     totaldist = 0
#     weight = 1
#     for X, (pair, dist) in enumerate(result):
#         totaldist += dist * weight
#         weight *= 0.8
#     return totaldist


def getSimilarity(data1, data2):
    result = getMetric(data1, data2)
    totaldist = 0
    weight = 1
    for X, (pair, dist) in enumerate(result):
        totaldist += dist * weight
        weight *= 0.5
    return totaldist


def showResult(raw):
    fig = plt.figure()
    currentAxis = plt.gca()
    for X, (pair, dist) in enumerate(raw):
        color1 = list(float(x)/255 for x in pair[0][1])
        color2 = list(float(x)/255 for x in pair[1][1])
        currentAxis.add_patch(Rectangle((X, 0), 1, 1, facecolor=color1))
        currentAxis.add_patch(Rectangle((X, 2), 1, 1, facecolor=color2))
    plt.ylim([-10, 10])
    plt.xlim([-10, 15])
    fig.show()
    plt.show()



# def test():
#     dir = "/home/peter/dev/PhotoFeature/py/"
#     file = "1.jpg"
#     file2 = "7.jpg"
#     im1 = Image.open(dir + file)
#     im2 = Image.open(dir + file2)
#     data1 = getMain_colors(im1)
#     data2 = getMain_colors(im2)
#     result = getMetric(data1, data2)
#     # print result
#     # showResult(result)
#     print getSimilarity(im1, im2)


def getAlldist():
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    full_data = pickle.load(open(dir + "main_colors", 'rb'))
    result = []
    for i, data1 in enumerate(full_data):
        line = []
        for data2 in full_data:
            if data1 == data2:
                sim = 0
            else:
                sim = getSimilarity(data1, data2)
            line.append(sim)
        print i
        result.append(line)
    pickle.dump(result, open(dir+"distance", 'wb'))

def main():
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    lst = os.listdir(dir)
    lst.sort()
    result = []
    for file1 in lst:
        if not file1.endswith(".jpg"):
            continue
        print file1
        im1 = Image.open(dir + file1)
        data = getMain_colors(im1)
        result.append(data)
    pickle.dump(result, open(dir+"main_colors", 'wb'))


def getLabel():
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    lst = os.listdir(dir)
    lst.sort()
    l = []
    for file1 in lst:
        if not file1.endswith(".jpg"):
            continue
        l.append(file1)
    return l


def findNN():
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    full_data = pickle.load(open(dir + "distance", 'rb'))
    result = []
    label = getLabel()
    for i, line in enumerate(full_data):
        # if i > 10:
        #     continue
        line_id = list(enumerate(line))
        line_id.sort(key=lambda tup: tup[1])
        # print line_id
        nn_raw = line_id[1:6+1]
        print label[i],
        nns = [label[x[0]] for x in nn_raw]
        nn = [int(x[0]>=257) for x in nn_raw]
        print nns
        res = int(sum(nn) >= 3)
        result.append(res)
    return result


def saveResult(result):
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    writer = csv.writer(open(dir+"nn.csv", 'w'))
    for row in result:
        writer.writerow(str(row))
    print "result saved"


def compareResult(result):
    return (len(result) - (sum(result[:257])+(len(result)-257-sum(result[258:]))))/float(len(result))


def test():
    dir = "/home/peter/dev/PhotoFeature/data/all/"
    lst = os.listdir(dir)
    lst.sort()
    result = []
    file1 = "pro(164).jpg"
    file2 = "pro(127).jpg"
    im1 = Image.open(dir + file1)
    im2 = Image.open(dir + file2)
    data1 = getMain_colors(im1)
    data2 = getMain_colors(im2)
    full_data = pickle.load(open(dir + "distance", 'rb'))
    print full_data[324][281]
    print getSimilarity(data1, data2)


# getAlldist()
# main()
# print getLabel()
result = findNN()
# print compareResult(result)
# saveResult(result)
# test()