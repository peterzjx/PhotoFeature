import os
import sys
import PIL.ImageStat
import PIL.Image as Image
import math
import skimage.data
import matplotlib.pyplot as plt
import numpy
import pywt

def brightness1(im_file):
    im = Image.open(im_file).convert('L')
    stat = PIL.ImageStat.Stat(im)
    return stat.mean[0]


def brightness2(im_file):
    im = Image.open(im_file).convert('L')
    stat = PIL.ImageStat.Stat(im)
    return stat.rms[0]


def brightness3(im_file):
    im = Image.open(im_file)
    stat = PIL.ImageStat.Stat(im)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def brightness4(im_file):
    im = Image.open(im_file)
    stat = PIL.ImageStat.Stat(im)
    r, g, b = stat.rms
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def brightness5(im_file):
    im = Image.open(im_file)
    stat = PIL.ImageStat.Stat(im)
    gs = (math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2)) for r, g, b in im.getdata())
    return sum(gs) / stat.count[0]


def showResult(files, attributes_list):
    n = len(files)
    fig = plt.figure()
    max = 5
    for i in xrange(n):
        file = files[i]
        image = skimage.data.imread(file)
        ax = fig.add_subplot(n / max, max, i + 1)
        ax.imshow(image)
        text = ""
        for attribute, value in attributes_list[i].iteritems():
            text = text + "\r\n" + attribute + ":" + str(value)
        ax.text(0, 0, text)
        ax.axis('off')
    plt.show()


def getBrightness(im):
    im = im.convert('L')
    stat = PIL.ImageStat.Stat(im)
    return stat.mean[0]


def getColorfulness(im):
    rg = list(((r - g) for r, g, b in im.getdata()))
    yb = list(((0.5 * (r + g) - b) for r, g, b in im.getdata()))
    sigma = math.sqrt(numpy.std(rg) ** 2 + numpy.std(yb) ** 2)
    mu = math.sqrt(numpy.average(rg) ** 2 + numpy.average(yb) ** 2)
    return sigma + 0.3 * mu


def getContrast(im):
    im = im.convert('L')
    dots = list(im.getdata())
    x = list(1.0 * d / 255 for d in dots)
    avg = numpy.average(x)
    return math.sqrt(1.0 / (len(x) - 1) * sum((xi - avg) ** 2 for xi in x))


def getSaturation(im):
    return numpy.average(list(max(r, g, b) - min(r, g, b) for r, g, b in im.getdata()))


def getSaturationStd(im):
    return numpy.std(list(max(r, g, b) - min(r, g, b) for r, g, b in im.getdata()))

# code is from https://gist.github.com/shahriman/3289170
def getBlur(im):
    thresh = 35
    MinZero = 0.05
    im = im.convert('F')
    x = numpy.asarray(im)
    x_cropped = x[0:(numpy.shape(x)[0] / 16) * 16 - 1, 0:(numpy.shape(x)[1] / 16) * 16 - 1]
    LL1, (LH1, HL1, HH1) = pywt.dwt2(x_cropped, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')
    Emap1 = numpy.square(LH1) + numpy.square(HL1) + numpy.square(HH1)
    Emap2 = numpy.square(LH2) + numpy.square(HL2) + numpy.square(HH2)
    Emap3 = numpy.square(LH3) + numpy.square(HL3) + numpy.square(HH3)

    dimx = numpy.shape(Emap1)[0] / 8
    dimy = numpy.shape(Emap1)[1] / 8
    Emax1 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax1.append([])
        for k in range(0, dimy - 2):
            Emax1[j].append(numpy.max(numpy.max(Emap1[vert:vert + 7, horz:horz + 7])))
            horz = horz + 8
        vert = vert + 8

    dimx = numpy.shape(Emap2)[0] / 4
    dimy = numpy.shape(Emap2)[1] / 4
    Emax2 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax2.append([])
        for k in range(0, dimy - 2):
            Emax2[j].append(numpy.max(numpy.max(Emap2[vert:vert + 3, horz:horz + 3])))
            horz = horz + 4
        vert = vert + 4

    dimx = numpy.shape(Emap3)[0] / 2
    dimy = numpy.shape(Emap3)[1] / 2
    Emax3 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax3.append([])
        for k in range(0, dimy - 2):
            Emax3[j].append(numpy.max(numpy.max(Emap3[vert:vert + 1, horz:horz + 1])))
            horz = horz + 2
        vert = vert + 2

    N_edge = 0
    N_da = 0
    N_rg = 0
    N_brg = 0

    EdgeMap = []
    for j in range(0, dimx - 2):
        EdgeMap.append([])
        for k in range(0, dimy - 2):
            if (Emax1[j][k] > thresh) or (Emax2[j][k] > thresh) or (Emax3[j][k] > thresh):
                EdgeMap[j].append(1)
                N_edge = N_edge + 1
                rg = 0
                if (Emax1[j][k] > Emax2[j][k]) and (Emax2[j][k] > Emax3[j][k]):
                    N_da = N_da + 1
                elif (Emax1[j][k] < Emax2[j][k]) and (Emax2[j][k] < Emax3[j][k]):
                    rg = 1
                    N_rg = N_rg + 1
                elif (Emax2[j][k] > Emax1[j][k]) and (Emax2[j][k] > Emax3[j][k]):
                    rg = 1
                    N_rg = N_rg + 1
                if rg and (Emax1[j][k] < thresh):
                    N_brg = N_brg + 1
            else:
                EdgeMap[j].append(0)

    per = float(N_da) / N_edge
    BlurExtent = float(N_brg) / N_rg
    return BlurExtent


'''
    Measures for blur
'''


def blur(im_file):
    pass


def main():
    attributes_list = []
    files = []
    dir = "house/"
    for file in os.listdir(dir):
        if not file.endswith(".jpg"):
            continue
        attributes = {}
        im = Image.open(dir + file)
        files.append(dir + file)
        attributes["brightness"] = getBrightness(im)
        attributes["colorfulness"] = getColorfulness(im)
        attributes["contrast"] = getContrast(im)
        attributes["saturation"] = getSaturation(im)
        attributes["saturation_std"] = getSaturationStd(im)
        attributes["blur"]=getBlur(im)
        attributes_list.append(attributes)
    showResult(files, attributes_list)


main()
