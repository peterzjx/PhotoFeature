import os
import csv
import PIL.ImageStat
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import math
import skimage.data
import matplotlib.pyplot as plt
import numpy as np
import pywt


def getBrightness_distribution(im):
    im = im.convert('L')
    dots = list(im.getdata())
    l1 = float(sum(list(int(d < 0.01*255) for d in dots)))/len(dots)
    l2 = float(sum(list(int(d < 0.10*255) for d in dots)))/len(dots)
    g1 = float(sum(list(int(d > 0.99*255) for d in dots)))/len(dots)
    g2 = float(sum(list(int(d > 0.90*255) for d in dots)))/len(dots)
    return [l1, l2, g1, g2]


# def getBrightness_gt(im):
#     im = im.convert('L')
#     dots = list(im.getdata())
#     gt = sum(list(int(d > 0.99*255) for d in dots))
#     return float(gt)/len(dots)


# def brightness1(im_file):
#     im = Image.open(im_file).convert('L')
#     stat = PIL.ImageStat.Stat(im)
#     return stat.mean[0]
#
#
# def brightness2(im_file):
#     im = Image.open(im_file).convert('L')
#     stat = PIL.ImageStat.Stat(im)
#     return stat.rms[0]
#
#
# def brightness3(im_file):
#     im = Image.open(im_file)
#     stat = PIL.ImageStat.Stat(im)
#     r, g, b = stat.mean
#     return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
#
#
# def brightness4(im_file):
#     im = Image.open(im_file)
#     stat = PIL.ImageStat.Stat(im)
#     r, g, b = stat.rms
#     return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))
#
#
# def brightness5(im_file):
#     im = Image.open(im_file)
#     stat = PIL.ImageStat.Stat(im)
#     gs = (math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2)) for r, g, b in im.getdata())
#     return sum(gs) / stat.count[0]


def showResult(files, attributes_list):
    n = len(files)
    fig = plt.figure()
    max = 3
    for i in xrange(n):
        file = files[i]
        image = skimage.data.imread(file)
        ax = fig.add_subplot(n / max, max, i + 1)
        ax.imshow(image)
        text = os.path.basename(file)
        for attribute, value in attributes_list[i].iteritems():
            text = text + "\n" + attribute + ":" + str(value)
        ax.text(0, 0, text)
        ax.axis('off')
    plt.show()


def getBrightness_tuple(im):
    im = im.convert('L')
    stat = PIL.ImageStat.Stat(im)
    return [stat.mean[0], stat.stddev[0]]


def getColorfulness(im):
    if len(im.split()) == 1:  # black and white photo
        return 0
    rg = list(((r - g) for r, g, b in im.getdata()))
    yb = list(((0.5 * (r + g) - b) for r, g, b in im.getdata()))
    sigma = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    mu = math.sqrt(np.average(rg) ** 2 + np.average(yb) ** 2)
    return sigma + 0.3 * mu


def getContrast(im):
    im = im.convert('L')
    dots = list(im.getdata())
    x = list(1.0 * d / 255 for d in dots)
    avg = np.average(x)
    return math.sqrt(1.0 / (len(x) - 1) * sum((xi - avg) ** 2 for xi in x))


def autoContrast(im):
    return ImageOps.autocontrast(im)


def getRGBContrast(im):
    if len(im.split()) == 1:
        contrast = getContrast(im)
        return contrast, contrast, contrast
    R, G, B = im.split()
    return getContrast(R), getContrast(G), getContrast(B)


def getSaturation_tuple(im):
    if len(im.split()) == 1:  # black and white photo
        return [0, 0]
    d = list(max(r, g, b) - min(r, g, b) for r, g, b in im.getdata())
    return [np.average(d), np.std(d)]


def getRaw_data(im):
    size = im.size
    mat = np.array(list(im.getdata())).reshape(size)
    return mat


def getSpacial_from_mat(mat):
    s = np.std(mat)
    # y = np.mean(mat, axis=0)
    # x = np.mean(mat, axis=1)
    return s


def getSpacial(im):
    im = convert_edge(im)
    return getSpacial_from_mat(getRaw_data(im))


# code is from https://gist.github.com/shahriman/3289170
def getBlur(im):
    thresh = 35
    MinZero = 0.05
    im = im.convert('F')
    x = np.asarray(im)
    x_cropped = x[0:(np.shape(x)[0] / 16) * 16 - 1, 0:(np.shape(x)[1] / 16) * 16 - 1]
    LL1, (LH1, HL1, HH1) = pywt.dwt2(x_cropped, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')
    Emap1 = np.square(LH1) + np.square(HL1) + np.square(HH1)
    Emap2 = np.square(LH2) + np.square(HL2) + np.square(HH2)
    Emap3 = np.square(LH3) + np.square(HL3) + np.square(HH3)

    dimx = np.shape(Emap1)[0] / 8
    dimy = np.shape(Emap1)[1] / 8
    Emax1 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax1.append([])
        for k in range(0, dimy - 2):
            Emax1[j].append(np.max(np.max(Emap1[vert:vert + 7, horz:horz + 7])))
            horz = horz + 8
        vert = vert + 8

    dimx = np.shape(Emap2)[0] / 4
    dimy = np.shape(Emap2)[1] / 4
    Emax2 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax2.append([])
        for k in range(0, dimy - 2):
            Emax2[j].append(np.max(np.max(Emap2[vert:vert + 3, horz:horz + 3])))
            horz = horz + 4
        vert = vert + 4

    dimx = np.shape(Emap3)[0] / 2
    dimy = np.shape(Emap3)[1] / 2
    Emax3 = []
    vert = 1
    for j in range(0, dimx - 2):
        horz = 1;
        Emax3.append([])
        for k in range(0, dimy - 2):
            Emax3[j].append(np.max(np.max(Emap3[vert:vert + 1, horz:horz + 1])))
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

def saveResult(files, dir, attributes_list):
    n = len(files)
    writer = csv.writer(open(dir+"features.csv", 'w'))
    for i in xrange(n):
        file = files[i]
        name = os.path.basename(file)
        if i == 0:
            row = []
            row.append('name')
            for attribute, value in iter(sorted(attributes_list[i].iteritems())):
                row.append(attribute)
            writer.writerow(row)
        row = []
        row.append(name)
        for attribute, value in iter(sorted(attributes_list[i].iteritems())):
            row.append(value)
        writer.writerow(row)
    print "result saved"


def main():
    attributes_list = []
    files = []
    dir = "/home/peter/dev/PhotoFeature/data/pro2/"
    lst = os.listdir(dir)
    lst.sort()
    for file in lst:
        if not file.endswith(".jpg"):
            continue
        print file
        attributes = {}
        im = Image.open(dir + file)
        if dir.endswith('pro/'):
            x, y = im.size
            im = im.resize((int(x/2), int(y/2)), resample=PIL.Image.BICUBIC)
            im.save("/home/peter/dev/PhotoFeature/data/pro2/"+file)
        files.append(dir + file)
        attributes["brightness"], attributes["brightness_std"] = getBrightness_tuple(im)
        attributes["colorfulness"] = getColorfulness(im)
        attributes["contrast"] = getContrast(im)
        attributes["contrast_diff"] = getContrast(autoContrast(im)) - attributes["contrast"]
        attributes["R_contrast"], attributes["G_contrast"], attributes["B_contrast"] = getRGBContrast(im)
        attributes["saturation"], attributes["saturation_std"] = getSaturation_tuple(im)
        attributes["brightness_diff"] = getBrightness_tuple(autoContrast(im))[0] - attributes["brightness"]
        attributes["saturation_diff"] = getSaturation_tuple(autoContrast(im))[0] - attributes["saturation"]
        attributes["blur"] = getBlur(im)
        attributes["spacial_std"] = getSpacial(im)
        attributes["dark_black"], attributes["black"], attributes["light_white"], attributes["white"] = getBrightness_distribution(im)

        attributes_list.append(attributes)
    # showResult(files, attributes_list)
    saveResult(files, dir, attributes_list)
#   saturation brightness_std colorfulness brightness light_white dark_black black blur saturation_std white contrast


def convert_edge(im):
    im = im.filter(ImageFilter.GaussianBlur(radius=100))
    im = im.filter(ImageFilter.FIND_EDGES)
    im = im.convert('L')
    im = im.filter(ImageFilter.GaussianBlur(radius=100))
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(5.0)
    im = im.filter(ImageFilter.GaussianBlur(radius=100))
    im = im.convert('1')
    x, y = im.size
    im = im.crop(box=(5, 5, x-5, y-5))
    return im

def test():
    dir = "/home/peter/dev/PhotoFeature/py/"
    file = "8.jpg"
    im = Image.open(dir + file)
    print getContrast(autoContrast(im)) - getContrast(im)
    print getBrightness_tuple(autoContrast(im))[0] - getBrightness_tuple(im)[0]
    im = autoContrast(im)
    im = im.convert('RGB')
    im.save("edge.jpg")

main()
# test()
