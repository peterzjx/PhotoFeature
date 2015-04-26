__author__ = 'peter'


def getMain_colors(im):
    im = im.quantize(colors=8, kmeans=8)
    print im.getpalette()[:3*8]
