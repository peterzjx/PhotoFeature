__author__ = 'Peter'
from PIL import ImageStat
from PIL import Image
from PIL import ImageMath, ImageEnhance
import math


def brightness(im):
    """return (mean, var)"""
    im = im.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0], stat.var[0]


def saturation(im):
    """return (mean, var)"""
    r, g, b = im.split()
    im_max = max(r, g, b)
    im_min = min(r, g, b)
    im_diff = ImageMath.eval("convert(a - b, 'L')", a = im_max, b = im_min)
    stat = ImageStat.Stat(im_diff)
    return stat.mean[0], stat.var[0]


def colorfulness(im):
    """return (cf, sigma_rgyb, mu_rgyb)"""
    r, g, b = im.split()
    im_rg = ImageMath.eval("convert(a - b, 'L')", a=r, b=g)
    im_yb = ImageMath.eval("convert((a + b)/2 - c, 'L')", a=r, b=g, c=b)
    stat_rg = ImageStat.Stat(im_rg)
    stat_yb = ImageStat.Stat(im_yb)
    sigma_rg, sigma_yb = stat_rg.stddev[0], stat_yb.stddev[0]
    mu_rg, mu_yb = stat_rg.mean[0], stat_yb.mean[0]
    sigma_rgyb = math.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rgyb = math.sqrt(mu_rg**2 + mu_yb**2)
    cf = sigma_rgyb + 0.3*mu_rgyb
    return cf, sigma_rgyb, mu_rgyb


def naturalness(im):
    pass


def contrast(im):
    """return contrast"""
    im = im.convert('L')
    stat = ImageStat.Stat(im)
    mean = stat.mean[0]
    im_diff = ImageMath.eval("convert(a - b, 'L')", a=im, b=mean)
    stat_diff = ImageStat.Stat(im_diff)
    return stat_diff.sum2[0]/(stat_diff.count[0]-1)

def sharpness(im):
    pass


def main():
    imgname = '5.jpg'
    im = Image.open(imgname)
    print contrast(im)
main()
