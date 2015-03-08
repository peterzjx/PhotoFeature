__author__ = 'Peter'
from PIL import ImageStat
from PIL import Image
from PIL import ImageMath
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
    sigma_rg, sigma_yb = stat_rg.var[0], stat_yb.var[0]
    mu_rg, mu_yb = stat_rg.mean[0], stat_yb.mean[0]
    sigma_rgyb = math.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rgyb = math.sqrt(mu_rg**2 + mu_yb**2)
    cf = sigma_rgyb + 0.3*mu_rgyb
    return cf, sigma_rgyb, mu_rgyb


def naturalness(im):
    pass


def contrast(im):
    pass


def main():
    imgname = '1.jpg'
    im = Image.open(imgname)
    print colorfulness(im)
main()
