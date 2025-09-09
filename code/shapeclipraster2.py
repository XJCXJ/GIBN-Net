from osgeo import gdal, gdal_array
import shapefile
import numpy as np
import os

#批量shp裁剪tiff影像
try:
    import Image
    import ImageDraw
except:
    from PIL import Image, ImageDraw

def read_tiff(inpath):
  ds=gdal.Open(inpath)
  row=ds.RasterXSize
  col=ds.RasterYSize
  band=ds.RasterCount

  data=np.zeros([row,col,band])
  for i in range(band):
   dt=ds.GetRasterBand(1)
   data[:,:,i]=dt.ReadAsArray(0,0,col,row)
  return data

def image2Array(i):
    """
    将一个Python图像库的数组转换为一个gdal_array图片
    """
    a = gdal_array.numpy.frombuffer(i.tobytes(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a

def world2Pixel(geoMatrix, x, y):
    """
    使用GDAL库的geomatrix对象((gdal.GetGeoTransform()))计算地理坐标的像素位置
    """
    ulx = geoMatrix[0]
    uly = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulx) / xDist)
    line = int((uly - y) / abs(yDist))
    return (pixel, line)


def write_img(filename,im_proj,im_geotrans,im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype,options=["TILED=YES", "COMPRESS=LZW"])

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:

        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def sha_raster(raster,shp,output):
    srcArray = gdal_array.LoadFile(raster)
    # 同时载入gdal库的图片从而获取geotransform
    srcImage = gdal.Open(raster)
    geoProj = srcImage.GetProjection()
    geoTrans = srcImage.GetGeoTransform()
    r = shapefile.Reader(shp)
    # 将图层扩展转换为图片像素坐标
    minX, minY, maxX, maxY = r.bbox
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    clip = srcArray[ulY:lrY, ulX:lrX]
    # 为图片创建一个新的geomatrix对象以便附加地理参照数据
    geoTrans = list(geoTrans)
    geoTrans[0] = minX
    geoTrans[3] = maxY
    # 在一个空白的8字节黑白掩膜图片上把点映射为像元绘制市县
    # 边界线
    pixels = []
    for p in r.shape(0).points:
        pixels.append(world2Pixel(geoTrans, p[0], p[1]))
    rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
    # 使用PIL创建一个空白图片用于绘制多边形
    rasterize = ImageDraw.Draw(rasterPoly)
    rasterize.polygon(pixels, 0)
    # 使用PIL图片转换为Numpy掩膜数组
    mask = image2Array(rasterPoly)
    name = os.path.basename(raster).split(".tif")[0]
    outfile = output + "\\" + name+ "_cut.tif"  # 对输出文件命名
    # 根据掩膜图层对图像进行裁剪
    clip = gdal_array.numpy.choose(mask, (clip, 0)).astype(gdal_array.numpy.uint16)
    write_img(outfile, geoProj, geoTrans, clip)
    gdal.ErrorReset()

if __name__ == "__main__":
    raster = r"D:\downloads\研究区\xiaotu\02.tif"
    # 用于裁剪的多边形shp文件
    shp = r"D:\咸鱼\矢量\01.shp"
    # 裁剪后的栅格数据
    output = r"D:\downloads\研究区\xiaotu"

    #依据shp创建掩膜进行对tiff文件的裁剪
    sha_raster(raster,shp,output)
