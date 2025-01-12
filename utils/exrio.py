import OpenEXR
import Imath
import numpy as np


def read(filename):
    img = OpenEXR.InputFile(filename)
    HEADER = img.header()
    dw = HEADER['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    if HEADER['channels']['G'].type == Imath.PixelType(Imath.PixelType.HALF):
        np_type = np.float16
    else:
        np_type = np.float32
    R = np.fromstring(img.channel('R'), dtype=np_type).reshape((size[1], size[0]))
    G = np.fromstring(img.channel('G'), dtype=np_type).reshape((size[1], size[0]))
    B = np.fromstring(img.channel('B'), dtype=np_type).reshape((size[1], size[0]))
    result = np.concatenate((R[:, :, np.newaxis], G[:, :, np.newaxis], B[:, :, np.newaxis]), axis=2)
    img.close()
    return result


def save(filename, img, float_32=False, need_compress=False):
    if float_32:
        pixels = img.astype(np.float32)
    else:
        pixels = img.astype(np.float16)
    height = img.shape[0]
    width = img.shape[1]
    HEADER = _header(width, height, float_32=float_32, need_compress=need_compress)
    #if float_32:
    #    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    #else:
    #    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    #HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
    exr = OpenEXR.OutputFile(filename, HEADER)
    R = pixels[:, :, 0].tostring()
    G = pixels[:, :, 1].tostring()
    B = pixels[:, :, 2].tostring()
    exr.writePixels({'R': R, 'G': G, 'B': B})
    exr.close()


def _header(width, height, float_32=False, need_compress=False):
    head = OpenEXR.Header(width, height)
    if not float_32:
        head['channels']['G'] = Imath.Channel(type=Imath.PixelType(Imath.PixelType.HALF))
        head['channels']['B'] = Imath.Channel(type=Imath.PixelType(Imath.PixelType.HALF))
        head['channels']['R'] = Imath.Channel(type=Imath.PixelType(Imath.PixelType.HALF))
    if not need_compress:
        head['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    return head
