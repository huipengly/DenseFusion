import numpy as np
import matplotlib.pyplot as plt

from openni import openni2
from openni import _openni2 as c_api


class XtionPro(object):
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fps = 30
        try:
            openni2.initialize()
        except:
            print("Device not initialized.")
            return
        try:
            self.dev = openni2.Device.open_any()
        except:
            print("Open device failed.")
            return
        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                       resolutionX=self.width,
                                                       resolutionY=self.height,
                                                       fps=self.fps))
        self.color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                       resolutionX=self.width,
                                                       resolutionY=self.height,
                                                       fps=self.fps))
        self.dev.set_image_registration_mode(True)
        self.dev.set_depth_color_sync_enabled(True)
        self.depth_stream.set_mirroring_enabled(True)
        self.color_stream.set_mirroring_enabled(True)
        self.depth_stream.start()
        self.color_stream.start()

    def __del__(self):
        self.depth_stream.close()
        self.color_stream.close()
        self.dev.close()
        print("xtion closed")

    def depth_data(self):
        return np.frombuffer(self.depth_stream.read_frame().get_buffer_as_triplet(), dtype='uint16',
                             count=self.width*self.height).reshape([self.height, self.width])

    def color_data(self):
        return np.frombuffer(self.color_stream.read_frame().get_buffer_as_triplet(), dtype='uint8',
                             count=self.width*self.height*3).reshape([self.height, self.width, 3])


def main():
    xtion = XtionPro()
    plt.ion()
    for i in range(100):
        color_data = xtion.color_data()
        depth_data = xtion.depth_data()
        plt.subplot(1, 2, 1)
        plt.imshow(depth_data, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(color_data)
        plt.show()
        plt.pause(0.0333)


if __name__ == '__main__':
    main()
