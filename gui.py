from tkinter import *
from imagecanvas import ImageCanvas
from stickgenerator import get_stick

import numpy as np

import math

class PackFrame:
    def __init__(self, root, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.frame = Frame(root)

    def __enter__(self):
        return self.frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.frame.pack(*self.args, **self.kwargs)


class MainFrame(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()

        with PackFrame(self, side=LEFT) as slider_frame:
            self.sliders = self._create_sliders(slider_frame, 5, self._slider_moved)

        self.canvas = ImageCanvas(self, 128, 128)
        self.canvas.pack()

        self.update_image()


    def update_image(self):
        values = [slider.get() / 180 * math.pi for slider in self.sliders]
        stick_img = get_stick(values)
        print(np.array(stick_img, dtype=np.float32).shape)
        self.canvas.set_image(stick_img)


    def _slider_moved(self, value=None):
        self.update_image()


    def _create_sliders(self, root, count, callback):
        def slider_constructor(root):
            s = Scale(root, from_= -50, to=50, orient=HORIZONTAL, command=callback)
            s.pack(side=TOP)
            return s

        return [slider_constructor(root) for _ in range(count)]
