from tkinter import *
from PIL import Image
from keras.models import load_model
from random import randint as ri

from imagecanvas import ImageCanvas
from stickgenerator import get_stick
from trainthread import TrainThread

import math
import multiprocessing
import numpy as np
import signal

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
        
        self.mmaster = master
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        with PackFrame(self, side=TOP) as vertical:
            button = Button(vertical, text="Train!", command=self.order_train)
            button.pack(side=LEFT)
            
            button = Button(vertical, text="Load!", command=self.load_autoencoder)
            button.pack(side=LEFT)
            
            self.test_move_cb_state = IntVar()
            cb= Checkbutton(vertical, text="Move!", variable=self.test_move_cb_state, command=self.test_body)
            cb.pack(side=LEFT)
            
        with PackFrame(self, side=TOP) as vertical:
            with PackFrame(vertical, side=LEFT) as slider_frame:
                self.sliders = self._create_sliders(slider_frame, 5, self._slider_moved)

            self.canvas = ImageCanvas(vertical, 128, 128)
            self.canvas.pack(side=LEFT)

            with PackFrame(vertical, side=LEFT) as slider_frame:
                self.nn_sliders = self._create_sliders(slider_frame, 16, self._slider_moved)

            self.nn_canvas = ImageCanvas(vertical, 128, 128)
            self.nn_canvas.pack(side=LEFT)

        self.autoencoder = None
        self.encoder = None
        self.test_targets = [0 for _ in range(len(self.sliders))]
        
        
        self.in_queue = multiprocessing.Queue()
        self.out_queue = multiprocessing.Queue()
        
        self.trt = TrainThread(self.out_queue, self.in_queue)
        
        self.trt.start()
        
        self.update_image()
        self.after(500, self.update_timeout)

    def update_timeout(self):
        if not self.in_queue.empty():
            msg = self.in_queue.get()
            print("got from queue:", msg)
            self.load_autoencoder()
            self.update_image()
        
        self.after(500, self.update_timeout)
        
    def on_closing(self):
        self.out_queue.put(("quit", ""))
        #self.trt.send_signal(signal.SIGINT)
        self.mmaster.destroy()

    def order_train(self):
        self.out_queue.put(("train", (5, 5)))

    def update_image(self):
        values = [slider.get() / 180 * math.pi for slider in self.sliders]
        stick_img = get_stick(values)
        self.canvas.set_image(stick_img)
        
        if self.autoencoder:
            stick = np.array(stick_img, dtype=np.float32).reshape((1, 128, 128, 1))
            reenc_np = self.autoencoder.predict(stick)[0]
            reenc = Image.fromarray(255-np.uint8(reenc_np.reshape((128, 128))*255), mode="L")
            self.nn_canvas.set_image(reenc)
            
            if self.encoder:
                encoded = self.encoder.predict(stick)[0].reshape(16)
                for value, slider in zip(encoded, self.nn_sliders):
                    slider.set(int(value*10))
            
    
    def test_body(self):
        if self.test_move_cb_state.get():
            for i, slider in enumerate(self.sliders):
                cur = slider.get()
                tar = self.test_targets[i]
                if cur < tar:
                    slider.set(cur + 1)
                elif cur > tar:
                    slider.set(cur - 1)
                else:
                    self.test_targets[i] = ri(-50, 50)
            
            self.after(20, self.test_body)
        
    
    def load_autoencoder(self):
        self.autoencoder = load_model("nns/ac")
        self.encoder = load_model("nns/enc")
        self.update_image()

    def _slider_moved(self, value=None):
        self.update_image()

    def _create_sliders(self, root, count, callback):
        def slider_constructor(root):
            s = Scale(root, from_= -50, to=50, orient=HORIZONTAL, command=callback)
            s.pack(side=TOP)
            return s

        return [slider_constructor(root) for _ in range(count)]
