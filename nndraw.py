import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from nn import NN, encode_x

class Window:
    size = 28
    scale = 8
    imgsize = size * scale

    def __init__(self):
        self.nn = NN.load("nn.json")

        self.root = tk.Tk()
        self.active = False

        self.img = Image.new("L", (Window.imgsize, Window.imgsize))
        self.hidden_canvas = ImageDraw.Draw(self.img)

        self.canvas = tk.Canvas(self.root, width=Window.imgsize,height=Window.imgsize,background="black")
        self.canvas.pack()
        self.canvas.focus()

        self.root.bind("<Button-1>", self.on_click)
        self.root.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Motion>", self.on_move)
        self.root.mainloop()

    def on_click(self, e):
        self.active = True
    def on_release(self, e):
        self.active = False
        self.test()
        self.clear()

    def on_move(self, e):
        if self.active:
            r = 9
            xy1 = (e.x-r, e.y-r)
            xy2 = (e.x+r, e.y+r)
            self.canvas.create_oval(xy1[0], xy1[1], xy2[0], xy2[1], width = 0, fill = "white")
            self.hidden_canvas.ellipse([xy1, xy2], fill=255, width=r*2)

    def test(self):
        img = self.img.resize((Window.size, Window.size), resample=Image.BILINEAR)
        r = self.nn.feedforward(encode_x(np.array(img))).flatten()
        i = np.argmax(r)
        print("y =", i, "(", r[int(i)], ")")

    def clear(self):
        self.canvas.delete("all")
        self.hidden_canvas.rectangle((0, 0, Window.imgsize, Window.imgsize), fill=0)

if __name__ == "__main__":
    Window()
