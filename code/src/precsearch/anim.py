"""
Helper code to make interactive plots and animations
"""

import numpy as np
import matplotlib as mpl
import matplotlib.animation
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.widgets import Slider, CheckButtons

def dummy(*args, **kwargs):
    pass

class Draggable:
    locked = None

    def __init__(self, rect):
        self.shape = rect
        self.press = None
        self.cids = []
        self.connect()
        self.on_press_cb = dummy
        self.on_move_cb = dummy
        self.on_press_cb_ret = None

    def connect(self):
        "connect to all the events we need"
        canvas = self.shape.figure.canvas
        self.cids = [
            canvas.mpl_connect("button_press_event", self.on_press),
            canvas.mpl_connect("button_release_event", self.on_release),
            canvas.mpl_connect("motion_notify_event", self.on_motion),
        ]

    def on_press(self, event):
        "on button press we will see if the mouse is over us and store some data"
        if Draggable.locked is not None:
            return
        if event.inaxes != self.shape.axes:
            return
        contains, attrd = self.shape.contains(event)
        if not contains:
            return
        Draggable.locked = self

        self.on_press_cb_ret = self.on_press_cb()
        self.press = *self.shape.get_center(), event.xdata, event.ydata

    def set_callback(self, press, move):
        """Defines callbacks to be called on press and move."""
        self.on_press_cb = press if press is not None else dummy
        self.on_move_cb = move if move is not None else dummy

    def on_motion(self, event):
        should_move = (
            (Draggable.locked is self)
            and (self.press is not None)
            and (event.inaxes == self.shape.axes)
        )
        if not should_move:
            return

        x0, y0, xp, yp = self.press
        dx = event.xdata - xp
        dy = event.ydata - yp

        self.shape.set_center((x0 + dx, y0 + dy))
        self.on_move_cb(x0, y0, xp, yp, event.xdata, event.ydata, self.on_press_cb_ret)
        self.shape.figure.canvas.draw()

    def on_release(self, event):
        if Draggable.locked is not self:
            return
        self.press = None
        self.on_press_cb_ret = None
        Draggable.locked = None
        self.shape.figure.canvas.draw()

    def disconnect(self):
        for cid in self.cids:
            self.shape.figure.canvas.mpl_disconnect(cid)


class MyEllipse(Ellipse):
    def set_width(self, w):
        setattr(self, "width", w)

    def set_height(self, h):
        setattr(self, "height", h)

    def get_width(self):
        return getattr(self, "width")

    def get_height(self):
        return getattr(self, "height")


class MyRectangle(Rectangle):
    def set_center(self, xy):
        self.set_xy((xy[0] - self.get_width() / 2, xy[1] - self.get_height() / 2))

    def get_center(self):
        return (
            self.get_xy()[0] + self.get_width() / 2,
            self.get_xy()[1] + self.get_height() / 2,
        )


class InteractivePoint:
    def __init__(self, ax, x, y, w=15.0, c="k", ec="k", zo=4, shape="o"):
        """
        c: color (defaults "k")
        ec: edge color (defaults "k")
        zo: zorder (defaults 4)
        shape: "o" or "s": circle or square (default: "o"
        """
        self.w = w
        if shape == "o":
            patch = MyEllipse((x, y), 1.0, 1.0, color=c, ec=ec, zorder=zo)
        elif shape == "s":
            patch = MyRectangle((x - 0.5, y - 0.5), 1.0, 1.0, color=c, ec=ec, zorder=zo)

        self.shape = shape
        self.point = ax.add_patch(patch)
        self.ax = ax
        self.update()
        self.draggable = Draggable(self.point)
        self.point.figure.canvas.mpl_connect("resize_event", self.update)

    def set_callback(self, press=None, move=None):
        self.draggable.set_callback(press=press, move=move)

    def update(self, *args, **kwargs):
        # Updates the width and height of the patch to keep it
        # a circle/rectangle even if the axes change or the window
        # gets reshaped
        # Note: there might be a simpler way to do this with a transform
        old_center = self.point.get_center()
        print("update")

        dx = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
        dy = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        bbox = self.ax.get_window_extent()
        self.point.set_width(self.w * dx / bbox.width)
        self.point.set_height(self.w * dy / bbox.height)

        self.point.set_center(old_center)


class Animation:
    def __init__(self, start, length, update, onstart=None):
        self.start, self.length, self.update = start, length, update
        self.onstart = onstart
        self.has_started = False
        self.onstart_out = None

    def __call__(self, p):
        if not self.has_started:
            if self.onstart is not None:
                self.onstart_out = self.onstart()
            self.has_started = True

        if self.onstart_out is not None:
            self.update(p, self.onstart_out)
        else:
            self.update(p)

    def apply(self, fig, *args, **kwargs):
        return animate(fig, self, self.length, *args, **kwargs)

    def easing(self, easing):
        return Animation(self.start, self.length, lambda p: self.update(easing(p)))

    @staticmethod
    def concat(animations):
        animations += [Animation(0, 1, dummy)]
        start = min([a.start for a in animations])
        end = max([a.start + a.length for a in animations])
        length = end - start

        def update(p):
            r = []
            for i, a in enumerate(animations):
                sp = (a.start - start) / (length)
                ep = (a.start + a.length - start) / (length)
                if sp < p < ep:
                    r.append(a((p - sp) / (ep - sp)))
            return r

        return Animation(start, length, update)


def animate(fig, update, time=1000, fps=60, init=None, blit=False, repeat=False):
    numframes = int(time / 1000 * fps)
    anim = mpl.animation.FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, 1, numframes),
        init_func=init,
        interval=1000 / fps,
        blit=blit,
        repeat=repeat,
    )
    return anim


def save_anim(anim, name, writerName="ffmpeg", bitrate=1800):
    ms_per_frame = anim._interval
    fps = 1000 * 1 / ms_per_frame
    writer = mpl.animation.writers[writerName](fps=fps, bitrate=bitrate)
    anim.save(name + ".mp4", writer=writer)


def save_anim_to_pdf(anim, name):
    from tqdm import tqdm

    ms_per_frame = anim._interval
    percentages = list(np.linspace(0, 1, anim.save_count))
    for i, p in tqdm(list(enumerate(percentages))):
        anim._func(p)
        save(anim._fig, name + str(i) + ".pdf")


def make_slider(fig, coordinates, label, start, end, init, step):
    """Makes a new axis and puts a slider at the coordinates

    Coordinates format is ``[left,bottom,width,height]``

    ``label``, ``start``, ``end``, ``init`` and ``step`` are passed to the
    slider constructor

    Returns the slider handle
    """
    slider_ax = fig.add_axes(coordinates)
    slider = Slider(slider_ax, label, start, end, valinit=init, valstep=step)
    return slider


def make_checkbox(fig, coordinates, label, default=False):
    """Makes a new axis and puts a slider at the coordinates

    Coordinates format is ``[left,bottom,width,height]``

    ``label`` and ``default`` are passed to the checkbox constructor

    Returns the checkbox handle
    """
    checkbox_ax = fig.add_axes(coordinates)
    checkbox = CheckButtons(checkbox_ax, labels=[label], actives=[default])
    return checkbox
