
import tkinter as tk
import time

class SampleApp(tk.Tk):
    str = ''

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.clock = tk.Label(self, text="", font=('Verdana', 40, 'bold'))
        self.clock.pack()

        # start the clock "ticking"
        self.update_clock()

    def update_clock(self):
        self.clock.configure(text=self.str)
        # call this function again in one second
        self.after(100, self.update_clock)


