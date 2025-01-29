import torch
import glob
import tkinter as tk
from tkinter import ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

def plot_gradients_with_tabs(epoch):
    root = tk.Tk()
    root.title(f"Gradient Distributions for Epoch {epoch}")

    tab_control = ttk.Notebook(root)

    colormap = matplotlib.colormaps['viridis']

    for layer in os.listdir('./Gradients'):
        tab = ttk.Frame(tab_control)
        tab_control.add(tab, text=layer)

        fig, ax = plt.subplots(figsize=(10, 6))
        grad_files = sorted(glob.glob(f"Gradients/{layer}/*.pt"))
        # print(f"GRAD FILES = {grad_files}")
        num_batches = len(grad_files)

        for batch_idx, grad_file in enumerate(grad_files):
            grad = torch.load(grad_file, weights_only=True)
            grad_values = grad.view(-1).cpu().numpy()

            ax.hist(
                grad_values, bins=50, alpha=0.5,
                color=colormap(batch_idx / num_batches),
                label=f"Batch {grad_file.split('/')[2].split('.')[0]}", density=True
            )

        ax.set_title(f"Layer: {layer}", fontsize=12)
        ax.set_xlabel("Gradient Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill=tk.BOTH)
        canvas.draw()

    tab_control.pack(expand=True, fill=tk.BOTH)

    root.mainloop()

epoch_to_plot = 0
# layer_names_to_plot = ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
plot_gradients_with_tabs(epoch_to_plot)
