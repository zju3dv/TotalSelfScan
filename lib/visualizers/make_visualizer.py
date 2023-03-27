import os
import importlib


def make_visualizer(cfg):
    module = cfg.visualizer_module
    visualizer = importlib.import_module(module).Visualizer()
    return visualizer
