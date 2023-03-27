import os
import importlib


def make_renderer(cfg, network):
    module = cfg.renderer_module
    renderer = importlib.import_module(module).Renderer(network)
    return renderer
