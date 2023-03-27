import os
import importlib


def make_network(cfg):
    module = cfg.network_module
    network = importlib.import_module(module).Network()
    return network
