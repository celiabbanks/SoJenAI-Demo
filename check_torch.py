import torch
import sys

print("torch module:", torch)
print("torch.__file__:", getattr(torch, "__file__", "NO __file__"))
print("sys.path[0]:", sys.path[0])