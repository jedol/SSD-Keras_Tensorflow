import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)

## parameters
PROJECT_ROOT = os.path.abspath(os.path.join(this_dir, '..'))

## data path
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

## experiments path
EXP_PATH = os.path.join(PROJECT_ROOT, 'experiments')
if not os.path.isdir(EXP_PATH):
    os.makedirs(EXP_PATH)

## library path
LIB_PATH = os.path.join(PROJECT_ROOT, 'lib')
if not os.path.isdir(LIB_PATH):
    os.makedirs(LIB_PATH)

## library path
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')

## add library to PYTHONPATH
add_path(LIB_PATH)
