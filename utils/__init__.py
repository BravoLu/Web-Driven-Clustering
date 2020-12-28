import yaml
import os
import sys
from .progress.progress.bar import Bar as Bar
from .logging import *
from .meters import *
from .osutils import *
from .re_ranking import *
from .serialization import * 

sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))

def load_configs(cfg):
    name = cfg
    cfg = open(os.path.join("configs", cfg+'.yml'), 'r')
    cfg = yaml.load(cfg)
    cfg['NAME'] = name
    return cfg

