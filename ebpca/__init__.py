import os
from collections import namedtuple

if not os.path.exists("figures"):
    os.makedirs("figures")

from . import amp
from . import pca
from . import empbayes
from . import preprocessing
