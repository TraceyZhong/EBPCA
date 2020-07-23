import os

if not os.path.exists("figures"):
    os.makedirs("figures")

from . import amp
from . import pca
from . import empbayes

