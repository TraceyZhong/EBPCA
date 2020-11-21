import os
from collections import namedtuple

if not os.path.exists("figures"):
    os.makedirs("figures")

from . import amp
from . import pca
from . import empbayes
<<<<<<< HEAD
from . import preprocessing
=======
from . import preprocessing
>>>>>>> 74e4318... reset the algorithm as the paper
