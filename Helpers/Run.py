import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
from scipy import stats
import ./Event as EV

# Cell dimensions
XCELL = 42.
ZCELL = 13.

# X coordinates translation
global_x_shifts = [994.2, 947.4,-267.4,-261.5,]

# Z coordinates translations
local_z_shifts = [z*ZCELL for z in range(0,4)]
global_z_shifts = [823.5, 0, 823.5, 0]

class Run:
    '''
    Class for handling and analyzing the entire run 
    
    Parameters
    ----------
    run_path : str
        path to the datafile

    isPhysics : bool
        True if physics run, False if calibration

    Attributes
    ----------
    isPhysics : bool
        True if physics run, False if calibration
    '''

    def __init__(self, run_path, isPhysics):
        self.isPhysics = isPhysics


