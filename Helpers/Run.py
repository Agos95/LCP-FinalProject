import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
from scipy import stats
import Event as EV

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
    data_path : str
        path to the datafile

    isPhysics : bool
        True if physics run, False if calibration

    Attributes
    ----------
    isPhysics : bool
        True if physics run, False if calibration

    data_path : str
        path to the datafile
    '''

    def __init__(self, data_path, isPhysics):
        self.isPhysics = isPhysics
        self.data_path = data_path
        self.Event_List = []

    def read_events(self):
    '''Read each line from the file and store the 'Event' object in a list'''

        with open(self.data_path) as f:
            for line in f:
                ev = EV.Event(line, isPhysics=self.isPhysics)
                self.Event_List.append(ev)
        return


