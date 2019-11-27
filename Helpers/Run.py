import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random
from scipy import stats
from tqdm.autonotebook import tqdm

import Helpers.Event as EV

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
        self.isPhysics     = isPhysics
        self.data_path     = data_path
        self.Event_List    = []
        self.Event_Numbers = []
        self.Event_Hits    = []

    def read_events(self):
        '''Read each line from the file and store the 'Event' object in a list'''

        tot_ev = 0
        with open(self.data_path) as f:
            tot_ev = len(list(f))
        with open(self.data_path) as f:
            with tqdm(total=tot_ev) as pbar:
                for line in f:
                    ev = EV.Event(line, isPhysics=self.isPhysics)
                    self.Event_List.append(ev)
                    self.Event_Numbers.append(ev.event_number)
                    self.Event_Hits.append(ev.hits_number)
                    pbar.update()
        return

    def Plot_Event(self, event_number):
        if event_number not in self.Event_Numbers:
            print("Event number not present in the Run")
            return
        # get index corresponding to the ev number
        idx = self.Event_Numbers.index(event_number)
        ev  = self.Event_List[idx]
        ev.Make_Plot()
        return
