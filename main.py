# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
get_ipython().run_line_magic('matplotlib', 'inline')
import glob

import Helpers.Event as EV
import Helpers.Run   as RUN


# %%
data_file = "../data_merged/calibration/Run000260.txt"
run = RUN.Run(data_file, isPhysics=False)

run.read_events()

# %%
print("Event Number:", ev.event_number, "\n# of hits:", ev.hits_number)
ev.dataframe


# %%
ev.Make_Plot()


# %%
data = ev.dataframe
data


# %%
test = ev._Local_Fit()


# %%
type(test)


# %%
test


# %%
test[2]['model'].pvalues


# %%


