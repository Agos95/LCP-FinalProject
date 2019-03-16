#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

get_ipython().run_line_magic('run', 'Helper.py')


# In[2]:


data_file = "D:/Desktop/University/Magistrale/Anno1_Sem1/LabOfComputationalPhysics/LabFiles/FinalProject/data_merged/calibration/Run000260.txt"
with open(data_file) as f:
    line = f.readline()
    line = f.readline()
    line = f.readline()
    line = f.readline()
    event = line.split()
    #event = [float(i) for i in event]
    print(event)
    


# In[3]:


ev = Event(event)
print("Event Number:", ev.event_number)
print("# Hits:", ev.hits_number)
ev.dataframe


# In[4]:


ev.Make_Plot()


# In[5]:


ev.event_number, ev.hits_number


# In[7]:


ev.local_fit


# In[ ]:




