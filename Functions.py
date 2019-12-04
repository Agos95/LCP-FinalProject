import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api   as sm
from matplotlib.patches import Polygon
from scipy import stats

# Cell dimensions
XCELL = 42.
ZCELL = 13.

# X coordinates translation
global_x_shifts = [994.2, 947.4,-267.4,-261.5,]

# Z coordinates translations
local_z_shifts = [z*ZCELL for z in range(0,4)]
global_z_shifts = [823.5, 0, 823.5, 0]

def Read_Data(event_str):
    '''Read input event as string and create the Dataframe

    Parameters
    ----------
    event_str : str
        Single line of the data file

    Returns
    -------
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    event_number : int
        number of the Event in the run
    hits_number : int
        number of hits in the Event
    '''

    event        = event_str.split()
    event_number = int(event[0])
    hits_number  = int(event[1])
    if hits_number == 0:
        hit       = [np.nan]
        chamber   = [np.nan]
        layer     = [np.nan]
        xl_local  = [np.nan]
        xr_local  = [np.nan]
        z_local   = [np.nan]
        time      = [np.nan]
        xl_global = [np.nan]
        xr_global = [np.nan]
        z_global  = [np.nan]
    else:
        hit       = np.arange(hits_number)
        chamber   = np.fromiter((event[2+5*i] for i in range(hits_number)), int)
        layer     = np.fromiter((event[3+5*i] for i in range(hits_number)), int)
        xl_local  = np.fromiter((event[4+5*i] for i in range(hits_number)), float)
        xr_local  = np.fromiter((event[5+5*i] for i in range(hits_number)), float)
        z_local   = np.fromiter((local_z_shifts[i-1]+ZCELL/2 for i in layer), float)
        time      = np.fromiter((event[6+5*i] for i in range(hits_number)), float)
        xl_global = np.fromiter((global_x_shifts[i] for i in chamber), float) - xl_local
        xr_global = np.fromiter((global_x_shifts[i] for i in chamber), float) - xr_local
        z_global  = np.fromiter((global_z_shifts[i] for i in chamber), float) + z_local
    dataframe = pd.DataFrame({
        'EvNumber' : event_number,
        'Hit'      : hit,
        'Chamber'  : chamber,
        'Layer'    : layer,
        'XL_local' : xl_local,
        'XR_local' : xr_local,
        'Z_local'  : z_local,
        'Time'     : time,
        'XL_global': xl_global,
        'XR_global': xr_global,
        'Z_global' : z_global,
        })
    return dataframe, event_number, hits_number

def Plot_Background():
    '''Makes the plot for the background of the event display

    Parameters
    ----------

    Returns
    -------
    axes : list(pyplot.axes)
        background images of the detector and the chambers
    '''

    # create Pandas DataFrame for the chambers positions
    chamber_position = pd.DataFrame({
    'chamber' : [i for i in range(4)],
    'x_vertices' : [(global_x_shifts[i], global_x_shifts[i] - 720, global_x_shifts[i] - 720, global_x_shifts[i])
                    for i in range(4)],
    'y_vertices' : [(global_z_shifts[i], global_z_shifts[i], global_z_shifts[i] + 52, global_z_shifts[i] + 52)
                    for i in range(4)],
    })
    x_lim = [[-1000, 1000], # global detector
                [    0, 1000], # chamber 0
                [    0, 1000], # chamber 1
                [-1000,    0], # chamber 2
                [-1000,    0]] # chamber 3
    y_lim = [[-100, 1000],  # global detector
                [800 ,  900],  # chamber 0
                [ -25,   75],  # chamber 1
                [ 800,  900],  # chamber 2
                [ -25,   75]]  # chamber 3
    title = ["DETECTOR", "Chamber 0", "Chamber 1", "Chamber 2", "Chamber 3"]
    # create pyplot 'Axes' objects
    gridsize = (5,2)
    ax_global = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax_0 = plt.subplot2grid(gridsize, (2, 1), colspan=1, rowspan=1) # top-right
    ax_1 = plt.subplot2grid(gridsize, (3, 1), colspan=1, rowspan=1) # bottom-right
    ax_2 = plt.subplot2grid(gridsize, (2, 0), colspan=1, rowspan=1) # top-left
    ax_3 = plt.subplot2grid(gridsize, (3, 0), colspan=1, rowspan=1) # bottom-left

    axes = [ax_global, ax_0, ax_1, ax_2, ax_3]
    for index, ax in enumerate(axes):
        ax.set_xlim(x_lim[index])
        ax.set_ylim(y_lim[index])
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        if index == 0: ax.set_title(title[index])
        else: ax.set_title(title[index], pad=-20)
        # plot the 4 chambers in each 'Axes'
        for j in range(4):
            chamber = chamber_position[chamber_position["chamber"] == j]
            ax.fill(chamber["x_vertices"].values[0], chamber["y_vertices"].values[0], color='gray', fill=False)
    return axes

def Plot_Events(dataframe, evNumber):
    '''Plot the positions of the hits

    Parameters
    ----------
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    event_number : int
        number of the Event in the run

    Returns
    -------
    _event_display : list(pyplot.axes)
        images of the hits of the events
    '''

    # get the EvNumber as argument, because, if the dataframe is empty,
    # I can't get it from data
    plots = Plot_Background()
    plots[0].set_title("Event:"+str(evNumber), {'size':'18'})
    if dataframe.empty == False:
        xL = dataframe["XL_global"]
        xR = dataframe["XR_global"]
        z  = dataframe["Z_global"]
        for image in plots:
            image.plot(xL, z, "bo", markersize=3)
            image.plot(xR, z, "ro", markersize=3)
    #local_fit = self.local_fit
    return plots

def Make_Plot(dataframe, event_number):
    '''Plots of the background and the hits'''

    gridsize = (5, 2)
    plt.figure(figsize = (12, 24))
    Plot_Events(dataframe, event_number)
    plt.show()
    return

def Select_Events_Calibration(dataframe, hits_number):
    '''Select good Events (calibration)

    Returns
    -------
    select : bool
        True if the event pass the selection, False otherwise
    chambers : list(int)
        list with the number of the chambers where we find the hits
    n_layer : list(int)
        list with the number of different
    '''

    # hits only in the right side
    # if we have less than 6 hits or more than 20
    # mark the event as 'bad'
    if (hits_number < 6 or hits_number > 20):
        select=False
        chambers=[]
        n_layer=[]
        return select, chambers, n_layer

    elif((dataframe['Chamber']<=1).all()):
        chambers=[0,1]
        # compute number of different layers in each chamber
        n_layer_ch0 = dataframe[dataframe['Chamber']==0]['Layer'].nunique()
        n_layer_ch1 = dataframe[dataframe['Chamber']==1]['Layer'].nunique()

        n_layer=[n_layer_ch0, n_layer_ch1]

        # require at least 3 different layers for each chamber
        if(n_layer_ch0>=3 and n_layer_ch1>=3):
            select=True
        else:
            select=False

        return select, chambers, n_layer

    # hits only in the left side
    elif((dataframe['Chamber']>=2).all()):
        chambers=[2,3]
        # compute number of different layers in each chamber
        n_layer_ch2 = dataframe[dataframe['Chamber']==2]['Layer'].nunique()
        n_layer_ch3 = dataframe[dataframe['Chamber']==3]['Layer'].nunique()

        n_layer=[n_layer_ch2, n_layer_ch3]

        # require at least 3 different layers for each chamber
        if(n_layer_ch2>=3 and n_layer_ch3>=3):
            select=True
        else:
            select=False

        return select, chambers, n_layer

    # hits in both left and right side
    else:
        select=False
        chambers=[]
        n_layer=[]
        return select, chambers, n_layer

