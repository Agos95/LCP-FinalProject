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

class Event:
    '''
    Class for handling and analyzing single Events (e.g. a single line in the data file)

    Parameters
    ----------
    event_str : str
        single line of the data file
    isPhysics : bool
        True if physics data, False if calibration

    Attributes
    ----------
    isPhysics : bool
        True if physics data, False if calibration
    dataframe : pandas.dataframe
        Pandas Dataframe with all the information about the Event;
        each row is a single hit recorded in the event
    event_number : int
        number of the Event in the run
    hits_number : int
        number of hits in the Event
    local_fit : list(dict)
        information about local fit (one dictionary per chamber)
    '''

    def __init__(self, event_str, isPhysics=True):

        self.isPhysics = isPhysics
        self.dataframe, self.event_number, self.hits_number = self._Read_Data(event_str)
        #self.local_fit = self._Local_Fit()

    def _Read_Data(self, event_str):
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

    def _Plot_Background(self):
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

    def _Plot_Events(self, dataframe, evNumber):
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
        plots = self._Plot_Background()
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

    def Make_Plot(self):
        '''Plots of the background and the hits'''

        #gridsize = (5, 2)
        plt.figure(figsize = (12, 24))
        self._Plot_Events(self.dataframe, self.event_number)
        plt.show()
        return

    def _Select_Events(self):
        '''Call the Calibration or Physics run function to select events'''

        if self.isPhysics == False:
            return self._Select_Events_Calibration()
        else:
            return self._Select_Events_Physics()

    def _Select_Events_Calibration(self):
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
        dataframe = self.dataframe
        # if we have less than 6 hits or more than 20
        # mark the event as 'bad'
        if (self.hits_number < 6 or self.hits_number > 20):
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

    def _Select_Events_Physics(self):
        '''Select good Events (physics)

        Returns
        -------
        select : bool
            True if the event pass the selection, False otherwise
        chambers : list(int)
            list with the number of the chambers where we find the hits
        n_layer : list(int)
            list with the number of different
        '''

        dataframe = self.dataframe
        # hits in less than 4 chambers
        if dataframe['Chamber'].nunique() < 4:
            select=False
            chambers=[]
            n_layer=[]
            return select, chambers, n_layer
        # hits in all the 4 chambers
        else:
            chambers=[0,1,2,3]
            n_layer=[]
            for n_ch in chambers:
                # compute number of different layers in each chamber
                n_layer_ch = dataframe[dataframe['Chamber']==n_ch]['Layer'].nunique()
                n_layer.append(n_layer_ch)

                # require at least 3 different layers for each chamber
                if n_layer_ch<3: # break if hits in chamber are less than 3 layers
                    select=False
                    chambers = []
                    n_layer = []
                    break
                else:
                    select=True
            return select, chambers, n_layer

    def _Local_Fit(self):
        #Perform local fit in every chamber

        dataframe = self.dataframe
        select, list_chambers, list_layers = self._Select_Events()
        # list to store results for each chamber
        results=[]#{"LocalCoordinates", "GlobalCoordinates"}
        # loop over the (two) chambers
        for i in range(0,len(list_chambers)):
        # if we have 4 different layers we randomly select a layer to be excluded
        # we will use the point from the excluded layer to check the goodness of the global fit
            if(list_layers[i]==4):
                rand_layer=np.random.randint(1,4)
            else:
                rand_layer=0 # layers are 1,2,3,4: excluding layer 0 is equivalent to keeping them all

            # create dataframe_cl filtered by chamber and excluded layer
            dataframe_c  = dataframe[dataframe['Chamber']==list_chambers[i]] # dataframe filtered by chamber
            dataframe_cl = dataframe_c[dataframe_c['Layer']!=rand_layer]     # filtered by chamber and excluded layer

            # Z local coordinates corresponding to the 4 different layers
            Z=[6.5,19.5, 32.5, 45.5]

            # create a list l containing 3 lists of points (z,x), one for each selected layer
            l=[]

            # loop over selected layers and fill l
            for layer_index in dataframe_cl['Layer'].unique():
                XR=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XR_local'])
                XL=np.array(dataframe_cl[dataframe_cl['Layer']==layer_index]['XL_local'])

                z=Z[(layer_index - 1)] # layer_index is in range [1,4], list index must be in range [0,3]
                l_temp=[]

                for x in XR:
                    l_temp.append((z,x))
                for x in XL:
                    l_temp.append((z,x))
                l.append(l_temp)

            # create numpy array with all possible combinations of 3 points p1,p2,p3
            combinations=np.array([(p1,p2,p3) for p1 in l[0] for p2 in l[1] for p3 in l[2]])

            # interpolate each combination and select the combination with least chi squared
            min_chisq=100000 # to store minimum chisq
            optimal_comb=np.zeros((3,2)) # to store best combination of points
            slope_opt=0 # to store slope obtained with the best combination
            intercept_opt=0 # to store intercept obtained with the best combination
            for points in combinations:
                # linear regression
                slope, intercept, r_value, p_value, std_err=stats.linregress(points[:,0],points[:,1])
                # compute expected x using the interpolating function
                expect_x=intercept+slope*(points[:,0])
                # compute chi squared
                chisq, p_value=stats.chisquare(points[:,1],expect_x)
                # eventually update min_chisq and optimal_comb
                if(chisq<min_chisq):
                    min_chisq=chisq
                    optimal_comb=points
                    slope_opt=slope
                    intercept_opt=intercept
                else:
                    continue

            #add to results: results is a list of 2 dictionaries, one for each chamber
            results.append({"slope":slope_opt,
                            "intercept":intercept_opt,
                            "optimal_comb": optimal_comb,
                            "excl_layer": rand_layer})

        return results

    def _Local_Fit_Agostini(self):
        """Perform local fit in every chamber"""

        def make_fit(df):
            '''Local fit

            Parameters
            ----------
            df : pandas.dataframe
                dataframe containing the hits in a single chamber

            Returns
            -------
            model : statsmodel.RegressionResults
                summary of linear regression model
            chisq : float
                chi^2 of the regression model
            '''
            l = []
            for layer_index in df['Layer'].unique():
                print("> Analyzing Layer =", layer_index)
                XR=np.array(df[df['Layer']==layer_index]['XR_global'])
                XL=np.array(df[df['Layer']==layer_index]['XL_global'])
                Z =np.array(df[df['Layer']==layer_index]['Z_global' ])
                l_layer = []
                for xr, xl, z in zip(XR, XL, Z):
                    l_layer.append((z, xr))
                    l_layer.append((z, xl))
                l.append(l_layer)
            # create numpy array with all possible combinations of 3 points p1,p2,p3
            combinations = np.array([(p1,p2,p3) for p1 in l[0] for p2 in l[1] for p3 in l[2]])
            # interpolate each combination and select the combination with make_fit(df_chamber_reduced)(df_chamber_reduced)ast chi squared
            min_chisq = 100000 # to store minimum chisq
            for points in combinations:
                # linear regression
                ### use (z,x) because I assume the value in z axis to be without error
                Z = points[:,0]
                Z = sm.add_constant(Z) # add costant term in order to have the intercept
                X = points[:,1]
                lin_reg = sm.OLS(X,Z).fit()
                predict_X = lin_reg.predict()
                # compute chi squared
                chisq, _ = stats.chisquare(X, predict_X)
                # eventually update min_chisq and optimal_comb
                if(chisq < min_chisq):
                    min_chrand_layer=0 #layers are isq = chisq
                    model = lin_reg # save model with smaller chi^2
                else:
                    continue
            return model, chisq
            ####################################################################################

        dataframe = self.dataframe
        select, list_chambers, list_layers = self._Select_Events()

        if select == True:
            grouped_chamber = dataframe.groupby("Chamber")
            results = {}
            for n_chamber, df_chamber in grouped_chamber:
                print("### n_chamber =", n_chamber, "###")
                n_layer = df_chamber["Layer"].nunique()
                print("## n_layer =", n_layer, "##")
                # if we have hits in 4 layers, we use 3 of them to do the interpolation,
                # and the last one to check the goodness
                if n_layer == 4:
                    print("# There are 4 layers")
                    chisq = 100000
                    model = None
                    for num_layer in np.arange(1,5): # layer index is in [1,2,3,4]
                        print("Excluding layer", num_layer)
                        df_chamber_reduced = df_chamber[df_chamber['Layer'] != num_layer]
                        print(df_chamber_reduced)
                        model_tmp, chisq_tmp = make_fit(df_chamber_reduced)
                        if chisq_tmp < chisq:
                            chisq = chisq_tmp
                            model = model_tmp
                        else: continue
                    results[n_chamber] = {'model':model, 'chisq':chisq}

                else:
                    model, chisq = make_fit(df_chamber)
                    results[n_chamber] = {'model':model, 'chisq':chisq}
            return results
        else:
            return None


        return