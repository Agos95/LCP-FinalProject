import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Functions import *

def Open_File_Calibration(data_file):
    with open(data_file) as f:
        tot_ev = len(list(f))

    # loop over events and perform analysis
    Ev_list = []
    selected_ev = 0
    with open(data_file) as f:
        for line in f:
            # read event
            ev, evNum, hits = Read_Data(line)
            # select event
            sel, chambers, n_layers = Select_Events_Calibration(ev, hits)

            if sel: selected_ev += 1

            # save most important information, in order to plot or reperform the analysis
            # without reading the whole file again
            Ev_list.append(
                {
                    "Number"    : evNum,
                    "Dataframe" : ev,
                    "Hits"      : hits,
                    "Accepted"  : sel,
                    "Chambers"  : chambers,
                    "Layers"    : n_layers
                }
            )
            sg.OneLineProgressMeter('My Meter', len(Ev_list), tot_ev, 'key', 'Reading CALIBRATION file', orientation="h")

    print("{:35} = {:d}"    .format("Total number of events in the Run", tot_ev))
    print("{:35} = {:d}"    .format("Number of accepted events"        , selected_ev))
    print("{:35} = {:.4f} %".format("Fraction of accepted events"      , selected_ev/tot_ev*100))
    return Ev_list

def Open_File_Physics(data_file):
    with open(data_file) as f:
        tot_ev = len(list(f))

    # loop over events and perform analysis
    Ev_list = []
    selected_ev = 0
    with open(data_file) as f:
        for line in f:
            # read event
            ev, evNum, hits = Read_Data(line)
            # filter by hit position
            ev_left, ev_right, hits_left, hits_right = Points_Filter(ev)
            # select event
            sel_left,  chambers_left,  n_layers_left  = Select_Events_Calibration(ev_left,  hits_left)
            sel_right, chambers_right, n_layers_right = Select_Events_Calibration(ev_right, hits_right)

            if sel_left and sel_right: selected_ev += 1

            # save most important information, in order to plot or reperform the analysis
            # without reading the whole file again
            Ev_list.append(
                {
                    "Number"    : evNum,
                    "Dataframe" : ev,
                    "Hits"      : hits,
                    "Accepted"  : sel_left and sel_right,
                    "Chambers"  : chambers_left+chambers_right,
                    "Layers"    : [n_layers_left, n_layers_right]
                }
            )
            sg.OneLineProgressMeter('My Meter', len(Ev_list), tot_ev, 'key', 'Optional message', orientation="h")

    print("{:35} = {:d}"    .format("Total number of events in the Run", tot_ev))
    print("{:35} = {:d}"    .format("Number of accepted events"        , selected_ev))
    print("{:35} = {:.4f} %".format("Fraction of accepted events"      , selected_ev/tot_ev*100))
    return Ev_list

def Make_Plot_GUI(ev, calibration=False):
    '''Plots of the background and the hits

    Parameters
    ----------
    ev : dict
        dictionary with information about the event; it contains: Number, Dataframe, Hits, Accepted, Chambers, Layers
    calibration : bool
        wether the event is taken during calibration or physics run
    '''

    plt.figure(figsize = (20,10))
    plots = Plot_Events_GUI(ev["Dataframe"], ev["Number"])
    if ev["Accepted"]: Plot_Fit(ev, plots, calibration=calibration)
    return plt.gcf()

def Plot_Events_GUI(dataframe, evNumber):
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
    event_display : list(pyplot.axes)
        images of the hits of the events
    '''

    # get the EvNumber as argument, because, if the dataframe is empty,
    # I can't get it from data
    plots = Plot_Background_GUI()
    plots[0].set_title("Event: {:d}".format(evNumber), {'size':'18'})
    if dataframe.empty == False:
        xL = dataframe["XL_global"]
        xR = dataframe["XR_global"]
        z  = dataframe["Z_global"]
        for image in plots:
            image.plot(xL, z, "bo", markersize=3)
            image.plot(xR, z, "ro", markersize=3)

    return plots

def Plot_Background_GUI():
    '''Makes the plot for the background of the event display

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
    gridsize = (7, 10)
    ax_global = plt.subplot2grid(gridsize, (0, 0), colspan=4, rowspan=7)
    ax_0 = plt.subplot2grid(gridsize, (0, 8), colspan=2, rowspan=3) # top-right
    ax_1 = plt.subplot2grid(gridsize, (4, 8), colspan=2, rowspan=3) # bottom-right
    ax_2 = plt.subplot2grid(gridsize, (0, 5), colspan=2, rowspan=3) # top-left
    ax_3 = plt.subplot2grid(gridsize, (4, 5), colspan=2, rowspan=3) # bottom-left

    axes = [ax_global, ax_0, ax_1, ax_2, ax_3]

    if exists("./wire_pos_glob.txt"):
        wires = np.loadtxt("./wire_pos_glob.txt")
    else: wires = None

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
        if wires is not None:
            ax.plot(wires[:,0], wires[:,1], marker=".", markersize=.5, linestyle="", color="gray")
    return axes

# magic function to get a tkinter object from a pyplot Figure to display in the GUI
# copied from PySimpleGUI examples
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

############
### MAIN ###
############

#table_header = ['EvNumber', 'Hit', 'Chamber', 'Layer', 'XL_local', 'XR_local', 'Z_local', 'Time', 'XL_global', 'XR_global', 'Z_global']
#table_values = ['', '', '', '', '', '', '', '', '', '', '']

font = ("Courier", 14, "")

fig = plt.figure(figsize = (20,10))
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

listbox_widget = sg.Listbox(values=[], select_mode='LISTBOX_SELECT_MODE_SINGLE', size=(10, 35), bind_return_key=True, disabled=True, key='_EventList_', font=font)
canvas_widget  = sg.Canvas(size=(figure_w, figure_h), key='_Plot_')
checkbox_widget = [sg.Radio("Calibration", "RADIO", default=True, auto_size_text=True, key="_Calibration_", font=font), sg.Radio("Physics", "RADIO", default=False, auto_size_text=True, font=font)]
column_widget = sg.Column([[listbox_widget], [sg.Checkbox("Checkbox", auto_size_text=True)]])

layout = [[sg.Text('Select File', font=font), sg.InputText(key='_InputFileName_', font=font), sg.FileBrowse(key='_InputFile_', font=font),
           sg.Submit('Load Datafile', key='_Load_', font=font), checkbox_widget[0], checkbox_widget[1]],
          [column_widget, canvas_widget],
         #[sg.Table(values=table_values, headings=table_header, key="_Table_")]
         ]

window = sg.Window('Event Display', layout, resizable=True, finalize=True)
# get GUI elements
list_box = window['_EventList_']
plot_el = window['_Plot_']

figure_agg = None
data_file = ""
Ev_list = []
highlight = []

while True:
    event, values = window.read()
    # highlight accepted events
    if highlight: list_box.Update(set_to_index=highlight)

    # close window
    if event in (None, 'Exit'):
        break

    # load datafile
    elif event == "_Load_" and values['_InputFileName_']:
        #window.Disable()
        data_file = values['_InputFileName_']
        if values['_Calibration_'] :
            Ev_list = Open_File_Calibration(data_file)
        else :
            Ev_list = Open_File_Physics(data_file)
        # once the analysis is done, update the list of events in the listbox
        highlight = [i for i in range(len(Ev_list)) if Ev_list[i]["Accepted"]]
        list_box.Update(values=[str(i) for i in range(len(Ev_list))], disabled=False, set_to_index=highlight)
        #window.Enable()

    # plot selected event
    elif event == "_EventList_":
        # ** IMPORTANT ** Clean up previous drawing before drawing again
        if figure_agg: figure_agg.get_tk_widget().forget()
        n = int(values['_EventList_'][0])
        fig = Make_Plot_GUI(Ev_list[n], values['_Calibration_'])
        figure_agg = draw_figure(plot_el.TKCanvas, fig)  # draw the figure

window.close()
