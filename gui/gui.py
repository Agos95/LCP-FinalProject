import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasAgg
import matplotlib.backends.tkagg as tkagg
import tkinter as Tk

import analysis

# magic function to get a tkinter object from a pyplot Figure to display in the GUI
# copied from PySimpleGUI examples
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
    return photo

############
### MAIN ###
############

fig = plt.figure(figsize = (15, 6))
background = analysis.plot_background()
for ax in background: fig.add_subplot(ax)

figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
layout = [[sg.Text('Select File'), sg.InputText(key='_InputFileName_'), sg.FileBrowse(key='_InputFile_'),
           sg.Submit('Run the Analyses', key='_Analyze_'), sg.ProgressBar(10000, orientation='h', size=(30,20), key='_Progressbar_')],
          [sg.Listbox(values=[],size=(30, 30), bind_return_key=True, disabled=True, key='_EventList_'), sg.Canvas(size=(figure_w, figure_h), key='_Plot_')]]

window = sg.Window('Event Display').Layout(layout)
# need to call Finalize() to use the Canvas
window.Finalize()
# get GUI elements
progress_bar = window.FindElement('_Progressbar_')
list_box = window.FindElement('_EventList_')
plot = window.FindElement('_Plot_')

# in order to display the pyplot Figure in the GUI
fig_photo = draw_figure(plot.TKCanvas, fig)

# Dictionary for all the events in the file: {evNumber: dataframe}
Events = {}

# GUI loop
while True:
    event, values = window.Read(timeout=0)
    # get filename when sent
    data_file = ''

    # action based on GUI events

    ### analyze datafile
    if event == '_Analyze_':
        data_file = values['_InputFileName_']
        with open(data_file) as f:
            # count total events in the file (e.g. total lines) to display the progress bar
            tot_lines = 0
            for line in f: tot_lines += 1
            # set max value in the progress bar
            progress_bar.UpdateBar(current_count=0, max=tot_lines)
            # read the file again from the beginning, create dataframes and save them in the Event dictionary
            f.seek(0) # to set the current position in the file to the first line
            for line_number, line in enumerate(f):
                event = line.split()
                event = [float(i) for i in event]
                dataframe, event_number, hits_number = analysis.read_data(event)
                #sg.Print("Event", event_number, "- Hits:", hits_number)
                Events.update({event_number:dataframe})
                # update progress bar
                progress_bar.UpdateBar(line_number + 1)

        # once the analysis is done, update the list of events in the listbox
        list_box.Update(values=sorted(Events), disabled=False)

    ### display events
    elif event == '_EventList_':
        event_number = values['_EventList_'][0]
        # clear pyplot Figure()
        plt.clf()
        # plot the event
        axes = analysis.plot_events(Events[event_number], event_number)
        for ax in axes: fig.add_subplot(ax)
        fig_photo = draw_figure(plot.TKCanvas, fig)
        # update figure in the GUI
        window.Read()

    ### exit
    elif event is None: break
