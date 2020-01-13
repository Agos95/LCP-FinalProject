import PySimpleGUI as sg
#sg.theme("Topanga")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Functions_GUI import *

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

font = ("bitstream charter", 14, "")

fig = plt.figure(figsize = (20,10))
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

listbox_widget = sg.Listbox(values=[], select_mode='LISTBOX_SELECT_MODE_SINGLE', size=(10, 30), bind_return_key=True, disabled=True, key='_EventList_', font=font)
canvas_widget  = sg.Canvas(size=(figure_w, figure_h), key='_Plot_')
checkbox_widget = [sg.Radio("Calibration", "RADIO", default=True, auto_size_text=True, key="_Calibration_", font=font), sg.Radio("Physics", "RADIO", default=False, auto_size_text=True, font=font)]
column_widget = sg.Column([[listbox_widget], [sg.Submit('Analyze', key="_Analyze_", font=font, disabled=True)]])

layout = [
    [sg.Text('Select File', font=font), sg.InputText(key='_InputFileName_', font=font), sg.FileBrowse(key='_InputFile_', font=font),
     sg.Submit('Load Datafile', key='_Load_', font=font), checkbox_widget[0], checkbox_widget[1]],
     [column_widget, canvas_widget],
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
    # close window
    if event in (None, 'Exit'):
        break

    # load datafile
    elif event == "_Load_" and values['_InputFileName_']:
        data_file = values['_InputFileName_']
        if values['_Calibration_'] :
            Ev_list = Open_File_Calibration(data_file)
        else :
            Ev_list = Open_File_Physics(data_file)
        # once the analysis is done, update the list of events in the listbox
        highlight = [i for i in range(len(Ev_list)) if Ev_list[i]["Accepted"]]
        list_box.Update(values=[str(i) for i in range(len(Ev_list))], disabled=False, set_to_index=highlight)
        window["_Analyze_"].Update(disabled=False)

    # plot selected event
    elif event == "_EventList_":
        # ** IMPORTANT ** Clean up previous drawing before drawing again
        if figure_agg: figure_agg.get_tk_widget().forget()
        n = int(values['_EventList_'][0])
        fig = Make_Plot_GUI(Ev_list[n], values['_Calibration_'])
        figure_agg = draw_figure(plot_el.TKCanvas, fig)  # draw the figure
        list_box.Update(set_to_index=highlight)
        m = n

    elif event == "_Analyze_" and Ev_list:
        if values['_Calibration_'] :
            Calibration(Ev_list)
        else:
            Physics(Ev_list)

window.close()
