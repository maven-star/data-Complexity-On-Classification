import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from Main import Run

sg.change_look_and_feel('LightGreen2')  # look and feel theme

# Designing layout
layout = [[sg.Text("\t\t\tDataset\t          "), sg.Combo(['Abalone', 'Ecoli', 'Glass','Habermanlmb','Iris','New_Thyroid','Page_Blocks','Pimalmb','Segment','Shuttle','Vehicle','Vowel','Wisconsinlmb','Yeast'])],
            [sg.Text("\t\t\tTraining data(%)"), sg.Combo(['60', '70', '80','90'], size=(13, 20)), sg.Text("\n"),

            sg.Button("START", size=(10, 2))],
        [sg.Text(
              "\t          DE algorithm\t            N-measures \t      CP generation \t          F-oversampling    Proposed DC-aware RideNN EC")],
          [sg.Text('Accuracy'), sg.In(key='11', size=(20, 20)), sg.In(key='12', size=(20, 20)),
           sg.In(key='13', size=(20, 20)), sg.In(key='14', size=(20, 20)), sg.In(key='15', size=(20, 20))],
          [sg.Text('TPR\t'), sg.In(key='21', size=(20, 20)), sg.In(key='22', size=(20, 20)),
           sg.In(key='23', size=(20, 20)), sg.In(key='24', size=(20, 20)), sg.In(key='25', size=(20, 20))],
          [sg.Text('TNR\t'), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),
           sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)), sg.In(key='35', size=(20, 20))],
          [sg.Text('\t\t\t\t\t\t\t\t\t\t\t\t            '), sg.Button('Run Graph'), sg.Button('CLOSE')]]


# to plot graphs
def plot_graph(result_1, result_2, result_3):
    plt.figure(dpi=120)
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)

    result = np.transpose(result)

    # labels for bars
    labels = ['Dynamic ensemble algorithm', 'Neighborhood-based measures', 'Classifier pool generation ',
              'Filtering-based oversampling',
              'Proposed Data complexity-aware RideNN ensemble classifier ']  # x-axis labels ############################
    tick_labels = ['Accuracy','TPR', 'TNR']  #### metrics
    bar_width, s = 0.15, 0.0  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i is 0:  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:  # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

    plt.legend(loc=(0.25, 0.25))  # show a legend on the plot -- here legends are metrics
    plt.show()  # to show the plot


# Create the Window layout
window = sg.Window('285520', layout)

# event loop
while True:
    event, values = window.read()  # displays the window
    if event == "START":

        dts,tr=values[0],int(values[1])/100

        print("\n Running..")

        ACC,TPR,TNR = Run.callmain(dts,tr)

        window.Element('11').Update(ACC[0])
        window.Element('12').Update(ACC[1])
        window.Element('13').Update(ACC[2])
        window.Element('14').Update(ACC[3])
        window.Element('15').Update(ACC[4])

        window.Element('21').Update(TPR[0])
        window.Element('22').Update(TPR[1])
        window.Element('23').Update(TPR[2])
        window.Element('24').Update(TPR[3])
        window.Element('25').Update(TPR[4])

        window.Element('31').Update(TNR[0])
        window.Element('32').Update(TNR[1])
        window.Element('33').Update(TNR[2])
        window.Element('34').Update(TNR[3])
        window.Element('35').Update(TNR[4])



    if event == 'Run Graph':
        plot_graph(ACC,TPR, TNR)
    if event == 'CLOSE':
        break
        window.close()
