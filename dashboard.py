import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys
import getopt
import streamlit as st
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import numpy as np
import configparser as cp

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr
from support_modules.readers import log_reader as lr

#---Workaround for "tensorflow.python.framework.errors_impl.UnknownError: Fail to find the dnn implementation."
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
#-----


st.set_page_config(layout="wide")
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-o': 'one_timestamp', '-a': 'activity',
              '-f': 'file_name', '-i': 'imp', '-l': 'lstm_act',
              '-d': 'dense_act', '-p': 'optim', '-n': 'norm_method',
              '-m': 'model_type', '-z': 'n_size', '-y': 'l_size',
              '-c': 'folder', '-b': 'model_file', '-x': 'is_single_exec',
              '-t': 'max_trace_size', '-e': 'splits', '-g': 'sub_group',
              '-v': 'variant', '-r': 'rep'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


# --setup--
def main(argv):

    #Dashboard Title
    st.title("Next Event Activity Prediction Dashboard") #Adding title bar
    st.sidebar.title("App Control Menu")  #Adding the header to the sidebar as well as the sidebar
    st.markdown("This dashboard is used to *predict* and *recommend* next event for the provided eventlog")

    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parameters['one_timestamp'] = True  # Only one timestamp in the log
    # Similarity btw the resources profile execution (Song e.t. all)
    #parameters['rp_sim'] = 0.85
    #parameters['batch_size'] = 128 # Usually 16/32/64/128/256
    #parameters['epochs'] = 200 #v1 200, for embedded training it's 100.
    # Parameters setting manual fixed or catched by console

    st.sidebar.subheader("Choose Activity")
    mode_sel = st.sidebar.selectbox("Mode", ('batch', 'next'), )
    #dropdown list and the options are in the tuples of string
    classifier = st.sidebar.selectbox("Activity", ('predict_next','pred_log', 'pred_sfx'), )

    if not argv:
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next
        parameters['mode'] = mode_sel
        parameters['activity'] = classifier  # Change Here
        # Event-log reading parameters
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            #'timeformat': '%d-%m-%Y %H:%M',
            #'2014-09-19 06:14:00'
            #'timeformat': '%H:%M.%S',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}

        # Folder picker button
        if parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            variant_opt = st.sidebar.selectbox("Variant", ('random_choice', 'arg_max', 'top3'), key='variant_opt')
            parameters['variant'] = variant_opt  # random_choice, arg_max for variants and repetitions to be tested

            iskey = st.sidebar.radio("Single Exec", (True, False), key='iskey')
            parameters['is_single_exec'] = iskey  # single or batch execution

            parameters['rep'] = 1
            ml_model = st.sidebar.file_uploader("Upload ML Model", type=['h5'])
            if ml_model is not None:
                parameters['model_file'] = ml_model.name

            # Set up tkinter
            root = tk.Tk()
            root.withdraw()
            # Make folder picker dialog appear on top of other windows
            root.wm_attributes('-topmost', 1)

            #Option to choose the folder where model and test log is kept

            #st.sidebar.title('Folder Picker')
            #st.sidebar.write('Please select a folder:')
            #clicked = st.sidebar.button('Folder Picker')
            #if clicked:
            #    dirname = st.sidebar.text_input('Selected folder:', filedialog.askdirectory(master=root))
            #    path, folder_name = os.path.split(dirname)
            #    parameters['folder'] = folder_name
            #    print("folder_name : ", folder_name)

            folder_name  = st.sidebar.text_input('Folder Name')
            parameters['folder'] = folder_name

        else:
            raise ValueError(parameters['activity'])
    else:
        # Catch parameters by console
        try:
            opts, _ = getopt.getopt(
                argv,
                "ho:a:f:i:l:d:p:n:m:z:y:c:b:x:t:e:v:r:",
                ['one_timestamp=', 'activity=',
                 'file_name=', 'imp=', 'lstm_act=',
                 'dense_act=', 'optim=', 'norm_method=',
                 'model_type=', 'n_size=', 'l_size=',
                 'folder=', 'model_file=', 'is_single_exec=',
                 'max_trace_size=', 'splits=', 'sub_group=',
                 'variant=', 'rep='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parameters[key] = None
                elif key in ['is_single_exec', 'one_timestamp']:
                    parameters[key] = arg in ['True', 'true', 1]
                elif key in ['imp', 'n_size', 'l_size',
                             'max_trace_size','splits', 'rep']:
                    parameters[key] = int(arg)
                else:
                    parameters[key] = arg
            parameters['read_options'] = {'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
                                          'column_names': column_names,
                                          'one_timestamp':
                                              parameters['one_timestamp'],
                                              'ns_include': True}
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)

    #   Execution

    if parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        if parameters['mode'] in ['next']:
            caseid_output_route = os.path.join('output_files', parameters['folder'], 'parameters', 'test_log.csv')
            col_list = ["caseid"]
            caseid_output = pd.read_csv(caseid_output_route, usecols=col_list)
            caseid_next = st.selectbox("Select Case ID", caseid_output["caseid"].unique())
            parameters['nextcaseid'] = caseid_next
            if st.button("Process"):
                print(parameters)
                print(parameters['folder'])
                print(parameters['model_file'])
                with st.spinner(text='In progress'):
                    predictor = pr.ModelPredictor(parameters)
                    print("predictor : ", predictor.acc)
                #st.sidebar.write("Prediction Accuracy : ", (predictor.acc *100), " %")
                #print(predictor.acc)
                st.success('Done')
        elif parameters['mode'] in ['batch']:
            if st.button("Process"):
                print(parameters)
                print(parameters['folder'])
                print(parameters['model_file'])
                with st.spinner(text='In progress'):
                    predictor = pr.ModelPredictor(parameters)
                st.success('Done')

if __name__ == "__main__":
    main(sys.argv[1:])


#streamlit run dashboard.py