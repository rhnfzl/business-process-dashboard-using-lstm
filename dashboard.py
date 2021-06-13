import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys
import json
import getopt
import streamlit as st
import SessionState
import tkinter as tk
from tkinter import filedialog

import pandas as pd
import numpy as np
import configparser as cp

from model_prediction import model_predictor as pr

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


def main(argv, filter_parms=None, filter_parameter=None):

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

    st.sidebar.subheader("Choose Mode of Prediction")
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


    def next_feature_test_log(input_file):
        filter_log = pd.read_csv(input_file, dtype={'user': str})
        # Standard Code based on log_reader
        filter_log = filter_log.rename(columns=column_names)
        filter_log = filter_log.astype({'caseid': object})
        filter_log = (filter_log[(filter_log.task != 'Start') & (filter_log.task != 'End')].reset_index(drop=True))
        return filter_log

    def _list2dictConvert(_a):
        _it = iter(_a)
        _res_dct = dict(zip(_it, _it))
        return _res_dct

    #   Execution
    app_state = st.experimental_get_query_params()
    if "my_saved_result" in app_state:
        saved_result = app_state["my_saved_result"][0]
        nxt_button_idx = int(saved_result)
        #st.write("Here is your result", saved_result)
    else:
        st.write("No result to display, compute a value first.")
        nxt_button_idx = 0

    #print("all_lines:", all_lines)
    if parameters['folder']  != "":
        #if parameters['model_file'] != "":
        if parameters['model_file'] is not None:
            if parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
                if parameters['mode'] in ['next']:
                    next_option = st.selectbox("Variant", ('history_with_next', 'next_action'), key='next_dropdown_opt')
                    # Read the Test Log
                    input_file = os.path.join('output_files', parameters['folder'], 'parameters', 'test_log.csv')

                    filter_log = next_feature_test_log(input_file)
                    if next_option == 'next_action':
                        #Dashboard selection of Case ID
                        filter_caseid = st.selectbox("Select Case ID", filter_log["caseid"].unique())

                        # Filtering values based on the chosen Case ID
                        filter_caseid_attr_df = filter_log.loc[filter_log["caseid"].isin([filter_caseid])]
                        filter_caseid_attr_df = filter_caseid_attr_df[['task', 'role', 'end_timestamp']].values.tolist()
                        filter_caseid_attr_list = st.select_slider("Choose [Activity, User, Time]", options=filter_caseid_attr_df, key="caseid_attr_slider")
                        _idx = filter_caseid_attr_df.index(filter_caseid_attr_list)
                        filter_caseid_attr_list.append(_idx)

                        #Selected suffix key, Converting list to dictionary
                        filter_key_attr = ["filter_acitivity", "filter_role", "filter_time", "filter_index"]
                        filter_key_pos = [0, 1, 2, 3]
                        assert (len(filter_key_attr) == len(filter_key_pos))
                        _acc_val = 0
                        for i in range(len(filter_key_attr)):
                            filter_caseid_attr_list.insert(filter_key_pos[i] + _acc_val, filter_key_attr[i])
                            _acc_val += 1
                        filter_caseid_attr_dict = _list2dictConvert(filter_caseid_attr_list)
                        print("Value of Slider :", filter_caseid_attr_dict)

                        #Passing the respective paramter to Parameters
                        parameters['nextcaseid'] = filter_caseid
                        parameters['nextcaseid_attr'] = filter_caseid_attr_dict
                        parameters['next_mode'] = next_option
                        print("Parameters : ", parameters)
                        if (_idx+1) < len(filter_caseid_attr_df): #Index starts from 0 so added 1 to equate with length value
                            if st.button("Process", key='next_process'):
                                print(parameters)
                                print(parameters['folder'])
                                print(parameters['model_file'])
                                with st.spinner(text='In progress'):
                                    predictor = pr.ModelPredictor(parameters)
                                    print("predictor : ", predictor.acc)
                                st.success('Done')
                        else:
                            st.error('Reselect the Suffix to a lower Value')
                    elif next_option == 'history_with_next':
                        # Dashboard selection of Case ID
                        filter_caseid = st.selectbox("Select Case ID", filter_log["caseid"].unique())
                        st.experimental_set_query_params(my_saved_caseid=filter_caseid)

                        # Filtering values based on the chosen Case ID
                        filter_caseid_attr_df = filter_log.loc[filter_log["caseid"].isin([filter_caseid])]
                        filter_caseid_attr_df = filter_caseid_attr_df[['task', 'role', 'end_timestamp']].values.tolist()

                        parameters['nextcaseid'] = filter_caseid
                        parameters['next_mode'] = next_option
                        #st.write("Length of DF is :", len(filter_caseid_attr_df))
                        #nxt_button_idx += 1
                        next_button = st.button("Next Action", key='next_process_action')
                        #if (nxt_button_idx) < len(filter_caseid_attr_df):  # Index starts from 0 so added 1 to equate with length value
                        if (next_button) and ((nxt_button_idx) < len(filter_caseid_attr_df)):
                            nxt_button_idx += 1
                            st.experimental_set_query_params(my_saved_result=nxt_button_idx, my_saved_caseid=filter_caseid)  # Save value

                            filter_caseid_attr_list = [nxt_button_idx - 1]

                            filter_key_attr = ["filter_index"]
                            filter_key_pos = [0]
                            assert (len(filter_key_attr) == len(filter_key_pos))
                            _acc_val = 0
                            for i in range(len(filter_key_attr)):
                                filter_caseid_attr_list.insert(filter_key_pos[i] + _acc_val, filter_key_attr[i])
                                _acc_val += 1
                            filter_caseid_attr_dict = _list2dictConvert(filter_caseid_attr_list)

                            parameters['nextcaseid_attr'] = filter_caseid_attr_dict
                            #st.write(parameters['nextcaseid_attr'])

                            print("Parameters : ", parameters)
                            print(parameters)
                            print(parameters['folder'])
                            print(parameters['model_file'])
                            with st.spinner(text='In progress'):
                                predictor = pr.ModelPredictor(parameters)
                                print("predictor : ", predictor.acc)
                            st.success('Done')
                            if (nxt_button_idx) >= len(filter_caseid_attr_df):
                                #next_button.enabled = False
                                st.experimental_set_query_params(my_saved_result=0)
                                st.error('End of Current Case Id, Select the Next Case ID')
                        elif ((nxt_button_idx) >= len(filter_caseid_attr_df)):
                            st.experimental_set_query_params(my_saved_result=0)  # reset value
                            st.error('End of Current Case Id, Select the Next Case ID')

                elif parameters['mode'] in ['batch']:
                    parameters['next_mode'] = ''
                    if st.button("Process", key='batch_process'):
                        print(parameters)
                        print(parameters['folder'])
                        print(parameters['model_file'])
                        with st.spinner(text='In progress'):
                            predictor = pr.ModelPredictor(parameters)
                        st.success('Done')

if __name__ == "__main__":
    main(sys.argv[1:])


#streamlit run dashboard.py