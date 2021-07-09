import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys
import json
import getopt
import streamlit as st
#import SessionState
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
config.inter_op_parallelism_threads = 4
config.intra_op_parallelism_threads = 4
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

    if not argv:
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next
        #parameters['activity'] = classifier
        parameters['activity'] = 'predict_next' # Change Here
        # Event-log reading parameters
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}

        # Folder picker button
        if parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:

            #--Hard Coded Value
            parameters['is_single_exec'] = True  # single or batch execution
            parameters['rep'] = 1

            _folder_name  = st.sidebar.text_input('Folder Name')
            parameters['folder'] = _folder_name

            if parameters['folder'] != "":
                #--Selecting Model
                _model_directory = os.path.join('output_files', _folder_name)
                _model_name = []
                for file in os.listdir(_model_directory):
                    # check the files which are end with specific extention
                    if file.endswith(".h5"):
                        _model_name.append(file)
                parameters['model_file'] = _model_name[-1]

                #--Selecting Mode
                st.sidebar.subheader("Choose Mode of Prediction")
                _mode_sel = st.sidebar.radio('Mode', ['Batch Processing', 'Single Event Processing'])

                if _mode_sel == 'Batch Processing':
                    _mode_sel = 'batch'
                elif _mode_sel == 'Single Event Processing':
                    _mode_sel = 'next'

                parameters['mode'] = _mode_sel

                #--Type of Variant
                # if parameters['mode'] == 'batch':
                #     parameters['multiprednum'] = 2 #Change here for batch mode Prediction
                #     variant_opt = st.sidebar.selectbox("Variant", ('Max Probability Prediction', 'Multiple Prediction', 'Random Prediction'),
                #                                        key='variant_opt')
                #     if variant_opt == 'Max Probability Prediction':
                #         variant_opt = 'arg_max'
                #     elif variant_opt == 'Multiple Prediction':
                #         variant_opt = 'multi_pred'
                #     elif variant_opt == 'Random Prediction':
                #         variant_opt = 'random_choice'
                #     parameters['variant'] = variant_opt  # random_choice, arg_max for variants and repetitions to be tested
            #--Selecting the model file using folder picker
            # if parameters['folder'] != "":
            # ml_model = st.sidebar.file_uploader("Upload ML Model", type=['h5'])
            #
            # print("Model Name :", ml_model)
            # if ml_model is not None:
            #     parameters['model_file'] = ml_model.name
            # # Set up tkinter
            # root = tk.Tk()
            # root.withdraw()
            # # Make folder picker dialog appear on top of other windows
            # root.wm_attributes('-topmost', 1)

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

    def _list2dictConvert(_a):
        _it = iter(_a)
        _res_dct = dict(zip(_it, _it))
        return _res_dct

    #   Saves the result in the URL in the Next mode
    app_state = st.experimental_get_query_params()
    if "my_saved_result" in app_state:
        saved_result = app_state["my_saved_result"][0]
        nxt_button_idx = int(saved_result)
        #st.write("Here is your result", saved_result)
    else:
        st.write("No result to display, compute a value first.")
        nxt_button_idx = 0

    @st.cache(persist=True)
    def read_next_testlog():
        input_file = os.path.join('output_files', parameters['folder'], 'parameters', 'test_log.csv')
        filter_log = pd.read_csv(input_file, dtype={'user': str})
        # Standard Code based on log_reader
        filter_log = filter_log.rename(columns=column_names)
        filter_log = filter_log.astype({'caseid': object})
        filter_log = (filter_log[(filter_log.task != 'Start') & (filter_log.task != 'End')].reset_index(drop=True))
        filter_log_columns = filter_log.columns
        return filter_log, filter_log_columns

    def next_columns(filter_log, display_columns):
        # Dashboard selection of Case ID
        filter_caseid = st.selectbox("Select Case ID", filter_log["caseid"].unique())
        filter_caseid_attr_df = filter_log.loc[filter_log["caseid"].isin([filter_caseid])]
        filter_attr_display = filter_caseid_attr_df[display_columns]
        return filter_attr_display, filter_caseid, filter_caseid_attr_df


    def num_predictions(_df):
        # _df = pd.DataFrame(_listoflist)
        # _df.columns = ['activity', 'role', 'time']
        # deciding max number of prediction; Choosing the lowest count considering role and activity as pair
        #print("Possible number of predictions :", _df['task'].nunique(), _df['role'].nunique())
        if _df['task'].nunique() > _df['role'].nunique():
            _dfmax = _df['role'].nunique()
        elif _df['role'].nunique() > _df['task'].nunique():
            _dfmax = _df['task'].nunique()
        else:
            _dfmax = _df['task'].nunique()
        #st.sidebar.subheader("Choose the number of Prediction")
        slider = st.sidebar.slider(
            label='Variant : Max Probability = 1 ; Multiple Prediction > 1 ', min_value=1,
            max_value=_dfmax, key='my_number_prediction_slider')
        return slider

    #print("all_lines:", all_lines)
    if parameters['folder']  != "":
        if parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
            if parameters['mode'] in ['next']:
                next_option = st.sidebar.selectbox("Type of Single Event Processing", ('history_with_next', 'next_action'), key='next_dropdown_opt')

                # Read the Test Log
                filter_log, filter_log_columns = read_next_testlog()
                essential_columns = ['task', 'role', 'end_timestamp']
                extra_columns = ['caseid', 'label', 'dur', 'acc_cycle', 'daytime', 'dur_norm', 'ac_index', 'rl_index', 'label_index', 'wait_norm']
                display_columns = list(set(filter_log_columns) - set(essential_columns+extra_columns ))
                filter_attr_display, filter_caseid, filter_caseid_attr_df = next_columns(filter_log, display_columns)
                parameters['nextcaseid'] = filter_caseid

                print("Display Attributes :", filter_attr_display.iloc[[2]])
                st.text('State of the Process')
                display_slot1 = st.empty()

                #--Defining Number of Multiprediction
                #if parameters['variant'] in ['multi_pred']:
                parameters['multiprednum'] = num_predictions(filter_caseid_attr_df)

                print("Multi pred :", type(parameters['multiprednum']), parameters['multiprednum'])

                if parameters['multiprednum'] == 1:
                    variant_opt = 'arg_max'
                elif parameters['multiprednum'] > 1:
                    variant_opt = 'multi_pred'

                parameters['variant'] = variant_opt

                filter_caseid_attr_df = filter_caseid_attr_df[essential_columns].values.tolist()

                if next_option == 'next_action':
                    # Filtering values based on the chosen Case ID

                    filter_caseid_attr_list = st.select_slider("Choose [Activity, User, Time]", options=filter_caseid_attr_df, key="caseid_attr_slider")
                    #st.session_state.caseid_attr_slider = filter_caseid_attr_list

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
                    parameters['nextcaseid_attr'] = filter_caseid_attr_dict
                    parameters['next_mode'] = next_option
                    print("Parameters : ", parameters)
                    if (_idx+1) < len(filter_caseid_attr_df): #Index starts from 0 so added 1 to equate with length value
                        _filterdf = filter_attr_display.iloc[[_idx]]
                        _filterdf.index = [""] * len(_filterdf)
                        display_slot1.dataframe(_filterdf)
                        if st.sidebar.button("Process", key='next_process'):
                            print("Under Next Action")
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
                    #filter_caseid = st.selectbox("Select Case ID", filter_log["caseid"].unique())
                    st.experimental_set_query_params(my_saved_caseid=filter_caseid)

                    # Filtering values based on the chosen Case ID
                    #filter_caseid_attr_df = filter_log.loc[filter_log["caseid"].isin([filter_caseid])]
                    #filter_caseid_attr_df = filter_caseid_attr_df[essential_columns].values.tolist()

                    parameters['next_mode'] = next_option
                    #st.write("Length of DF is :", len(filter_caseid_attr_df))
                    #nxt_button_idx += 1
                    next_button = st.sidebar.button("Next Action", key='next_process_action')
                    #if (nxt_button_idx) < len(filter_caseid_attr_df):  # Index starts from 0 so added 1 to equate with length value
                    _filterdf = filter_attr_display.iloc[[nxt_button_idx]]
                    _filterdf.index = [""] * len(_filterdf)
                    display_slot1.dataframe(_filterdf)
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
                parameters['multiprednum'] = 2  # Change here for batch mode Prediction
                variant_opt = st.sidebar.selectbox("Variant", (
                'Max Probability', 'Multiple Prediction', 'Random Prediction'),
                                                   key='variant_opt')
                if variant_opt == 'Max Probability':
                    variant_opt = 'arg_max'
                elif variant_opt == 'Multiple Prediction':
                    variant_opt = 'multi_pred'
                elif variant_opt == 'Random Prediction':
                    variant_opt = 'random_choice'
                parameters['variant'] = variant_opt  # random_choice, arg_max for variants and repetitions to be tested
                parameters['next_mode'] = ''
                if st.sidebar.button("Process", key='batch_process'):
                    print(parameters)
                    print(parameters['folder'])
                    print(parameters['model_file'])
                    with st.spinner(text='In progress'):
                        predictor = pr.ModelPredictor(parameters)
                    st.success('Done')

if __name__ == "__main__":
    main(sys.argv[1:])


#streamlit run dashboard.py