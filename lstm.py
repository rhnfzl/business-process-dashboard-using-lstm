# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""

#---Workaround for Not creating XLA devices, tf_xla_enable_xla_devices not set
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#---

import sys
import getopt

from model_prediction import model_predictor as pr
from model_training import model_trainer as tr


#---Workaround for "tensorflow.python.framework.errors_impl.UnknownError: Fail to find the dnn implementation."
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
session = InteractiveSession(config=config)
#-----

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
    """Main aplication method"""
    parameters = dict()
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parameters['one_timestamp'] = True  # Only one timestamp in the log
    # Similarity btw the resources profile execution (Song e.t. all)
    parameters['rp_sim'] = 0.85
    parameters['batch_size'] = 32 # Usually 32/64/128/256
    parameters['epochs'] = 2
    # Parameters setting manual fixed or catched by console
    if not argv:
        # Type of LSTM task -> training, pred_log
        # pred_sfx, predict_next
        parameters['activity'] = 'training' #Change Here
        # Event-log reading parameters
        parameters['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': parameters['one_timestamp'],
            'ns_include': True}
        # General training parameters
        if parameters['activity'] in ['training']:
            # Event-log parameters
            parameters['file_name'] = 'Helpdesk.xes' #Change Here
            # Specific model training parameters
            if parameters['activity'] == 'training':
                parameters['imp'] = 2  # keras lstm implementation 1 cpu,2 gpu
                parameters['lstm_act'] = 'relu'  # optimization function Keras, None in v1
                parameters['dense_act'] = None  # optimization function Keras
                parameters['optim'] = 'Adam'  # optimization function Keras, Nadam in v1
                parameters['norm_method'] = 'lognorm'  # max, lognorm
                # Model types --> shared_cat, specialized, concatenated, 
                # shared_cat_gru, specialized_gru, concatenated_gru
                parameters['model_type'] = 'shared_cat'
                parameters['n_size'] = 5  # n-gram size
                parameters['l_size'] = 100  # LSTM layer sizes
                # Generation parameters
        elif parameters['activity'] in ['pred_log', 'pred_sfx', 'predict_next']:
            parameters['folder'] = '20210325_D1BCD634_A2D9_4726_AD88_ADD2BC7D403E'
            parameters['model_file'] = 'model_shared_cat_01-1.82.h5'
            parameters['is_single_exec'] = False  # single or batch execution
            # variants and repetitions to be tested random_choice, arg_max
            parameters['variant'] = 'random_choice'
            parameters['rep'] = 1
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
    if parameters['activity'] == 'training':
        print(parameters)
        trainer = tr.ModelTrainer(parameters)
        print(trainer.output, trainer.model, sep=' ')
    elif parameters['activity'] in ['predict_next', 'pred_sfx', 'pred_log']:
        print(parameters)
        print(parameters['folder'])
        print(parameters['model_file'])
        predictor = pr.ModelPredictor(parameters)
        print(predictor.acc)
if __name__ == "__main__":
    main(sys.argv[1:])