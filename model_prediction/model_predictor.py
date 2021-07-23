# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:49:28 2020

@author: Manuel Camargo
"""
import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import configparser as cp
from num2words import num2words as nw

from st_aggrid import AgGrid

from tensorflow.keras.models import load_model

from support_modules.readers import log_reader as lr
from support_modules import support as sup

# ----model_training import----
from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev
from dashboard_module import prediction_dashboard as pdash


class ModelPredictor():
    """
    This is the main class encharged of the model evaluation
    """

    def __init__(self, parms):
        self.output_route = os.path.join('output_files', parms['folder'])
        self.parms = parms
        # load parameters
        self.load_parameters()
        self.model_name, _ = os.path.splitext(parms['model_file'])
        self.model = load_model(os.path.join(self.output_route,
                                             parms['model_file']))

        self.log = self.load_log_test(self.output_route, self.parms)

        self.samples = dict()
        self.predictions = None
        self.run_num = 0

        self.model_def = dict()
        self.read_model_definition(self.parms['model_type'])
        # print("Model Type :", self.model_def)
        self.parms['additional_columns'] = self.model_def['additional_columns']
        self.acc = self.execute_predictive_task()

    def execute_predictive_task(self):
        # create examples for next event and suffix
        if self.parms['activity'] == 'pred_log':
            self.parms['num_cases'] = len(self.log.caseid.unique())
        else:
            sampler = it.SamplesCreator()
            sampler.create(self, self.parms['activity'])

        self.parms['caseid'] = np.array(self.log.caseid)  # adding caseid to the parms

        self.imp = self.parms['variant']  # passes value arg_max and random_choice
        self.run_num = 0
        #prediction call
        for i in range(0, self.parms['rep']):
            self.predict_values()
            self.run_num += 1
        # export predictions
        self.export_predictions()
        # assesment
        evaluator = EvaluateTask()
        if self.parms['activity'] == 'pred_log':
            data = self.append_sources(self.log, self.predictions,
                                       self.parms['one_timestamp'])
            data['caseid'] = data['caseid'].astype(str)
            return evaluator.evaluate(self.parms, data)
        else:
            results_copy = self.predictions.copy()
            self.dashboard_prediction(results_copy, self.parms)
            return evaluator.evaluate(self.parms, self.predictions)

    def predict_values(self):
        # Predict values
        executioner = it.PredictionTasksExecutioner()
        executioner.predict(self, self.parms['activity'], self.parms['mode'], self.parms['next_mode'])

    @staticmethod
    def load_log_test(output_route, parms):
        df_test = lr.LogReader(
            os.path.join(output_route, 'parameters', 'test_log.csv'),
            parms['read_options'])
        if parms['mode'] == 'next':
            df_test = pd.DataFrame(df_test.data)
            df_test = df_test[~df_test.task.isin(['Start', 'End']) & df_test.caseid.isin([parms['nextcaseid']])]
        elif parms['mode'] == 'batch':
            df_test = pd.DataFrame(df_test.data)
            df_test = df_test[~df_test.task.isin(['Start', 'End'])]
        return df_test

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.output_route,
                            'parameters',
                            'model_parameters.json')
        with open(path) as file:
            data = json.load(file)
            if 'activity' in data:
                del data['activity']
            self.parms = {**self.parms, **{k: v for k, v in data.items()}}
            self.parms['dim'] = {k: int(v) for k, v in data['dim'].items()}
            if self.parms['one_timestamp']:
                self.parms['scale_args'] = {
                    k: float(v) for k, v in data['scale_args'].items()}
            else:
                for key in data['scale_args'].keys():
                    self.parms['scale_args'][key] = {
                        k: float(v) for k, v in data['scale_args'][key].items()}
            self.parms['index_ac'] = {int(k): v
                                      for k, v in data['index_ac'].items()}
            self.parms['index_rl'] = {int(k): v
                                      for k, v in data['index_rl'].items()}
            self.parms['index_label'] = {int(k): v
                                         for k, v in data['index_label'].items()}

            file.close()
            self.ac_index = {v: k for k, v in self.parms['index_ac'].items()}
            self.rl_index = {v: k for k, v in self.parms['index_rl'].items()}
            self.label_index = {v: k for k, v in self.parms['index_label'].items()}

    def sampling(self, sampler):
        print("Model Type : ", self.parms['model_type'])
        print("Model Def : ", self.model_def['vectorizer'])
        sampler.register_sampler(self.parms['model_type'],
                                 self.model_def['vectorizer'])
        self.samples = sampler.create_samples(
            self.parms, self.log, self.ac_index,
            self.rl_index, self.label_index, self.model_def['additional_columns'])
        #print("Samples : ", self.samples)
    #
    def predict(self, executioner, mode):
        if mode == 'next':
            results = executioner.predict(self.parms,
                                          self.model,
                                          self.samples,
                                          self.imp,
                                          self.model_def['vectorizer'])
        elif mode == 'batch':
            results = executioner.predict(self.parms,
                                          self.model,
                                          self.samples,
                                          self.imp,
                                          self.model_def['vectorizer'])

        results = pd.DataFrame(results)
        # print("Output of the predictions before :", results)
        # print("Output of the predictions after :", results)
        results['run_num'] = self.run_num
        results['implementation'] = self.imp
        if self.predictions is None:
            self.predictions = results
        else:
            self.predictions = self.predictions.append(results,
                                                       ignore_index=True)

    def export_predictions(self):
        output_folder = os.path.join(self.output_route, 'results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filename = self.model_name + '_' + self.parms['activity'] + '.csv'
        self.predictions.to_csv(os.path.join(output_folder, filename),
                                index=False)

    @staticmethod
    def append_sources(source_log, source_predictions, one_timestamp):
        log = source_log.copy()
        columns = ['caseid', 'task', 'end_timestamp', 'role']
        if not one_timestamp:
            columns += ['start_timestamp']
        log = log[columns]
        log['run_num'] = 0
        log['implementation'] = 'log'
        predictions = source_predictions.copy()
        columns = log.columns
        predictions = predictions[columns]
        return log.append(predictions, ignore_index=True)

    @staticmethod
    # ------This Can be removed as it is not getting used anywhere------
    def scale_feature(log, feature, parms, replace=False):
        """Scales a number given a technique.
        Args:
            log: Event-log to be scaled.
            feature: Feature to be scaled.
            method: Scaling method max, lognorm, normal, per activity.
            replace (optional): replace the original value or keep both.
        Returns:
            Scaleded value between 0 and 1.
        """
        method = parms['norm_method']
        scale_args = parms['scale_args']
        if method == 'lognorm':
            log[feature + '_log'] = np.log1p(log[feature])
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            log[feature + '_norm'] = np.divide(
                np.subtract(log[feature + '_log'], min_value), (max_value - min_value))
            log = log.drop((feature + '_log'), axis=1)
        elif method == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            log[feature + '_norm'] = np.divide(
                np.subtract(log[feature], min_value), (max_value - min_value))
        elif method == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            log[feature + '_norm'] = np.divide(np.subtract(log[feature], mean),
                                               std)
        elif method == 'max':
            max_value = scale_args['max_value']
            log[feature + '_norm'] = (np.divide(log[feature], max_value)
                                      if max_value > 0 else 0)
        elif method is None:
            log[feature + '_norm'] = log[feature]
        else:
            raise ValueError(method)
        if replace:
            log = log.drop(feature, axis=1)
        return log

    def read_model_definition(self, model_type):
        Config = cp.ConfigParser(interpolation=None)
        Config.read('models_spec.ini')
        # File name with extension
        self.model_def['additional_columns'] = sup.reduce_list(
            Config.get(model_type, 'additional_columns'), dtype='str')
        self.model_def['vectorizer'] = Config.get(model_type, 'vectorizer')



    #-------------------------------------------------------------------------------------------------------------------
    #                   Dashboard Code
    # -------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def dashboard_prediction(pred_results_df, parms):
        # Removing 'ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation' from the result
        results_dash = pred_results_df[
            ['ac_prefix', 'rl_prefix', 'label_prefix', 'tm_prefix', 'run_num', 'implementation']].copy()
        results_dash = pred_results_df.drop(
            ['ac_prefix', 'rl_prefix', 'label_prefix', 'tm_prefix', 'run_num', 'implementation'], axis=1)

        # Replacing from Dictionary Values to it's original name
        results_dash['ac_expect'] = results_dash.ac_expect.replace(parms['index_ac'])
        results_dash['rl_expect'] = results_dash.rl_expect.replace(parms['index_rl'])
        results_dash['label_expect'] = results_dash.label_expect.replace(parms['index_label'])

        if parms['mode'] in ['batch']:
            #as the static function is calling static function class has to be mentioned
            ModelPredictor.dashboard_prediction_batch(results_dash, parms)
        elif parms['mode'] in ['next']:
            ModelPredictor.dashboard_prediction_next(results_dash, parms)

    @staticmethod
    def dashboard_prediction_next(results_dash, parms):

        if parms['variant'] in ['multi_pred']:

            #converting the values to it's actual name from parms
            #--For Activity and Role
            ModelPredictor.dashboard_multiprediction_acrl(results_dash, parms)
            #--For Label
            ModelPredictor.dashboard_multiprediction_label(results_dash, parms)

        else:
            #converting the values to it's actual name from parms
            #--For Activity, Role and Label
            results_dash = ModelPredictor.dashboard_maxprediction(results_dash, parms)

        ModelPredictor.dashboard_nextprediction_write(results_dash, parms)


    @staticmethod
    @st.cache(persist=True)
    def dashboard_prediction_batch(results_dash, parms):
        #All the results has to be displayed in Tabular form i.e DataFrame
        if parms['variant'] in ['multi_pred']:
            #converting the values to it's actual name from parms
            #--For Activity and Role
            ModelPredictor.dashboard_multiprediction_acrl(results_dash, parms)
            #--For Label
            ModelPredictor.dashboard_multiprediction_label(results_dash, parms)

            results_dash.drop(['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'label_pred', 'label_prob'], axis=1, inplace=True)

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            _lst = [['caseid', 'ac_expect'] + multipreddict["ac_pred"] + multipreddict["ac_prob"] +
                    ['rl_expect'] + multipreddict["rl_pred"] + multipreddict["rl_prob"] +
                    ['label_expect', 'label_pred1', 'label_pred2', 'label_prob1', 'label_prob2', 'tm_expect', 'tm_pred']]

            results_dash = results_dash[_lst[0]]

        else:
            #converting the values to it's actual name from parms
            #--For Activity, Role and Label
            results_dash = ModelPredictor.dashboard_maxprediction(results_dash, parms)
            results_dash.rename(
                columns={'caseid': 'Case_ID', 'ac_expect': 'AC Expected', 'ac_pred': 'AC Predicted', 'ac_prob': 'AC Confidence',
                         'rl_expect': 'RL Expected', 'rl_pred': 'RL Predicted', 'rl_prob': 'RL Confidence',
                         'label_expect': 'LB Expected', 'label_pred': 'LB Predicted', 'label_prob': 'LB Confidence',
                         "tm_expect": 'TM Expected', 'tm_pred': 'TM Predicted'}, inplace=True)

            # results_dash.columns = pd.MultiIndex.from_tuples(
            #     zip(['', 'Activity', '', '', 'Role', '', '', 'Label', '', '', 'Time', ''],
            #         results_dash.columns))

        #st.table(results_dash)
        #AgGrid(results_dash)

    @staticmethod
    @st.cache(persist=True)
    def dashboard_multiprediction_acrl(results_dash, parms):

        multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

        # --------------------results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        for ix in range(len(results_dash['ac_pred'])):
            for jx in range(len(results_dash['ac_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['ac_pred'][ix].append(parms['index_ac'][results_dash.ac_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['ac_prob'][ix][jx] = (results_dash['ac_prob'][ix][jx] * 100)
            # poping out the values from the list
            ln = int(len(results_dash['ac_pred'][ix]) / 2)
            #st.session_state['_activity_pred'] = results_dash['ac_pred'][ix][:ln]
            del results_dash['ac_pred'][ix][:ln]
            results_dash[multipreddict["ac_pred"]] = pd.DataFrame(results_dash.ac_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[multipreddict["ac_prob"]] = pd.DataFrame(results_dash.ac_prob.tolist(),
                                                                  index=results_dash.index)

        #print("Session State At dashboard_multiprediction_acrl After for loop: ", st.session_state)
        # --------------------results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
        for ix in range(len(results_dash['rl_pred'])):
            for jx in range(len(results_dash['rl_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['rl_pred'][ix].append(parms['index_rl'][results_dash.rl_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['rl_prob'][ix][jx] = (results_dash['rl_prob'][ix][jx] * 100)
            # popping out the values from the list
            ln = int(len(results_dash['rl_pred'][ix]) / 2)
            del results_dash['rl_pred'][ix][:ln]
            results_dash[multipreddict["rl_pred"]] = pd.DataFrame(results_dash.rl_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[multipreddict["rl_prob"]] = pd.DataFrame(results_dash.rl_prob.tolist(),
                                                                  index=results_dash.index)

        return results_dash

    @staticmethod
    @st.cache(persist=True)
    def dashboard_multiprediction_label(results_dash, parms):
        for ix in range(len(results_dash['label_pred'])):
            for jx in range(len(results_dash['label_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['label_pred'][ix].append(parms['index_label'][results_dash.label_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['label_prob'][ix][jx] = (results_dash['label_prob'][ix][jx] * 100)
            # popping out the values from the list
            ln = int(len(results_dash['label_pred'][ix]) / 2)
            del results_dash['label_pred'][ix][:ln]
            results_dash[['label_pred1', 'label_pred2']] = pd.DataFrame(results_dash.label_pred.tolist(),
                                                                        index=results_dash.index)
            results_dash[['label_prob1', 'label_prob2']] = pd.DataFrame(results_dash.label_prob.tolist(),
                                                                        index=results_dash.index)
        return results_dash

    @staticmethod
    @st.cache(persist=True)
    def dashboard_maxprediction(results_dash, parms):
        results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
        results_dash['label_pred'] = results_dash.label_pred.replace(parms['index_label'])
        results_dash['ac_prob'] = (results_dash['ac_prob'] * 100)
        results_dash['rl_prob'] = (results_dash['rl_prob'] * 100)
        results_dash['label_prob'] = (results_dash['label_prob'] * 100)
        return results_dash

    @staticmethod
    def dashboard_nextprediction_write(results_dash, parms):

        if parms['next_mode'] == 'history_with_next':

            ModelPredictor.dashboard_nextprediction_execute_write(results_dash, parms)
        elif parms['next_mode'] == 'next_action':

            ModelPredictor.dashboard_nextprediction_evaluate_write(results_dash, parms)

        elif parms['next_mode'] == 'what_if':

            ModelPredictor.dashboard_nextprediction_whatif_write(results_dash, parms)

        print("------------------------------How Session State Looks Like : ", st.session_state)

    @staticmethod
    def dashboard_nextprediction_execute_write(results_dash, parms):
        st.header('📜 Process Historical Behaviour')
        results_dash_expected = results_dash[['ac_expect', 'rl_expect', 'label_expect', "tm_expect"]]
        results_dash_expected.rename(
            columns={'ac_expect': 'Activity', 'pos_rl_ss': 'Role', 'pos_lb_ss': 'Label', "pos_tm_ss": 'Time'},
            inplace=True)
        st.dataframe(results_dash_expected.iloc[:-1])
        st.markdown("""---""")
        if parms['variant'] in ['multi_pred']:

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            with st.beta_container():
                colstm = st.beta_columns(1)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration of Predictions')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T, use_column_width=True)
            cols = st.beta_columns(parms['multiprednum'])

            for kz in range(parms['multiprednum']):
                    with cols[kz]:
                        ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
            st.markdown("""---""")

        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.beta_columns([2, 2, 2, 1])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('🏷️ Label')
                st.write(results_dash[["label_pred", "label_prob"]].rename(
                    columns={"label_pred": 'Predicted', "label_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            st.markdown("""---""")
        #--Predictions of Predictions
        if st.session_state['multi_pred_ss']['ss_multipredict1']['ac_pred'] != []:
            ModelPredictor.dashboard_nextprediction_execute_multiverse(parms)

    @staticmethod
    def dashboard_nextprediction_execute_multiverse(parms):
        for lk in range(parms['multiprednum']):
            st.header("📜 " + nw(lk + 1, lang="en", to="ordinal_num") + " Prediction " + " Historical Behaviour ")

            _hist_columns = ['pos_ac_ss', 'pos_rl_ss', 'pos_lb_ss']  # selecting the columns
            _hist_predicted_dict = dict(
                [(k, st.session_state['initial_prediction']['ss_initpredict' + str(lk + 1)][k]) for k in
                 _hist_columns])  # constructing new dict fromm sessionstate
            _hist_predicted_dict.update(dict([(k, st.session_state[k]) for k in ['pos_tm_ss']]))  # Apending time
            # --Manuplation to see the Value properly
            _hist_predicted_dict['pos_tm_ss'] = sum(_hist_predicted_dict['pos_tm_ss'], [])  # flattening of time
            _hist_predicted_dict['pos_tm_ss'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in
                                                 _hist_predicted_dict[
                                                     'pos_tm_ss']]  # Normalizing back to original value
            _hist_predicted_dict = {k: _hist_predicted_dict[k][1:] for k in
                                    _hist_predicted_dict}  # removing first item in each key in a dictionary

            _hist_predicted_df = pd.DataFrame.from_dict(_hist_predicted_dict)
            # Replacing from Dictionary Values to it's original name
            _hist_predicted_df['pos_ac_ss'] = _hist_predicted_df.pos_ac_ss.replace(parms['index_ac'])
            _hist_predicted_df['pos_rl_ss'] = _hist_predicted_df.pos_rl_ss.replace(parms['index_rl'])
            _hist_predicted_df['pos_lb_ss'] = _hist_predicted_df.pos_lb_ss.replace(parms['index_label'])
            _hist_predicted_df.rename(
                columns={'pos_ac_ss': 'Activity', 'pos_rl_ss': 'Role', 'pos_lb_ss': 'Label', "pos_tm_ss": 'Time'},
                inplace=True)

            # _hist_predicted_df = _hist_predicted_df.iloc[1:]
            st.dataframe(_hist_predicted_df.iloc[:-1])
            st.markdown("""---""")
            _multi_columns = ['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'label_pred', 'label_prob', 'tm_pred']

            _multi_predicted_dict = dict(
                [(k, st.session_state['multi_pred_ss']['ss_multipredict' + str(lk + 1)][k]) for k in _multi_columns])
            _multi_predicted_dict['tm_pred'] = sum(_multi_predicted_dict['tm_pred'], [])  # flattening of time
            _multi_predicted_dict['tm_pred'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in
                                                _multi_predicted_dict['tm_pred']]  # Normalizing back to original value
            _multi_predicted_df = pd.DataFrame.from_dict(_multi_predicted_dict)
            _multi_predicted_df = ModelPredictor.dashboard_maxprediction(_multi_predicted_df, parms)
            _multi_predicted_df.index = _multi_predicted_df.index + 1 #to match with the index value of the main prediction
            st.subheader("🤔 Max Probability Prediction of " + nw(lk + 1, lang="en", to="ordinal_num") + " Prediction")
            cols1, cols2, cols3, cols4 = st.beta_columns([2, 2, 2, 1])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(_multi_predicted_df[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(_multi_predicted_df[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('🏷️ Label')
                st.write(_multi_predicted_df[["label_pred", "label_prob"]].rename(
                    columns={"label_pred": 'Predicted', "label_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('⌛ Time')
                st.write(
                    _multi_predicted_df[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            st.markdown("""---""")

    @staticmethod
    def dashboard_nextprediction_evaluate_write(results_dash, parms):
        if parms['variant'] in ['multi_pred']:
            cols = st.beta_columns(parms['multiprednum']+1)

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)
            for kz in range(parms['multiprednum']+1):
                if kz <= (parms['multiprednum']-1):
                    with cols[kz]:
                        ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
                elif kz == (parms['multiprednum']) and parms['next_mode']:
                    with cols[kz]:
                        st.header("🧐 Expected ")

                        st.subheader('🏋️ Activity')
                        # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(results_dash[["ac_expect"]].rename(columns={"ac_expect": 'Expected'}, inplace=False))

                        st.markdown("""---""")
                        st.subheader('👨‍💻 Role')
                        # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                        st.write(results_dash[["rl_expect"]].rename(columns={"rl_expect": 'Expected'}, inplace=False))

                        st.markdown("""---""")
                        st.subheader('🏷️ Label')
                        st.write(results_dash[["label_expect"]].rename(columns={"label_expect": 'Expected'}, inplace=False))
            st.markdown("""---""")
            with st.beta_container():
                colstm = st.beta_columns(2)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).T,
                             use_column_width=True)
                with colstm[1]:
                    st.subheader('⌚ Expected Time Duration')
                    st.write(
                        results_dash[["tm_expect"]].rename(columns={"tm_expect": 'Predicted'}, inplace=False).T,
                        use_column_width=True)

        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.beta_columns([2, 2, 1, 2])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["ac_pred", "ac_prob"]].rename(columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False))
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["rl_pred", "rl_prob"]].rename(columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False))
            with cols3:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
            with cols4:
                st.subheader('🏷️ Label')
                st.write(results_dash[["label_pred", "label_prob"]].rename(columns={"label_pred": 'Predicted', "label_prob": 'Confidence'}, inplace=False))


    @staticmethod
    def dashboard_nextprediction_whatif_write(results_dash, parms):
        st.header('📜 Process Historical Behaviour')
        _hist_columns = ['hist_ac_prefix', 'hist_rl_prefix', 'hist_lb_prefix', 'hist_tm_prefix', 'hist_pred_prefix']  # selecting the columns
        _hist_choice_dict = dict([(k, st.session_state['history_of_choice'][k]) for k in _hist_columns])  # constructing new dict fromm sessionstate
        _hist_choice_dict['hist_tm_prefix'] = sum(_hist_choice_dict['hist_tm_prefix'], []) # flattening of time
        _hist_choice_dict['hist_tm_prefix'] = [ModelPredictor.rescale(x, parms, parms['scale_args']) for x in _hist_choice_dict['hist_tm_prefix']]  # Normalizing back to original value
        _hist_choice_df = pd.DataFrame.from_dict(_hist_choice_dict)
        # Replacing from Dictionary Values to it's original name
        _hist_choice_df ['hist_ac_prefix'] = _hist_choice_df.hist_ac_prefix.replace(parms['index_ac'])
        _hist_choice_df['hist_rl_prefix'] = _hist_choice_df.hist_rl_prefix.replace(parms['index_rl'])
        _hist_choice_df['hist_lb_prefix'] = _hist_choice_df.hist_lb_prefix.replace(parms['index_label'])

        _hist_choice_df.rename(columns={'hist_ac_prefix': 'Activity', 'hist_rl_prefix': 'Role', 'hist_lb_prefix': 'Label', 'hist_tm_prefix': 'Time', 'hist_pred_prefix':'Choice'}, inplace=True)

        st.dataframe(_hist_choice_df)
        st.markdown("""---""")
        if parms['variant'] in ['multi_pred']:

            multipreddict = ModelPredictor.dashboard_multiprediction_columns(parms)

            with st.beta_container():
                colstm = st.beta_columns(1)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration of Predictions')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T, use_column_width=True)
            cols = st.beta_columns(parms['multiprednum'])

            for kz in range(parms['multiprednum']):
                    with cols[kz]:
                        ModelPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
            st.markdown("""---""")

        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.beta_columns([2, 2, 2, 1])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["ac_pred", "ac_prob"]].rename(
                    columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[["rl_pred", "rl_prob"]].rename(
                    columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols3:
                st.subheader('🏷️ Label')
                st.write(results_dash[["label_pred", "label_prob"]].rename(
                    columns={"label_pred": 'Predicted', "label_prob": 'Confidence'}, inplace=False).iloc[-1:])
            with cols4:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False).iloc[-1:])
            st.markdown("""---""")

        #------Supporting Functions
    @staticmethod
    def dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz):
        if kz <= parms['multiprednum']:
            if parms['next_mode'] == 'next_action': #Evaluation Mode
                st.subheader("🤔", nw(kz + 1, lang="en", to="ordinal_num") + "Prediction")
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]].rename(
                    columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                    inplace=False))
                st.markdown("""---""")
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["rl_pred"][kz]] + [multipreddict["rl_prob"][kz]]].rename(
                    columns={multipreddict["rl_pred"][kz]: 'Predicted', multipreddict["rl_prob"][kz]: 'Confidence'},
                    inplace=False))
                st.markdown("""---""")
                # st.subheader('⌛ Time')
                # st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
                st.subheader('🏷️ Label')
                if kz <= 1:
                    st.write(results_dash[["label_pred" + str(kz + 1)] + ["label_prob" + str(kz + 1)]].rename(
                        columns={"label_pred" + str(kz + 1): 'Predicted', "label_prob" + str(kz + 1): 'Confidence'},
                        inplace=False))
                elif kz > 1:
                    st.write("None")
            elif parms['next_mode'] in ['history_with_next', 'what_if']: #Execution Mode
                # st.header("🤔 Prediction " + str(kz + 1))
                st.header("🤔 " + nw(kz + 1, lang="en", to="ordinal_num") + " Prediction")
                # with st.beta_expander('ℹ️'):
                #     st.info("Predicted Events")
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["ac_pred"][kz]] + [multipreddict["ac_prob"][kz]]].rename(
                    columns={multipreddict["ac_pred"][kz]: 'Predicted', multipreddict["ac_prob"][kz]: 'Confidence'},
                    inplace=False).iloc[-1:])
                st.markdown("""---""")
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(results_dash[[multipreddict["rl_pred"][kz]] + [multipreddict["rl_prob"][kz]]].rename(
                    columns={multipreddict["rl_pred"][kz]: 'Predicted', multipreddict["rl_prob"][kz]: 'Confidence'},
                    inplace=False).iloc[-1:])
                st.markdown("""---""")
                # st.subheader('⌛ Time')
                # st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
                st.subheader('🏷️ Label')
                if kz <= 1:
                    st.write(results_dash[["label_pred" + str(kz + 1)] + ["label_prob" + str(kz + 1)]].rename(
                        columns={"label_pred" + str(kz + 1): 'Predicted', "label_prob" + str(kz + 1): 'Confidence'},
                        inplace=False).iloc[-1:])
                elif kz > 1:
                    st.write("None")

    # ------Supporting Functions
    @staticmethod
    @st.cache(persist=True)
    def dashboard_multiprediction_columns(parms):
        # Initialize list for multi pred
        ac_pred_lst = []
        ac_prob_lst = []
        rl_pred_lst = []
        rl_prob_lst = []
        multipreddict = {}
        for zx in range(parms['multiprednum']):
            zx += 1
            ac_pred_lst.append("ac_pred" + str(zx))
            ac_prob_lst.append("ac_prob" + str(zx))
            rl_pred_lst.append("rl_pred" + str(zx))
            rl_prob_lst.append("rl_prob" + str(zx))
        multipreddict["ac_pred"] = ac_pred_lst
        multipreddict["ac_prob"] = ac_prob_lst
        multipreddict["rl_pred"] = rl_pred_lst
        multipreddict["rl_prob"] = rl_prob_lst
        return multipreddict

    # ------Supporting Functions
    @staticmethod
    @st.cache(persist=True)
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif parms['norm_method'] == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif parms['norm_method'] == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            value = (value * std) + mean
        elif parms['norm_method'] == 'max':
            max_value = scale_args['max_value']
            value = np.rint(value * max_value)
        elif parms['norm_method'] is None:
            value = value
        else:
            raise ValueError(parms['norm_method'])
        return value

class EvaluateTask():

    def evaluate(self, parms, data):
        sampler = self._get_evaluator(parms['activity'])
        if parms['variant'] in ['multi_pred']:
            data['ac_expect'] = data.ac_expect.replace(parms['index_ac'])
            data['rl_expect'] = data.rl_expect.replace(parms['index_rl'])
            data['label_expect'] = data.label_expect.replace(parms['index_label'])
        # print("Evaluate Data:", data)
        return sampler(data, parms)

    def _get_evaluator(self, activity):
        if activity == 'predict_next':
            return self._evaluate_predict_next
        elif activity == 'pred_sfx':
            return self._evaluate_pred_sfx
        elif activity == 'pred_log':
            return self._evaluate_predict_log
        else:
            raise ValueError(activity)

    def _evaluate_predict_next(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'])

        ac_sim = evaluator.measure('accuracy', data, 'ac')
        rl_sim = evaluator.measure('accuracy', data, 'rl')
        label_sim = evaluator.measure('accuracy', data, 'label')

        mean_ac = ac_sim.accuracy.mean()
        mean_rl = rl_sim.accuracy.mean()
        mean_label = label_sim.accuracy.mean()
        exp_desc = pd.DataFrame([exp_desc])

        st.sidebar.write("Activity Prediction Accuracy : ", round((mean_ac * 100), 2), " %")
        st.sidebar.write("Role Prediction Accuracy : ", round((mean_rl * 100), 2), " %")
        st.sidebar.write("Label Prediction Accuracy : ", round((mean_label * 100), 2), " %")

        exp_desc = pd.concat([exp_desc] * len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        label_sim = pd.concat([label_sim, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        self.save_results(label_sim, 'label', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_next', data, 'tm')
            tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_next', data, 'dur')
            wait_mae = evaluator.measure('mae_next', data, 'wait')
            dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(dur_mae, 'dur', parms)
            self.save_results(wait_mae, 'wait', parms)
        st.sidebar.write("Time MAE : ", round(tm_mae[0]['mae'], 2))
        return mean_ac

    def _evaluate_pred_sfx(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'])
        ac_sim = evaluator.measure('similarity', data, 'ac')
        rl_sim = evaluator.measure('similarity', data, 'rl')
        label_sim = evaluator.measure('similarity', data, 'label')
        mean_sim = ac_sim['mean'].mean()
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc] * len(ac_sim), ignore_index=True)
        ac_sim = pd.concat([ac_sim, exp_desc], axis=1).to_dict('records')
        rl_sim = pd.concat([rl_sim, exp_desc], axis=1).to_dict('records')
        label_sim = pd.concat([label_sim, exp_desc], axis=1).to_dict('records')
        self.save_results(ac_sim, 'ac', parms)
        self.save_results(rl_sim, 'rl', parms)
        self.save_results(label_sim, 'label', parms)
        if parms['one_timestamp']:
            tm_mae = evaluator.measure('mae_suffix', data, 'tm')
            tm_mae = pd.concat([tm_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(tm_mae, 'tm', parms)
        else:
            dur_mae = evaluator.measure('mae_suffix', data, 'dur')
            wait_mae = evaluator.measure('mae_suffix', data, 'wait')
            dur_mae = pd.concat([dur_mae, exp_desc], axis=1).to_dict('records')
            wait_mae = pd.concat([wait_mae, exp_desc], axis=1).to_dict('records')
            self.save_results(dur_mae, 'dur', parms)
            self.save_results(wait_mae, 'wait', parms)
        return mean_sim

    def _evaluate_predict_log(self, data, parms):
        exp_desc = self.clean_parameters(parms.copy())
        evaluator = ev.Evaluator(parms['one_timestamp'], parms['variant'])
        dl = evaluator.measure('dl', data)
        els = evaluator.measure('els', data)
        mean_els = els.els.mean()
        mae = evaluator.measure('mae_log', data)
        exp_desc = pd.DataFrame([exp_desc])
        exp_desc = pd.concat([exp_desc] * len(dl), ignore_index=True)
        # exp_desc = pd.concat([exp_desc]*len(els), ignore_index=True)
        dl = pd.concat([dl, exp_desc], axis=1).to_dict('records')
        els = pd.concat([els, exp_desc], axis=1).to_dict('records')
        mae = pd.concat([mae, exp_desc], axis=1).to_dict('records')
        self.save_results(dl, 'dl', parms)
        self.save_results(els, 'els', parms)
        self.save_results(mae, 'mae', parms)
        return mean_els

    @staticmethod
    def clean_parameters(parms):
        exp_desc = parms.copy()
        exp_desc.pop('activity', None)
        exp_desc.pop('read_options', None)
        exp_desc.pop('column_names', None)
        exp_desc.pop('one_timestamp', None)
        exp_desc.pop('reorder', None)
        exp_desc.pop('index_ac', None)
        exp_desc.pop('index_rl', None)
        exp_desc.pop('index_label', None)
        exp_desc.pop('dim', None)
        exp_desc.pop('max_dur', None)
        exp_desc.pop('variants', None)
        exp_desc.pop('is_single_exec', None)
        return exp_desc

    @staticmethod
    def save_results(measurements, feature, parms):
        if measurements:
            if parms['is_single_exec']:
                output_route = os.path.join('output_files',
                                            parms['folder'],
                                            'results')
                model_name, _ = os.path.splitext(parms['model_file'])
                sup.create_csv_file_header(
                    measurements,
                    os.path.join(
                        output_route,
                        model_name + '_' + feature + '_' + parms['activity'] + '.csv'))
                print("output_route : ", output_route)
            else:
                if os.path.exists(os.path.join(
                        'output_files', feature + '_' + parms['activity'] + '.csv')):
                    sup.create_csv_file(
                        measurements,
                        os.path.join('output_files',
                                     feature + '_' + parms['activity'] + '.csv'),
                        mode='a')
                else:
                    sup.create_csv_file_header(
                        measurements,
                        os.path.join('output_files',
                                     feature + '_' + parms['activity'] + '.csv'))
