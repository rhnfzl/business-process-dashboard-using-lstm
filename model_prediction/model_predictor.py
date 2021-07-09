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

from tensorflow.keras.models import load_model

from support_modules.readers import log_reader as lr
from support_modules import support as sup

# ----model_training import----
from model_prediction import interfaces as it
from model_prediction.analyzers import sim_evaluator as ev


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
        # self.parms['caseid'] = np.array(self.log.caseid.unique()) #adding caseid to the parms
        # print("parms :", self.parms)
        # print("Samples : ", self.samples)
        # print("Before attributes :",self.parms['nextcaseid_attr'], len(self.parms['nextcaseid_attr']))
        # print(self.parms['index_ac'])
        # print(self.parms['index_rl'])

        # for _key, _value in self.parms['index_ac'].items():
        #     if _value == self.parms['nextcaseid_attr']["filter_acitivity"]:
        #         self.parms['nextcaseid_attr']["filter_acitivity"] = _key
        #
        # for _key, _value in self.parms['index_rl'].items():
        #     if _value == self.parms['nextcaseid_attr']["filter_role"]:
        #         self.parms['nextcaseid_attr']["filter_role"] = _key

        # for key, value in self.parms['index_ac'].items():
        #     if value in self.parms['nextcaseid_attr']:
        #         index = self.parms['nextcaseid_attr'].index(value)
        #         if index == 0:
        #             self.parms['nextcaseid_attr'][index] = key
        # for key, value in self.parms['index_rl'].items():
        #     if value in self.parms['nextcaseid_attr']:
        #         index = self.parms['nextcaseid_attr'].index(value)
        #         if index == 1:
        #             self.parms['nextcaseid_attr'][index] = key
        # print("After attributes :", self.parms['nextcaseid_attr'], len(self.parms['nextcaseid_attr']))
        # results_dash['ac_expect'] = results_dash.ac_expect.replace(parms['index_ac'])
        # results_dash['rl_expect'] = results_dash.rl_expect.replace(parms['index_rl'])
        # predict
        self.imp = self.parms['variant']  # passes value arg_max and random_choice
        self.run_num = 0
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
        executioner.predict(self, self.parms['activity'])

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
        sampler.register_sampler(self.parms['model_type'],
                                 self.model_def['vectorizer'])
        self.samples = sampler.create_samples(
            self.parms, self.log, self.ac_index,
            self.rl_index, self.label_index, self.model_def['additional_columns'])

    def predict(self, executioner):
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
            st.write(results_dash)




    #
    # if parms['variant'] in ['multi_pred']:
    #     # --------------------results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
    #     for ix in range(len(results_dash['ac_pred'])):
    #         for jx in range(len(results_dash['ac_pred'][ix])):
    #             # replacing the value from the parms dictionary
    #             results_dash['ac_pred'][ix].append(parms['index_ac'][results_dash.ac_pred[ix][jx]])
    #             # Converting probability into percentage
    #             results_dash['ac_prob'][ix][jx] = (results_dash['ac_prob'][ix][jx] * 100)
    #         # ppping out the values from the list
    #         ln = int(len(results_dash['ac_pred'][ix]) / 2)
    #         del results_dash['ac_pred'][ix][:ln]
    #         results_dash[['ac_pred1', 'ac_pred2', 'ac_pred3']] = pd.DataFrame(results_dash.ac_pred.tolist(),
    #                                                                           index=results_dash.index)
    #         results_dash[['ac_prob1', 'ac_prob2', 'ac_prob3'] = pd.DataFrame(results_dash.ac_prob.tolist(),
    #                                                                           index=results_dash.index)
    #
    #     # --------------------results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
    #     for ix in range(len(results_dash['rl_pred'])):
    #         for jx in range(len(results_dash['rl_pred'][ix])):
    #             # replacing the value from the parms dictionary
    #             results_dash['rl_pred'][ix].append(parms['index_rl'][results_dash.rl_pred[ix][jx]])
    #             # Converting probability into percentage
    #             results_dash['rl_prob'][ix][jx] = (results_dash['rl_prob'][ix][jx] * 100)
    #         # popping out the values from the list
    #         ln = int(len(results_dash['rl_pred'][ix]) / 2)
    #         del results_dash['rl_pred'][ix][:ln]
    #         results_dash[['rl_pred1', 'rl_pred2', 'rl_pred3']] = pd.DataFrame(results_dash.rl_pred.tolist(),
    #                                                                           index=results_dash.index)
    #         results_dash[['rl_prob1', 'rl_prob2', 'rl_prob3']] = pd.DataFrame(results_dash.rl_prob.tolist(),
    #                                                                           index=results_dash.index)
    #
    #     # --------------------results_dash['label_pred'] = results_dash.label_pred.replace(parms['index_label'])
    #     for ix in range(len(results_dash['label_pred'])):
    #         for jx in range(len(results_dash['label_pred'][ix])):
    #             # replacing the value from the parms dictionary
    #             results_dash['label_pred'][ix].append(parms['index_label'][results_dash.label_pred[ix][jx]])
    #             # Converting probability into percentage
    #             results_dash['label_prob'][ix][jx] = (results_dash['label_prob'][ix][jx] * 100)
    #         # popping out the values from the list
    #         ln = int(len(results_dash['label_pred'][ix]) / 2)
    #         del results_dash['label_pred'][ix][:ln]
    #         # results_dash[['label_pred1', 'label_pred2', 'label_pred3']] = pd.DataFrame(
    #         results_dash[['label_pred1', 'label_pred2']] = pd.DataFrame(
    #             results_dash.label_pred.tolist(),
    #             index=results_dash.index)
    #         # results_dash[['label_prob1', 'label_prob2', 'label_prob3']] = pd.DataFrame(
    #         results_dash[['label_prob1', 'label_prob2']] = pd.DataFrame(
    #             results_dash.label_prob.tolist(),
    #             index=results_dash.index)
    #
    #     results_dash.drop(['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'label_pred', 'label_prob'], axis=1,
    #                       inplace=True)
    #     if parms['mode'] in ['next']:
    #         results_dash = results_dash[
    #             ['ac_expect', 'ac_pred1', 'ac_prob1', 'ac_pred2', 'ac_prob2',
    #             'rl_expect', 'rl_pred1', 'rl_prob1', 'rl_pred2', 'rl_prob2',
    #              'label_expect', 'label_pred1', 'label_prob1', 'label_pred2', 'label_prob2',
    #             "tm_expect", 'tm_pred']]
    #     elif parms['mode'] in ['batch']:
    #         results_dash = results_dash[
    #             ['caseid', 'ac_expect', 'ac_pred1', 'ac_prob1', 'ac_pred2', 'ac_prob2',
    #             'rl_expect', 'rl_pred1', 'rl_prob1', 'rl_pred2', 'rl_prob2',
    #             'label_expect', 'label_pred1', 'label_prob1', 'label_pred2', 'label_prob2',
    #             "tm_expect", 'tm_pred']]

        # ------------------------------------------------------------------------------------------------------------------------------------------------
        # results_dash.rename(
        #     columns={'caseid': 'Case_ID', 'ac_expect': 'Expected', 'ac_pred1': 'Prediction_1', 'ac_prob1': 'Confidence_1',
        #              'ac_pred2': 'Prediction_2', 'ac_prob2': 'Confidence_2', 'ac_pred3': 'Prediction_3', 'ac_prob3': 'Confidence_3',
        #              'rl_expect': 'Expected', 'rl_pred1': 'Predicted_1', 'rl_prob1': 'Confidence_1',
        #              'rl_pred2': 'Predicted_2', 'rl_prob2': 'Confidence_2', 'rl_pred3': 'Predicted_3', 'rl_prob3': 'Confidence_3',
        #              "tm_expect": 'Expected', 'tm_pred': 'Predicted'}, inplace=True)
        # results_dash.columns = pd.MultiIndex.from_tuples(
        #     zip(['', 'Activity', '', '', '', '', '', '', 'Role', '', '', '', '', '', '', 'Time', ''],
        #         results_dash.columns))
        # ------------------------------------------------------------------------------------------------------------------------------------------------
    # else:
    #     results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
    #     results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
    #     results_dash['label_pred'] = results_dash.label_pred.replace(parms['index_label'])
    #     results_dash['ac_prob'] = (results_dash['ac_prob'] * 100)
    #     results_dash['rl_prob'] = (results_dash['rl_prob'] * 100)
    #     results_dash['label_prob'] = (results_dash['label_prob'] * 100)
    #     results_dash.rename(
    #         columns={'caseid': 'Case_ID', 'ac_expect': 'Expected', 'ac_pred': 'Predicted', 'ac_prob': 'Confidence',
    #                  'rl_expect': 'Expected', 'rl_pred': 'Predicted', 'rl_prob': 'Confidence',
    #                  'label_expect': 'Expected', 'label_pred': 'Predicted', 'label_prob': 'Confidence',
    #                  "tm_expect": 'Expected', 'tm_pred': 'Predicted'}, inplace=True)
    #
    #     results_dash.columns = pd.MultiIndex.from_tuples(
    #         zip(['', 'Activity', '', '', 'Role', '', '', 'Label', '', '', 'Time', ''],
    #             results_dash.columns))
    #
    # st.table(results_dash)

    # expected = st.empty()
    #
    # prediction = st.empty()
    #
    # expected.write("This is a sample text")
    #
    # prediction.table(results_dash)
    #
    # prediction1, prediction2 = st.beta_columns(2)
    #
    # prediction1.subheader(" First Prediction")
    #
    # prediction1.subheader(" Second Prediction")
    #
    # st.title("Let's create a table!")
    # for i in range(1, 10)::
    #     cols = st.beta_columns(4)
    #     cols[0].write(f'{i}')
    #     cols[1].write(f'{i * i}')
    #     cols[2].write(f'{i * i * i}')
    #     cols[3].write('x' * i)

    # columns_results_dash = [('Case_ID', ''), ('Activity', 'Expected'), ('Activity', 'Predicted'), ('Activity', 'Confidence'),
    #           ('Role', 'Expected'), ('Role', 'Predicted'), ('Role', 'Confidence'),
    #           ('Time', 'Expected'), ('Time', 'Predicted')]
    # results_dash.columns = pd.MultiIndex.from_tuples(columns_results_dash)
    # results_dash.set_index('Case_ID', inplace=True)
    # results_dash.assign(idx='').set_index('idx')
    # results_dash.set_index('', inplace=True)
    #if parms['mode'] in ['next']:
    # Temporary Solution for multi column
    # if parms['mode'] in ['next']:
    #     st.table(results_dash)
    #     #st.dataframe(results_dash)
    # elif parms['mode'] in ['batch']:
    #     st.table(results_dash)


    @staticmethod
    def dashboard_prediction_batch(results_dash, parms):

        if parms['variant'] in ['multi_pred']:
            #converting the values to it's actual name from parms
            #--For Activity and Role
            ModelPredictor.dashboard_prediction_acrl(results_dash, parms)
            #--For Label
            ModelPredictor.dashboard_prediction_label(results_dash, parms)

        #     # --------------------results_dash['label_pred'] = results_dash.label_pred.replace(parms['index_label'])

            results_dash.drop(['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'label_pred', 'label_prob'], axis=1, inplace=True)

            if parms['mode'] in ['next']:
                results_dash = results_dash[
                    ['ac_expect', 'ac_pred1', 'ac_prob1', 'ac_pred2', 'ac_prob2',
                    'rl_expect', 'rl_pred1', 'rl_prob1', 'rl_pred2', 'rl_prob2',
                     'label_expect', 'label_pred1', 'label_prob1', 'label_pred2', 'label_prob2',
                    "tm_expect", 'tm_pred']]
            elif parms['mode'] in ['batch']:
                results_dash = results_dash[
                    ['caseid', 'ac_expect', 'ac_pred1', 'ac_prob1', 'ac_pred2', 'ac_prob2',
                    'rl_expect', 'rl_pred1', 'rl_prob1', 'rl_pred2', 'rl_prob2',
                    'label_expect', 'label_pred1', 'label_prob1', 'label_pred2', 'label_prob2',
                    "tm_expect", 'tm_pred']]

        else:
            results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
            results_dash['rl_pred'] = results_dash.rl_pred.replace(parms['index_rl'])
            results_dash['label_pred'] = results_dash.label_pred.replace(parms['index_label'])
            results_dash['ac_prob'] = (results_dash['ac_prob'] * 100)
            results_dash['rl_prob'] = (results_dash['rl_prob'] * 100)
            results_dash['label_prob'] = (results_dash['label_prob'] * 100)
            results_dash.rename(
                columns={'caseid': 'Case_ID', 'ac_expect': 'Expected', 'ac_pred': 'Predicted', 'ac_prob': 'Confidence',
                         'rl_expect': 'Expected', 'rl_pred': 'Predicted', 'rl_prob': 'Confidence',
                         'label_expect': 'Expected', 'label_pred': 'Predicted', 'label_prob': 'Confidence',
                         "tm_expect": 'Expected', 'tm_pred': 'Predicted'}, inplace=True)

            results_dash.columns = pd.MultiIndex.from_tuples(
                zip(['', 'Activity', '', '', 'Role', '', '', 'Label', '', '', 'Time', ''],
                    results_dash.columns))

        st.table(results_dash)

    @staticmethod
    def dashboard_prediction_acrl(results_dash, parms):
        # --------------------results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        for ix in range(len(results_dash['ac_pred'])):
            for jx in range(len(results_dash['ac_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['ac_pred'][ix].append(parms['index_ac'][results_dash.ac_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['ac_prob'][ix][jx] = (results_dash['ac_prob'][ix][jx] * 100)
            # poping out the values from the list
            ln = int(len(results_dash['ac_pred'][ix]) / 2)
            del results_dash['ac_pred'][ix][:ln]
            results_dash[['ac_pred1', 'ac_pred2']] = pd.DataFrame(results_dash.ac_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[['ac_prob1', 'ac_prob2']] = pd.DataFrame(results_dash.ac_prob.tolist(),
                                                                  index=results_dash.index)

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
            results_dash[['rl_pred1', 'rl_pred2']] = pd.DataFrame(results_dash.rl_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[['rl_prob1', 'rl_prob2']] = pd.DataFrame(results_dash.rl_prob.tolist(),
                                                                  index=results_dash.index)

        return results_dash

    @staticmethod
    def dashboard_prediction_label(results_dash, parms):
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
