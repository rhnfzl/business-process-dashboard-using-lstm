# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:35:53 2020

@author: Manuel Camargo
"""
import numpy as np
import pandas as pd
import streamlit as st
#import time

from support_modules import support as sup


class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        #self.nx = 2

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        #print("spl :", spl)
        self.spl = spl
        print("What is Imp :", imp)
        self.imp = imp
        # if params['mode'] == 'next':
        #     fltr_idx = params['nextcaseid_attr']["filter_index"]
        #     spl_df_prefx = pd.DataFrame(self.spl['prefixes'])[fltr_idx:]
        #     spl_df_next = pd.DataFrame(self.spl['next_evt'])[fltr_idx:]
        #     st.subheader("Prefixes")
        #     st.table(spl_df_prefx)
        #     st.subheader("Next Event")
        #     st.table(spl_df_next)
            #print("spl :", spl_df)
        self.nx = params['multiprednum']
        predictor = self._get_predictor(params['model_type'], params['mode'], params['next_mode'])
        sup.print_performed_task('Predicting next events')

        return predictor(params, vectorizer)

    def _get_predictor(self, model_type, mode, next_mode):
        # OJO: This is an extension point just incase
        # a different predictor being neccesary
        if mode == 'next':
            if next_mode == 'next_action':
                return self._predict_next_event_suffix_cat_next #Predicts what would be the prediction of the current case and what system has shown in the past
            elif next_mode == 'history_with_next':
                return self._predict_next_event_shared_cat_next  # Predicts what would come next and after
        elif mode == 'batch':
            return self._predict_next_event_shared_cat_batch

    def _predict_next_event_shared_cat_next(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"] + 1

        print("Activity Suffix :", self.spl['next_evt']['activities'])

        #print("Prediction Activity :", self.spl['prefixes']['activities'][pred_fltr_idx], type(self.spl['prefixes']['activities']))
        results = list()
        for i, _ in enumerate(self.spl['prefixes']['activities'][:pred_fltr_idx]):
            print("Activity Prefixes :", self.spl['prefixes']['activities'][:pred_fltr_idx][i])

            # Activities and roles input shape(1,5)
            # print("i :", i)
            # print("parameters['dim'] :", parameters['dim'])
            # print("parameters['dim']['time_dim'] :", parameters['dim']['time_dim'])
            # print("self.spl['prefixes']['activities']:", self.spl['prefixes']['activities'])
            # print("self.spl['prefixes']['activities'][pred_fltr_idx:] :", self.spl['prefixes']['activities'][:pred_fltr_idx])
            # print("np.array(self.spl['prefixes']['activities'][pred_fltr_idx:][i]) :", np.array(self.spl['prefixes']['activities'][:pred_fltr_idx][i]))
            #print("[-parameters['dim']['time_dim']:] :", ([-parameters['dim']['time_dim']:])

            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][:pred_fltr_idx][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            #print("x_ac_ngram :", x_ac_ngram)
            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][:pred_fltr_idx][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            # times input shape(1,5,1)
            times_attr_num = (self.spl['prefixes']['times'][:pred_fltr_idx][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parameters['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][:pred_fltr_idx][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], times_attr_num))])
            # add intercase features if necessary
            # if vectorizer in ['basic']:
            #     inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            #
            # elif vectorizer in ['inter']:
                # times input shape(1,5,1)
            inter_attr_num = (self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i].shape[1])
            x_inter_ngram = np.array(
                [np.append(np.zeros((
                    parameters['dim']['time_dim'], inter_attr_num)),
                    self.spl['prefixes']['inter_attr'][:pred_fltr_idx][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], inter_attr_num))])
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            # print("Model Prediction  :", preds)
            # print("Activity :", np.array(preds[0][0]))
            # print("Role :", np.array(preds[1][0]))

            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]


            elif self.imp == 'arg_max':
                # Use this to get the max prediction

                # print("Activity :", np.array(preds[0][0]))
                # print("Role :", np.array(preds[1][0]))

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]

            elif self.imp == 'multi_pred':
                #change it to get the number of predictions
                #nx = 2


                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()

                pos_prob = []
                pos1_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])

                #print("activity = ", posac)
                #print("activity probability = ", pos_probac)

                #print("role = ", pos1rl)
                #print("role probability = ", pos1_probrl)

            # save results
            predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]
            # print("Predictions 1 :", preds)
            # print("Predictions 2 :", preds[3])
            # print("Predictions 3 :", preds[3][0])
            # print("Predictions 4 :", preds[2][0][0])
            print("Predict Next Activity :", pos)
            #print(pos, pos1, pos2, preds[2][0][0], pos_prob, pos1_prob, pos2_prob)

            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(self._create_result_record_next(i, self.spl, predictions, parameters))
        sup.print_done_task()
        return results

    def _create_result_record_next(self, index, spl, preds, parms):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()
        #print("Preds under result :", preds)
        #record['caseid'] = parms['caseid'][_fltr_idx][index]
        record['ac_prefix'] = spl['prefixes']['activities'][:_fltr_idx][index]
        record['ac_expect'] = spl['next_evt']['activities'][:_fltr_idx][index]
        record['ac_pred'] = preds[0]
        record['ac_prob'] = preds[3]
        record['rl_prefix'] = spl['prefixes']['roles'][:_fltr_idx][index]
        record['rl_expect'] = spl['next_evt']['roles'][:_fltr_idx][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[4]



        if parms['one_timestamp']:
            #print("Time Next Event :", spl['next_evt']['times'][:_fltr_idx])
            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0],
                parms, parms['scale_args'])
            #print("Predicted :", preds)
            record['tm_pred'] = self.rescale(
                preds[2], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        return record

    def _predict_next_event_shared_cat_batch(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        results = list()
        for i, _ in enumerate(self.spl['prefixes']['activities']):
            # Activities and roles input shape(1,5)
            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            # times input shape(1,5,1)
            times_attr_num = (self.spl['prefixes']['times'][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parameters['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], times_attr_num))]
                )

            # add intercase features if necessary
            # if vectorizer in ['basic']:
            #     inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            #
            # elif vectorizer in ['inter']:
                # times input shape(1,5,1)
            inter_attr_num = (self.spl['prefixes']['inter_attr'][i].shape[1])
            x_inter_ngram = np.array(
                [np.append(np.zeros((
                    parameters['dim']['time_dim'], inter_attr_num)),
                    self.spl['prefixes']['inter_attr'][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                )
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            #print("input_time : ", x_t_ngram)
            # predict
            preds = self.model.predict(inputs)
            #print("Predictions :", preds)
            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]

            elif self.imp == 'arg_max':
                # Use this to get the max prediction
                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]



            elif self.imp == 'multi_pred':
                #change it to get the number of predictions
                #nx = 2

                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()

                pos_prob = []
                pos1_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])

                #print("activity = ", posac)
                #print("activity probability = ", pos_probac)

                #print("role = ", pos1rl)
                #print("role probability = ", pos1_probrl)

            # save results
            predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]

            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(self.create_result_record_batch(i, self.spl, predictions, parameters))
        sup.print_done_task()
        return results

    def create_result_record_batch(self, index, spl, preds, parms):
        record = dict()

        record['caseid'] = parms['caseid'][index]
        record['ac_prefix'] = spl['prefixes']['activities'][index]
        record['ac_expect'] = spl['next_evt']['activities'][index]
        record['ac_pred'] = preds[0]
        record['ac_prob'] = preds[3]
        record['rl_prefix'] = spl['prefixes']['roles'][index]
        record['rl_expect'] = spl['next_evt']['roles'][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[4]

        # print("ac_pred : ", record['ac_pred'])
        # print('ac_prob : ', record['ac_prob'])
        # print('rl_pred : ', record['rl_pred'])
        # print('rl_prob :', record['rl_prob'])

        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][index][0],
                parms, parms['scale_args'])
            #print("Predicted :", preds)
            record['tm_pred'] = self.rescale(
                preds[2], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        return record

    def _predict_next_event_suffix_cat_next(self, parameters, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions
        print("Filtered Index : ", parameters['nextcaseid_attr']["filter_index"])
        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"]

        print("pred_fltr_idx :", pred_fltr_idx, parameters['nextcaseid_attr']["filter_index"])

        #print("Index :", pred_fltr_idx)
        #print("Prediction Activity prefixes :", self.spl['prefixes']['activities'][pred_fltr_idx:], type(self.spl['prefixes']['activities']))
        print("Activity next_evt :", self.spl['next_evt']['activities'])

        results = list()
        # for i, _ in enumerate(self.spl['prefixes']['activities'][pred_fltr_idx:]):
        for i, _ in enumerate(self.spl['prefixes']['activities'][pred_fltr_idx:pred_fltr_idx+1]):
            print("Activity prefixes :", self.spl['prefixes']['activities'][pred_fltr_idx-1:pred_fltr_idx])
            # Activities and roles input shape(1,5)
            x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][pred_fltr_idx:][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))
            x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][pred_fltr_idx:][i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                .reshape((1, parameters['dim']['time_dim'])))

            # times input shape(1,5,1)
            times_attr_num = (self.spl['prefixes']['times'][pred_fltr_idx:][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parameters['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][pred_fltr_idx:][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], times_attr_num))]
                )
            # add intercase features if necessary
            # if vectorizer in ['basic']:
            #     inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            #
            # elif vectorizer in ['inter']:
                # times input shape(1,5,1)
            inter_attr_num = (self.spl['prefixes']['inter_attr'][pred_fltr_idx:][i].shape[1])
            x_inter_ngram = np.array(
                [np.append(np.zeros((
                    parameters['dim']['time_dim'], inter_attr_num)),
                    self.spl['prefixes']['inter_attr'][pred_fltr_idx:][i], axis=0)
                    [-parameters['dim']['time_dim']:]
                    .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                )
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            # print("Model Prediction  :", preds)
            # print("Activity :", np.array(preds[0][0]))
            # print("Role :", np.array(preds[1][0]))

            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]

            elif self.imp == 'arg_max':
                # Use this to get the max prediction

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]


            elif self.imp == 'multi_pred':
                #change it to get the number of predictions
                #nx = 2


                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()


                pos_prob = []
                pos1_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])

                #print("activity = ", posac)
                #print("activity probability = ", pos_probac)

                print("Activity Predicted :", pos, (-acx).argsort()[:self.nx])

            # save results
            predictions = [pos, pos1, preds[2][0][0], pos_prob, pos1_prob]

            if not parameters['one_timestamp']:
                predictions.extend([preds[2][0][1]])
            results.append(self.__create_result_record_next(i, self.spl, predictions, parameters))
        sup.print_done_task()
        return results

    def __create_result_record_next(self, index, spl, preds, parms):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()
        #record['caseid'] = parms['caseid'][_fltr_idx:][index]
        record['ac_prefix'] = spl['prefixes']['activities'][_fltr_idx:][index]
        record['ac_expect'] = spl['next_evt']['activities'][_fltr_idx:][index]
        record['ac_pred'] = preds[0]
        record['ac_prob'] = preds[3]
        record['rl_prefix'] = spl['prefixes']['roles'][_fltr_idx:][index]
        record['rl_expect'] = spl['next_evt']['roles'][_fltr_idx:][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[4]

        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][0],
                parms, parms['scale_args'])
            #print("Predicted :", preds)
            record['tm_pred'] = self.rescale(
                preds[2], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[2], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['wait'])
        return record


    @staticmethod
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
