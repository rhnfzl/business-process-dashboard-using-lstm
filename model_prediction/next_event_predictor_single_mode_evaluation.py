# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:35:53 2020

@author: Manuel Camargo
"""
import random
import numpy as np
import pandas as pd
import streamlit as st


from support_modules import support as sup
from model_prediction import interfaces as it

class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        #-----------------------------------------
        # self.log = pd.DataFrame
        # self.ac_index = dict()
        # self.rl_index = dict()
        # self.label_index = dict()
        # self._samplers = dict()
        # #self._samp_dispatcher = {'basic': self._sample_next_event}

    #parms : parameters
    #model : .h5 file
    #spl : prefix or the event log
    #imp : Value is 1 by default, basically signifies how many times the prediction has to be made on same event
    #vectorizer : 'basic' value by default


    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        #print("spl :", spl)
        self.spl = spl
        self.imp = imp
        if params['mode'] == 'next':
            fltr_idx = params['nextcaseid_attr']["filter_index"]
            spl_df_prefx = pd.DataFrame(self.spl['prefixes'])[fltr_idx:]
            spl_df_next = pd.DataFrame(self.spl['next_evt'])[fltr_idx:]
            #st.subheader("Prefixes")
            #st.table(spl_df_prefx)
            #st.subheader("Next Event")
            #st.table(spl_df_next)
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

        # print("----------------Suffix--------------------")
        # print("Activity Suffix :", self.spl['next_evt']['activities'])
        # print("Role Suffix :", self.spl['next_evt']['roles'])
        # print("Label Suffix :", self.spl['next_evt']['label'])
        # print("Time Suffix :", self.spl['next_evt']['times'], type(self.spl['next_evt']['times']))
        #
        # print("----------------SME Prefix--------------------")
        # print("Activity Prefixes :", self.spl['prefixes']['activities'][:pred_fltr_idx])
        #
        # print("Role Prefixes :", self.spl['prefixes']['roles'][:pred_fltr_idx])
        #
        # print("Label Prefix :", self.spl['prefixes']['label'][:pred_fltr_idx])
        #
        # print("Time Prefixes :", self.spl['prefixes']['times'][:pred_fltr_idx],
        #       type(self.spl['prefixes']['times'][:pred_fltr_idx]),
        #       type(self.spl['prefixes']['times'][:pred_fltr_idx][0]))

        print("Index Value :", pred_fltr_idx)

        if parameters['predchoice'] not in ['SME']:
            # print("----------------Prediction Prefix--------------------")
            # self._sessionstate_check(pred_fltr_idx)

            if 'pos_ac_ss' in st.session_state:
                # print("Input of Series :", st.session_state['pos_ac_ss'])

                serie_predict_ac = [st.session_state['pos_ac_ss'][:idx]
                                    for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                y_serie_predict_ac = [x[-1] for x in
                                      serie_predict_ac]  # selecting the last value from each list of list

                # print("serie Activity:", serie_predict_ac)
                # print("y-serie Activity:", y_serie_predict_ac)
                # Extra Check for the same length as the prediction would expect
                if len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == len(serie_predict_ac):
                    self.spl['prefixes']['activities'][:pred_fltr_idx] = serie_predict_ac

                    # print("Replaced Activity prefix : ", self.spl['prefixes']['activities'][:pred_fltr_idx])

            elif 'pos_ac_ss' not in st.session_state:
                st.session_state['pos_ac_ss'] = [0]

            # --Role
            if 'pos_rl_ss' in st.session_state:
                serie_predict_rl = [st.session_state['pos_rl_ss'][:idx]
                                    for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                y_serie_predict_rl = [x[-1] for x in
                                      serie_predict_rl]  # selecting the last value from each list of list

                # print("serie Role:", serie_predict_rl)
                # print("y-serie Role:", y_serie_predict_rl)

                # Extra Check for the same length as the prediction would expect
                if len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == len(serie_predict_rl):
                    self.spl['prefixes']['roles'][:pred_fltr_idx] = serie_predict_rl
                    # print("Replaced Role prefix : ", self.spl['prefixes']['roles'][:pred_fltr_idx])



            elif 'pos_rl_ss' not in st.session_state:
                st.session_state['pos_rl_ss'] = [0]

            # --Label

            if 'pos_lb_ss' in st.session_state:
                serie_predict_lb = [st.session_state['pos_lb_ss'][:idx]
                                    for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                y_serie_predict_lb = [x[-1] for x in
                                      serie_predict_lb]  # selecting the last value from each list of list

                # print("serie label:", serie_predict_lb)
                # print("y-serie label:", y_serie_predict_lb)

                # Extra Check for the same length as the prediction would expect
                if len(self.spl['prefixes']['label'][:pred_fltr_idx]) == len(serie_predict_lb):
                    self.spl['prefixes']['label'][:pred_fltr_idx] = serie_predict_lb
                    # print("Replaced Label prefix : ", self.spl['prefixes']['label'][:pred_fltr_idx])

            elif 'pos_lb_ss' not in st.session_state:
                # Initializing it with whatever is there in the testlog at postion 0 (i.e initial pos)
                st.session_state['pos_lb_ss'] = self.spl['prefixes']['label'][:pred_fltr_idx][0]

            # --Time

            if 'pos_tm_ss' in st.session_state:

                # print("Actual Shape of tm :", st.session_state['pos_tm_ss'])

                serie_predict_tm = [np.array(st.session_state['pos_tm_ss'][:idx])
                                    for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start

                y_serie_predict_tm = [x[-1] for x in
                                      serie_predict_tm]  # selecting the last value from each list of list

                # print("serie time:", serie_predict_tm)
                # print("y-serie time:", y_serie_predict_tm)

                # Extra Check for the same length as the prediction would expect
                if len(self.spl['prefixes']['times'][:pred_fltr_idx]) == len(serie_predict_tm):
                    self.spl['prefixes']['times'][:pred_fltr_idx] = serie_predict_tm
                    # print("Replaced Time prefix : ", self.spl['prefixes']['times'][:pred_fltr_idx])

            elif 'pos_tm_ss' not in st.session_state:
                st.session_state['pos_tm_ss'] = [[0]]

            # if parameters['predchoice'] not in ['SME']:
            #     print("----------------Prediction Prefix--------------------")
            #     print("Activity Prefixes :", self.spl['prefixes']['activities'][:pred_fltr_idx])
            #     print("Role Prefixes :", self.spl['prefixes']['roles'][:pred_fltr_idx])
            #     print("Label Prefix :", self.spl['prefixes']['label'][:pred_fltr_idx])
            #     print("Time Prefixes :", self.spl['prefixes']['times'][:pred_fltr_idx],
            #           type(self.spl['prefixes']['times'][:pred_fltr_idx]),
            #           type(self.spl['prefixes']['times'][:pred_fltr_idx][0]))




        #print("Prediction Activity :", self.spl['prefixes']['activities'][pred_fltr_idx], type(self.spl['prefixes']['activities']))
        results = list()
        for i, _ in enumerate(self.spl['prefixes']['activities'][:pred_fltr_idx]):

                # elif 'pos_tm_ss' not in st.session_state:
                #     st.session_state['pos_ac_ss'] = [0]

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
            x_label_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['label'][:pred_fltr_idx][i]),
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
            #     inputs = [x_ac_ngram, x_rl_ngram, x_label_ngram, x_t_ngram]
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
            inputs = [x_ac_ngram, x_rl_ngram, x_label_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            # print("Model Prediction  :", preds)
            # print("Activity :", np.array(preds[0][0]))
            # print("Role :", np.array(preds[1][0]))
            # print("Label :", np.array(preds[2][0]))

            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]
                pos2 = np.random.choice(np.arange(0, len(preds[2][0])),
                                        p=preds[2][0])
                pos2_prob = preds[2][0][pos2]


            elif self.imp == 'arg_max':
                # Use this to get the max prediction

                # print("Activity :", np.array(preds[0][0]))
                # print("Role :", np.array(preds[1][0]))
                # print("Label :", np.array(preds[2][0]))

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]

                pos2 = np.argmax(preds[2][0])
                pos2_prob = preds[2][0][pos2]

            elif self.imp == 'multi_pred':
                #change it to get the number of predictions
                #nx = 2


                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])
                lbx = np.array(preds[2][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()
                pos2 = (-lbx).argsort()[:self.nx].tolist()

                pos_prob = []
                pos1_prob = []
                pos2_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])
                for kx in range(len(pos2)):
                    # probability of label
                    pos2_prob.append(lbx[pos2[kx]])

                #print("activity = ", posac)
                #print("activity probability = ", pos_probac)

                #print("role = ", pos1rl)
                #print("role probability = ", pos1_probrl)

            # save results
            predictions = [pos, pos1, pos2, preds[3][0][0], pos_prob, pos1_prob, pos2_prob]

            # print("Predictions 1 :", preds)
            # print("Predictions 2 :", preds[3])
            # print("Predictions 3 :", preds[3][0])
            # print("Predictions 4 :", preds[3][0][0])

            #---Adding Session State Logic to Save the Predictions
            # print("Predict Next Activity :", pos, type(pos))
            # print("Predict Next Role :", pos1, type(pos1))
            # print("Predict Next Label :", pos2, type(pos2))
            # print("Predict Next Time :", preds[3][0][0], type(preds[3][0][0]))

            #SME Mode assumes that Functional Knowledge person is taking action
            if i == pred_fltr_idx-1 and parameters['predchoice'] not in ['SME']:
                #-Time
                st.session_state['pos_tm_ss'].extend([[preds[3][0][0]]])
                if  parameters['multiprednum'] > 1:

                    print("Multi Pred Number : ", parameters['multiprednum'], parameters['predchoice'],
                          (int(parameters['predchoice'][-1]) - 1), int(parameters['predchoice'][-1]))

                    # Selecting the Choice of Prediction for each run, int(parameters['predchoice'][-1])
                    # takes the last value of the choice and coverts it into integer, then to match with
                    # index -1 is subtracted, similarly we need only one value so, value after colon to
                    # match the position
                    #-Activity
                    st.session_state['pos_ac_ss'].extend(
                        pos[(int(parameters['predchoice'][-1]) - 1):int(parameters['predchoice'][-1])])
                    #-Role
                    st.session_state['pos_rl_ss'].extend(
                        pos1[(int(parameters['predchoice'][-1]) - 1):int(parameters['predchoice'][-1])])

                    #-Label
                    print("Label Prediction : ", pos2, pos2[0])
                    if parameters['multiprednum'] > 2:
                        #-- What to provide in case of prediction is selected more than 2
                        #-Sol1 : give the test log value to it (Biased)
                        #st.session_state['pos_lb_ss'].extend(self.spl['prefixes']['label'][:pred_fltr_idx][0])
                        ##-Sol2 : give the first predicted value
                        # st.session_state['pos_lb_ss'].extend(pos2[0])
                        #-Sol3 : give the second predicted value
                        # st.session_state['pos_lb_ss'].extend(pos2[1])
                        #-Sol4 : Let the system select it randomly (Probably not biased)
                        st.session_state['pos_lb_ss'].extend([random.choice(pos2)])

                    elif parameters['multiprednum'] <= 2:
                        st.session_state['pos_lb_ss'].extend(
                            pos2[(int(parameters['predchoice'][-1]) - 1):int(parameters['predchoice'][-1])])

                elif parameters['multiprednum'] == 1:
                    #When the prediction is selected as the Max probability
                    st.session_state['pos_ac_ss'].extend([pos])

                    st.session_state['pos_rl_ss'].extend([pos1])

                    st.session_state['pos_lb_ss'].extend([pos1])

                #print("Session State Activity Pred : ", st.session_state['pos_ac_ss'])

                #print("Session State : ", st.session_state)

            #print("Session State Activity Append : ", st.session_state._activity_pred_append)

            #print("What is there in the Session State : ", st.session_state)

            #print("Case Id :", parameters['caseid'])
            #print(pos, pos1, pos2, preds[3][0][0], pos_prob, pos1_prob, pos2_prob)

            if not parameters['one_timestamp']:
                predictions.extend([preds[3][0][1]])
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
        record['ac_prob'] = preds[4]
        record['rl_prefix'] = spl['prefixes']['roles'][:_fltr_idx][index]
        record['rl_expect'] = spl['next_evt']['roles'][:_fltr_idx][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[5]
        record['label_prefix'] = spl['prefixes']['label'][:_fltr_idx][index]
        record['label_expect'] = spl['next_evt']['label'][:_fltr_idx][index]
        record['label_pred'] = preds[2]
        record['label_prob'] = preds[6]



        if parms['one_timestamp']:

            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0],
                parms, parms['scale_args'])
            record['tm_pred'] = self.rescale(
                preds[3], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][:_fltr_idx][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][:_fltr_idx][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[4], parms, parms['scale_args']['wait'])
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
            x_label_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(self.spl['prefixes']['label'][pred_fltr_idx:][i]),
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
            #     inputs = [x_ac_ngram, x_rl_ngram, x_label_ngram, x_t_ngram]
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
            inputs = [x_ac_ngram, x_rl_ngram, x_label_ngram, x_t_ngram, x_inter_ngram]
            # predict
            preds = self.model.predict(inputs)

            # print("Model Prediction  :", preds)
            # print("Activity :", np.array(preds[0][0]))
            # print("Role :", np.array(preds[1][0]))
            # print("Label :", np.array(preds[2][0]))

            if self.imp == 'random_choice':
                # Use this to get a random choice following as PDF
                pos = np.random.choice(np.arange(0, len(preds[0][0])),
                                       p=preds[0][0])
                pos_prob = preds[0][0][pos]
                pos1 = np.random.choice(np.arange(0, len(preds[1][0])),
                                        p=preds[1][0])
                pos1_prob = preds[1][0][pos1]
                pos2 = np.random.choice(np.arange(0, len(preds[2][0])),
                                        p=preds[2][0])
                pos2_prob = preds[2][0][pos2]

            elif self.imp == 'arg_max':
                # Use this to get the max prediction

                pos = np.argmax(preds[0][0])
                pos_prob = preds[0][0][pos]

                pos1 = np.argmax(preds[1][0])
                pos1_prob = preds[1][0][pos1]

                pos2 = np.argmax(preds[2][0])
                pos2_prob = preds[2][0][pos2]

            elif self.imp == 'multi_pred':
                #change it to get the number of predictions
                #nx = 2


                #changing array to numpy
                acx = np.array(preds[0][0])
                rlx = np.array(preds[1][0])
                lbx = np.array(preds[2][0])

                pos = (-acx).argsort()[:self.nx].tolist()
                pos1 = (-rlx).argsort()[:self.nx].tolist()
                pos2 = (-lbx).argsort()[:self.nx].tolist()

                # print("Label after changing :", pos2, "---", lbx,  "---", -lbx,  "---",  (-lbx).argsort()[:self.nx],
                #       "---", (-lbx).argsort())

                pos_prob = []
                pos1_prob = []
                pos2_prob = []

                for ix in range(len(pos)):
                    # probability of activity
                    pos_prob.append(acx[pos[ix]])
                for jx in range(len(pos1)):
                    # probability of role
                    pos1_prob.append(rlx[pos1[jx]])
                for kx in range(len(pos2)):
                    # probability of label
                    pos2_prob.append(lbx[pos2[kx]])

                #print("activity = ", posac)
                #print("activity probability = ", pos_probac)

                # print("Label = ", pos2)
                # print("Label probability = ", pos2_prob)
                print("Activity Predicted :", pos, (-acx).argsort()[:self.nx])

            # save results
            predictions = [pos, pos1, pos2, preds[3][0][0], pos_prob, pos1_prob, pos2_prob]

            if not parameters['one_timestamp']:
                predictions.extend([preds[3][0][1]])
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
        record['ac_prob'] = preds[4]
        record['rl_prefix'] = spl['prefixes']['roles'][_fltr_idx:][index]
        record['rl_expect'] = spl['next_evt']['roles'][_fltr_idx:][index]
        record['rl_pred'] = preds[1]
        record['rl_prob'] = preds[5]
        record['label_prefix'] = spl['prefixes']['label'][_fltr_idx:][index]
        record['label_expect'] = spl['next_evt']['label'][_fltr_idx:][index]
        record['label_pred'] = preds[2]
        record['label_prob'] = preds[6]

        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['tm_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][0],
                parms, parms['scale_args'])
            #print("Predicted :", preds)
            record['tm_pred'] = self.rescale(
                preds[3], parms, parms['scale_args'])

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][0], parms,
                parms['scale_args']['dur'])
            record['dur_pred'] = self.rescale(
                preds[3], parms, parms['scale_args']['dur'])
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait'])
                for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['wait_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][1], parms,
                parms['scale_args']['wait'])
            record['wait_pred'] = self.rescale(
                preds[4], parms, parms['scale_args']['wait'])
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

    # def _sessionstate_check(self, pred_fltr_idx):
        # Activity
        # if 'pos_ac_ss' in st.session_state:
        #     print("Input of Series :", st.session_state['pos_ac_ss'])
        #
        #     serie_predict_ac = [st.session_state['pos_ac_ss'][:idx]
        #                         for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
        #
        #     y_serie_predict_ac = [x[-1] for x in
        #                           serie_predict_ac]  # selecting the last value from each list of list
        #
        #     print("serie Activity:", serie_predict_ac)
        #     print("y-serie Activity:", y_serie_predict_ac)
        #     # Extra Check for the same length as the prediction would expect
        #     if len(self.spl['prefixes']['activities'][:pred_fltr_idx]) == serie_predict_ac:
        #         self.spl['prefixes']['activities'][:pred_fltr_idx] = serie_predict_ac
        #
        #     print("Replaced prefix : ", self.spl['prefixes']['activities'][:pred_fltr_idx])
        #
        # elif 'pos_ac_ss' not in st.session_state:
        #     st.session_state['pos_ac_ss'] = [0]
        # # --Role
        # if 'pos_rl_ss' in st.session_state:
        #     serie_predict_rl = [st.session_state['pos_rl_ss'][:idx]
        #                         for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
        #
        #     y_serie_predict_rl = [x[-1] for x in
        #                           serie_predict_rl]  # selecting the last value from each list of list
        #
        #     print("serie Role:", serie_predict_rl)
        #     print("y-serie Role:", y_serie_predict_rl)
        #
        #     # Extra Check for the same length as the prediction would expect
        #     if len(self.spl['prefixes']['roles'][:pred_fltr_idx]) == serie_predict_rl:
        #         self.spl['prefixes']['roles'][:pred_fltr_idx] = serie_predict_rl
        #
        # elif 'pos_rl_ss' not in st.session_state:
        #     st.session_state['pos_rl_ss'] = [0]
        # # --Label
        #
        # if 'pos_lb_ss' in st.session_state:
        #     serie_predict_lb = [st.session_state['pos_lb_ss'][:idx]
        #                         for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
        #
        #     y_serie_predict_lb = [x[-1] for x in
        #                           serie_predict_lb]  # selecting the last value from each list of list
        #
        #     print("serie label:", serie_predict_lb)
        #     print("y-serie label:", y_serie_predict_lb)
        #
        #     # Extra Check for the same length as the prediction would expect
        #     if len(self.spl['prefixes']['label'][:pred_fltr_idx]) == serie_predict_lb:
        #         self.spl['prefixes']['label'][:pred_fltr_idx] = serie_predict_lb
        #
        # elif 'pos_lb_ss' not in st.session_state:
        #     # Initializing it with whatever is there in the testlog at postion 0 (i.e initial pos)
        #     st.session_state['pos_lb_ss'] = self.spl['prefixes']['label'][:pred_fltr_idx][0]
        #
        # # --Time
        #
        # if 'pos_tm_ss' in st.session_state:
        #
        #     print("Actual Shape of tm :", st.session_state['pos_tm_ss'])
        #
        #     serie_predict_tm = [np.array(st.session_state['pos_tm_ss'][:idx])
        #                         for idx in range(1, pred_fltr_idx + 1)]  # range starts with 1 to avoid start
        #
        #     y_serie_predict_tm = [x[-1] for x in
        #                           serie_predict_tm]  # selecting the last value from each list of list
        #
        #     print("serie time:", serie_predict_tm)
        #     print("y-serie time:", y_serie_predict_tm)
        #
        #     # Extra Check for the same length as the prediction would expect
        #     if len(self.spl['prefixes']['times'][:pred_fltr_idx]) == serie_predict_tm:
        #         self.spl['prefixes']['times'][:pred_fltr_idx] = serie_predict_tm
        #
        #
        # elif 'pos_tm_ss' not in st.session_state:
        #     st.session_state['pos_tm_ss'] = [[0]]
