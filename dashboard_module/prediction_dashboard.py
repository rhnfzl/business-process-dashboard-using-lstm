import os
import sys

import streamlit as st
import pandas as pd
import numpy as np

class DashPredictor():

    def __init__(self, pred_results_df, parms):
        """constructor"""
        self.pred_results_df = pred_results_df
        self.parms = parms

    @staticmethod
    def dashboard_prediction(pred_results_df, parms):
        # Removing 'ac_prefix', 'rl_prefix', 'tm_prefix', 'run_num', 'implementation' from the result
        results_dash = pred_results_df[
            ['ac_prefix', 'rl_prefix', 'label_prefix', 'tm_prefix', 'run_num', 'implementation']].copy()
        results_dash = pred_results_df.drop(
            ['ac_prefix', 'rl_prefix', 'label_prefix', 'tm_prefix', 'run_num', 'implementation'], axis=1)

        print("Type of result dash :", type(results_dash))
        # Replacing from Dictionary Values to it's original name
        results_dash['ac_expect'] = results_dash.ac_expect.replace(parms['index_ac'])
        results_dash['rl_expect'] = results_dash.rl_expect.replace(parms['index_rl'])
        results_dash['label_expect'] = results_dash.label_expect.replace(parms['index_label'])

        # results_aggrid = results_dash
        # AgGrid(results_dash)
        # st.write(results_dash)
        if parms['mode'] in ['batch']:
            # as the static function is calling static function class has to be mentioned
            DashPredictor.dashboard_prediction_batch(results_dash, parms)
        elif parms['mode'] in ['next']:
            DashPredictor.dashboard_prediction_next(results_dash, parms)


    @staticmethod
    def dashboard_prediction_next(results_dash, parms):
        if parms['variant'] in ['multi_pred']:

            # converting the values to it's actual name from parms
            # --For Activity and Role
            DashPredictor.dashboard_multiprediction_acrl(results_dash, parms)
            # --For Label
            DashPredictor.dashboard_multiprediction_label(results_dash, parms)

        else:
            # converting the values to it's actual name from parms
            # --For Activity, Role and Label
            DashPredictor.dashboard_maxprediction(results_dash, parms)

        DashPredictor.dashboard_nextprediction_write(results_dash, parms)


    @staticmethod
    def dashboard_prediction_batch(results_dash, parms):
        # All the results has to be displayed in Tabular form i.e DataFrame
        if parms['variant'] in ['multi_pred']:
            # converting the values to it's actual name from parms
            # --For Activity and Role
            DashPredictor.dashboard_multiprediction_acrl(results_dash, parms)
            # --For Label
            DashPredictor.dashboard_multiprediction_label(results_dash, parms)

            results_dash.drop(['ac_pred', 'ac_prob', 'rl_pred', 'rl_prob', 'label_pred', 'label_prob'], axis=1,
                              inplace=True)

            multipreddict = DashPredictor.dashboard_multiprediction_columns(parms)

            _lst = [['caseid', 'ac_expect'] + multipreddict["ac_pred"] + multipreddict["ac_prob"] +
                    ['rl_expect'] + multipreddict["rl_pred"] + multipreddict["rl_prob"] +
                    ['label_expect', 'label_pred1', 'label_pred2', 'label_prob1', 'label_prob2', 'tm_expect', 'tm_pred']]

            results_dash = results_dash[_lst[0]]

        else:
            # converting the values to it's actual name from parms
            # --For Activity, Role and Label
            DashPredictor.dashboard_maxprediction(results_dash, parms)
            results_dash.rename(
                columns={'caseid': 'Case_ID', 'ac_expect': 'AC Expected', 'ac_pred': 'AC Predicted',
                         'ac_prob': 'AC Confidence',
                         'rl_expect': 'RL Expected', 'rl_pred': 'RL Predicted', 'rl_prob': 'RL Confidence',
                         'label_expect': 'LB Expected', 'label_pred': 'LB Predicted', 'label_prob': 'LB Confidence',
                         "tm_expect": 'TM Expected', 'tm_pred': 'TM Predicted'}, inplace=True)

            # results_dash.columns = pd.MultiIndex.from_tuples(
            #     zip(['', 'Activity', '', '', 'Role', '', '', 'Label', '', '', 'Time', ''],
            #         results_dash.columns))

        # st.table(results_dash)
        # AgGrid(results_dash)


    @staticmethod
    @st.cache(persist=True)
    def dashboard_multiprediction_acrl(results_dash, parms):
        multipreddict = DashPredictor.dashboard_multiprediction_columns(parms)

        # --------------------results_dash['ac_pred'] = results_dash.ac_pred.replace(parms['index_ac'])
        for ix in range(len(results_dash['ac_pred'])):
            for jx in range(len(results_dash['ac_pred'][ix])):
                # replacing the value from the parms dictionary
                results_dash['ac_pred'][ix].append(parms['index_ac'][results_dash.ac_pred[ix][jx]])
                # Converting probability into percentage
                results_dash['ac_prob'][ix][jx] = (results_dash['ac_prob'][ix][jx] * 100)
            # poping out the values from the list
            ln = int(len(results_dash['ac_pred'][ix]) / 2)
            # st.session_state['_activity_pred'] = results_dash['ac_pred'][ix][:ln]
            del results_dash['ac_pred'][ix][:ln]
            results_dash[multipreddict["ac_pred"]] = pd.DataFrame(results_dash.ac_pred.tolist(),
                                                                  index=results_dash.index)
            results_dash[multipreddict["ac_prob"]] = pd.DataFrame(results_dash.ac_prob.tolist(),
                                                                  index=results_dash.index)

        # print("Session State At dashboard_multiprediction_acrl After for loop: ", st.session_state)
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


    @staticmethod
    def dashboard_nextprediction_history(results_dash, parms):
        st.expander('Expander')
        with st.expander('Expand'):
            st.write('Juicy deets')


    @staticmethod
    def dashboard_nextprediction_write(results_dash, parms):
        if parms['next_mode'] == 'history_with_next':
            DashPredictor.dashboard_nextprediction_execute_write(results_dash, parms)
        elif parms['next_mode'] == 'next_action':
            DashPredictor.dashboard_nextprediction_evaluate_write(results_dash, parms)
        # st.table(results_dash)
        # For the Execution Mode there is no need for the Expected Behaviour


    @staticmethod
    def dashboard_nextprediction_execute_write(results_dash, parms):
        # When not in SME mode, the historical data has to be the predicted output of the model
        st.subheader('📜 Historical Behaviour')
        with st.expander('ℹ️'):
            st.info("Events which has already been executed")
            # if parms['predchoice'] not in ['SME']:
            #     _hist_columns = ['pos_ac_ss', 'pos_rl_ss', 'pos_lb_ss', 'pos_tm_ss'] #selecting the columns
            #     _hist_predicted_dict = dict([(k, st.session_state[k]) for k in _hist_columns ]) #constructing new dict form sessionstate
            #     #--Manuplation to see the Value properly
            #     _hist_predicted_dict['pos_tm_ss'] = sum(_hist_predicted_dict['pos_tm_ss'], []) #flattening of time
            #     _hist_predicted_dict['pos_tm_ss'] = [DashPredictor.rescale(x, parms, parms['scale_args']) for x in _hist_predicted_dict['pos_tm_ss']] #Normalizing back to original value
            #     _hist_predicted_dict = {k: _hist_predicted_dict[k][1:] for k in _hist_predicted_dict} #removing first item in each key in a dictionary
            #     print("Predicted Disctionary :", _hist_predicted_dict)
            #     _hist_predicted_df = pd.DataFrame.from_dict(_hist_predicted_dict)
            #     # Replacing from Dictionary Values to it's original name
            #     _hist_predicted_df['pos_ac_ss'] = _hist_predicted_df.pos_ac_ss.replace(parms['index_ac'])
            #     _hist_predicted_df['pos_rl_ss'] = _hist_predicted_df.pos_rl_ss.replace(parms['index_rl'])
            #     _hist_predicted_df['pos_lb_ss'] = _hist_predicted_df.pos_lb_ss.replace(parms['index_label'])
            #     _hist_predicted_df.rename(
            #         columns={'pos_ac_ss': 'Activity', 'pos_rl_ss': 'Role', 'pos_lb_ss': 'Label', "pos_tm_ss": 'Time'},
            #         inplace=True)
            #     print("Predicted Dataframe :", _hist_predicted_df)
            #     # _hist_predicted_df = _hist_predicted_df.iloc[1:]
            #     st.dataframe(_hist_predicted_df.iloc[:-1])
            # else:
            results_dash_expected = results_dash[['ac_expect', 'rl_expect', 'label_expect', "tm_expect"]]
            results_dash_expected.rename(
                columns={'ac_expect': 'Activity', 'pos_rl_ss': 'Role', 'pos_lb_ss': 'Label', "pos_tm_ss": 'Time'},
                inplace=True)
            st.dataframe(results_dash_expected.iloc[:-1])
        st.markdown("""---""")
        if parms['variant'] in ['multi_pred']:
            cols = st.columns(parms['multiprednum'])

            multipreddict = DashPredictor.dashboard_multiprediction_columns(parms)

            for kz in range(parms['multiprednum']):
                with cols[kz]:
                    DashPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
            st.markdown("""---""")
            with st.beta_container():
                colstm = st.columns(1)
                with colstm[0]:
                    st.subheader('⌛ Predicted Time Duration')
                    st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Expected'}, inplace=False).iloc[-1:].T,
                             use_column_width=True)


        else:
            st.header("🤔 Max Probability Prediction")
            cols1, cols2, cols3, cols4 = st.columns([2, 2, 2, 1])
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
        print("Session State At The End : ", st.session_state)


    @staticmethod
    def dashboard_nextprediction_evaluate_write(results_dash, parms):
        if parms['variant'] in ['multi_pred']:
            cols = st.columns(parms['multiprednum'] + 1)

            multipreddict = DashPredictor.dashboard_multiprediction_columns(parms)
            for kz in range(parms['multiprednum'] + 1):
                if kz <= (parms['multiprednum'] - 1):
                    with cols[kz]:
                        DashPredictor.dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz)
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
                colstm = st.columns(2)
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
            cols1, cols2, cols3, cols4 = st.columns([2, 2, 1, 2])
            with cols1:
                st.subheader('🏋️ Activity')
                # writes Activity and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(
                    results_dash[["ac_pred", "ac_prob"]].rename(columns={"ac_pred": 'Predicted', "ac_prob": 'Confidence'},
                                                                inplace=False))
            with cols2:
                st.subheader('👨‍💻 Role')
                # writes Role and it's respective confidence on the dashboard with the renamed coulumns name but not modified in the dataframe
                st.write(
                    results_dash[["rl_pred", "rl_prob"]].rename(columns={"rl_pred": 'Predicted', "rl_prob": 'Confidence'},
                                                                inplace=False))
            with cols3:
                st.subheader('⌛ Time')
                st.write(results_dash[["tm_pred"]].rename(columns={"tm_pred": 'Predicted'}, inplace=False))
            with cols4:
                st.subheader('🏷️ Label')
                st.write(results_dash[["label_pred", "label_prob"]].rename(
                    columns={"label_pred": 'Predicted', "label_prob": 'Confidence'}, inplace=False))


    @staticmethod
    def dashboard_nextprediction_write_acrl(results_dash, parms, multipreddict, kz):
        if kz <= parms['multiprednum']:
            if parms['next_mode'] == 'next_action':
                st.header("🤔 Prediction " + str(kz + 1))
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
            elif parms['next_mode'] == 'history_with_next':
                st.header("🤔 Prediction " + str(kz + 1))
                with st.expander('ℹ️'):
                    st.info("Predicted Events")
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