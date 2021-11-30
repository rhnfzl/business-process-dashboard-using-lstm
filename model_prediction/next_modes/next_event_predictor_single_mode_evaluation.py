# -*- coding: utf-8 -*-
"""
@author: Rehan Fazal
"""
import numpy as np
import pandas as pd
# from numpy.core._multiarray_umath import ndarray

from support_modules import support as sup

class NextEventPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = model
        #print("spl :", spl)
        self.spl = spl
        self.imp = imp
        if params['mode'] == 'next':
            fltr_idx = params['nextcaseid_attr']["filter_index"]
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

        pred_fltr_idx = parameters['nextcaseid_attr']["filter_index"] + 1

        results = list()

        for i, enm in enumerate(self.spl['prefixes']['activities'][pred_fltr_idx:]):
            if i == 0:
            # if pred_fltr_idx == len(enm):
                preds_prefix = list()

                serie_predict_ac = [self.spl['prefixes']['activities'][pred_fltr_idx:][i][:idx]
                    for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start

                y_serie_predict_ac = [x[-1] for x in
                                      serie_predict_ac]  # selecting the last value from each list of list

                serie_predict_ac = serie_predict_ac[:-1]  # to avoid end value that is max value
                y_serie_predict_ac = y_serie_predict_ac[1:]  # to avoid start value i.e 0

                serie_predict_rl = [self.spl['prefixes']['roles'][pred_fltr_idx:][i][:idx]
                    for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start

                y_serie_predict_rl = [x[-1] for x in
                                      serie_predict_rl]  # selecting the last value from each list of list

                serie_predict_rl = serie_predict_rl[:-1]  # to avoid end value that is max value
                y_serie_predict_rl = y_serie_predict_rl[1:]  # to avoid start value i.e 0

                serie_predict_tm = [np.array(self.spl['prefixes']['times'][pred_fltr_idx:][i][:idx])
                                    for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start

                y_serie_predict_tm = [x[-1] for x in
                                      serie_predict_tm]  # selecting the last value from each list of list

                serie_predict_tm = serie_predict_tm[:-1]  # to avoid end value that is max value
                y_serie_predict_tm = y_serie_predict_tm[1:]  # to avoid start value i.e 0

                if vectorizer in ['basic']:

                    serie_predict_intr = []
                    y_serie_predict_intr = []

                    conf_results = self._predict_next_event_confermance_checking(parameters, serie_predict_ac, serie_predict_rl,
                                                                  serie_predict_tm, serie_predict_intr, vectorizer,
                                                                  y_serie_predict_ac, y_serie_predict_rl,
                                                                  y_serie_predict_tm, y_serie_predict_intr)

                elif vectorizer in ['inter']:
                    serie_predict_intr = [np.array(self.spl['prefixes']['inter_attr'][pred_fltr_idx:][i][:idx])
                                        for idx in range(1, pred_fltr_idx + 2)]  # range starts with 1 to avoid start

                    y_serie_predict_intr = [x[-1] for x in
                                          serie_predict_intr]  # selecting the last value from each list of list


                    serie_predict_intr = serie_predict_intr[:-1]  # to avoid end value that is max value
                    y_serie_predict_intr = y_serie_predict_intr[1:]  # to avoid start value i.e 0

                    conf_results = self._predict_next_event_confermance_checking(parameters, serie_predict_ac, serie_predict_rl,
                                                                  serie_predict_tm, serie_predict_intr, vectorizer,
                                                                  y_serie_predict_ac, y_serie_predict_rl,
                                                                  y_serie_predict_tm, y_serie_predict_intr)

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
                # intercase features if necessary
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
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

                if self.imp == 'arg_max':
                    # Use this to get the max prediction

                    pos = np.argmax(preds[0][0])
                    pos_prob = preds[0][0][pos]

                    pos1 = np.argmax(preds[1][0])
                    pos1_prob = preds[1][0][pos1]

                    predictions = [[pos] + [pos], [pos1] + [pos1],
                                   [abs(preds[2][0][0])] * (parameters['multiprednum'] + 1), [pos_prob] + [pos_prob],
                                   [pos1_prob] + [pos1_prob]]

                    preds_prefix.append([pos, pos1, abs(preds[2][0][0])])

                elif self.imp == 'multi_pred':
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

                    predictions = [[pos[0]] + pos, [pos1[0]] + pos1,
                                   [abs(preds[2][0][0])] * (parameters['multiprednum'] + 1), [pos_prob[0]] + pos_prob,
                                   [pos1_prob[0]] + pos1_prob]

                    preds_prefix.append([pos, pos1, [abs(preds[2][0][0])] * parameters['multiprednum']])

                    # print(predictions)

                #-------
                _pos = pos
                _pos1 = pos1
                _preds = abs(preds[2][0][0])

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.__create_result_record_next(i, self.spl, predictions, parameters))

                # print("prefix : ", preds_prefix, i)

            else:

                _ac = self.spl['prefixes']['activities'][pred_fltr_idx:][i] #--Activity
                _rl = self.spl['prefixes']['roles'][pred_fltr_idx:][i] #--Role
                _tm = self.spl['prefixes']['times'][pred_fltr_idx:][i] #--Time

                for lk in range(parameters['multiprednum']):

                    _temp_ac = list()
                    _temp_rl = list()
                    _temp_tm = list()

                    if self.imp == 'multi_pred':

                        for gk in range(len(preds_prefix)):
                            _temp_ac.append(float(preds_prefix[gk][0][lk]))
                            _temp_rl.append(float(preds_prefix[gk][1][lk]))
                            _temp_tm.append(float(preds_prefix[gk][2][lk]))

                    elif self.imp == 'arg_max':

                        for gk in range(len(preds_prefix)):
                            _temp_ac.append(float(preds_prefix[gk][0]))
                            _temp_rl.append(float(preds_prefix[gk][1]))
                            _temp_tm.append(float(preds_prefix[gk][2]))

                    _temp_ac = np.array(_ac[:pred_fltr_idx+1] + _temp_ac)
                    _temp_rl = np.array(_rl[:pred_fltr_idx+1] + _temp_rl)
                    _temp_tm = np.concatenate((_tm[:pred_fltr_idx+1], np.dstack([_temp_tm])[0]), axis=0)

                    x_ac_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_temp_ac),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))
                    x_rl_ngram = (np.append(
                        np.zeros(parameters['dim']['time_dim']),
                        np.array(_temp_rl),
                        axis=0)[-parameters['dim']['time_dim']:]
                                  .reshape((1, parameters['dim']['time_dim'])))
                    # times input shape(1,5,1)
                    times_attr_num = (_temp_tm.shape[1])
                    x_t_ngram = np.array(
                        [np.append(np.zeros(
                            (parameters['dim']['time_dim'], times_attr_num)),
                            _temp_tm, axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], times_attr_num))]
                    )
                    if vectorizer in ['basic']:
                        inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                    elif vectorizer in ['inter']:
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

                    if self.imp == 'multi_pred':
                        if lk == 0:
                            _arr_pos = []
                            _arr_pos_prob = []
                            _arr_pos1 = []
                            _arr_pos1_prob = []
                            _arr_pred = []

                        # Use this to get the max prediction in multipred
                        _arr_pos.append(np.argmax(preds[0][0]))
                        _arr_pos_prob.append(preds[0][0][_arr_pos[lk]])

                        _arr_pos1.append(np.argmax(preds[1][0]))
                        _arr_pos1_prob.append(preds[1][0][_arr_pos1[lk]])

                        _arr_pred.append(abs(preds[2][0][0]))
                    elif self.imp == 'arg_max':
                        _arr_pos = np.argmax(preds[0][0])
                        _arr_pos_prob = preds[0][0][_arr_pos]

                        _arr_pos1 = np.argmax(preds[1][0])
                        _arr_pos1_prob = preds[1][0][_arr_pos1]

                        # _arr_pred = preds[2][0][0]

                _pos = _arr_pos
                _pos1 = _arr_pos1
                if self.imp == 'multi_pred':
                    # _preds = sum(_arr_pred)/len(_arr_pred)
                    _preds = _arr_pred
                elif self.imp == 'arg_max':
                    _preds = abs(preds[2][0][0])

                preds_prefix.append([_pos, _pos1, _preds])

                #--SME Input Logic
                if vectorizer in ['basic']:

                    inter_case_vector = []

                    predsmeac, predsmeac_prob, predsmerl, predsmerl_prob, predsmetm = self._sme_predict_next_event_suffix_cat_next(parameters,
                                                                                                                                   self.spl['prefixes']['activities'][pred_fltr_idx:][i],
                                                                                                                                   self.spl['prefixes']['roles'][pred_fltr_idx:][i],
                                                                                                                                   self.spl['prefixes']['times'][pred_fltr_idx:][i],
                                                                                                                                   inter_case_vector,
                                                                                                                                   vectorizer)
                elif vectorizer in ['inter']:
                    predsmeac, predsmeac_prob, predsmerl, predsmerl_prob, predsmetm = self._sme_predict_next_event_suffix_cat_next(parameters,
                                                                                                                                   self.spl['prefixes']['activities'][pred_fltr_idx:][i],
                                                                                                                                   self.spl['prefixes']['roles'][pred_fltr_idx:][i],
                                                                                                                                   self.spl['prefixes']['times'][pred_fltr_idx:][i],
                                                                                                                                   self.spl['prefixes']['inter_attr'][pred_fltr_idx:][i],
                                                                                                                                   vectorizer)

                if self.imp == 'multi_pred':
                    _acfinal = [predsmeac] + _pos
                    _acprobfinal = [predsmeac_prob] + _arr_pos_prob
                    _rlfinal = [predsmerl] + _pos1
                    _rlprobfinal = [predsmerl_prob] + _arr_pos1_prob
                    _tmprobfinal = [predsmetm] + _preds
                elif self.imp == 'arg_max':
                    _acfinal = [predsmeac] + [_pos]
                    _acprobfinal = [predsmeac_prob] + [_arr_pos_prob]
                    _rlfinal = [predsmerl] + [_pos1]
                    _rlprobfinal = [predsmerl_prob] + [_arr_pos1_prob]
                    _tmprobfinal = [predsmetm] + [_preds]

                predictions = [_acfinal, _rlfinal, _tmprobfinal, _acprobfinal, _rlprobfinal]
                # print("Later Prediction Structure : ", i+1, " : ",predictions)
                #
                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(self.__create_result_record_next(i, self.spl, predictions, parameters))

        sup.print_done_task()

        return results, conf_results

    def __create_result_record_next(self, index, spl, preds, parms):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()
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

            record['tm_pred'] = [self.rescale(x, parms, parms['scale_args'])
                                 for x in preds[2]]

        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur'])
                for x in spl['prefixes']['times'][_fltr_idx:][index]]
            record['dur_expect'] = self.rescale(
                spl['next_evt']['times'][_fltr_idx:][index][0], parms,
                parms['scale_args']['dur'])

            record['dur_pred'] = [self.rescale(x, parms, parms['scale_args'])
                                 for x in preds[2]]
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

    def _sme_predict_next_event_suffix_cat_next(self, parameters, _smeac, _smerl, _smetm, _inter, vectorizer):
        # Activities and roles input shape(1,5)
        x_ac_ngram = (np.append(
            np.zeros(parameters['dim']['time_dim']),
            np.array(_smeac),
            axis=0)[-parameters['dim']['time_dim']:]
                      .reshape((1, parameters['dim']['time_dim'])))
        x_rl_ngram = (np.append(
            np.zeros(parameters['dim']['time_dim']),
            np.array(_smerl),
            axis=0)[-parameters['dim']['time_dim']:]
                      .reshape((1, parameters['dim']['time_dim'])))
        # times input shape(1,5,1)
        times_attr_num = (_smetm.shape[1])
        x_t_ngram = np.array(
            [np.append(np.zeros(
                (parameters['dim']['time_dim'], times_attr_num)),
                _smetm, axis=0)
             [-parameters['dim']['time_dim']:]
                 .reshape((parameters['dim']['time_dim'], times_attr_num))]
        )
        if vectorizer in ['basic']:
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
        elif vectorizer in ['inter']:
            inter_attr_num = (_inter.shape[1])
            x_inter_ngram = np.array(
                [np.append(np.zeros((
                    parameters['dim']['time_dim'], inter_attr_num)),
                    _inter, axis=0)
                 [-parameters['dim']['time_dim']:]
                     .reshape((parameters['dim']['time_dim'], inter_attr_num))]
            )
            inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
        # predict
        preds = self.model.predict(inputs)

        predsmeac = np.argmax(preds[0][0])
        predsmeac_prob = preds[0][0][predsmeac]

        predsmerl = np.argmax(preds[1][0])
        predsmerl_prob = preds[1][0][predsmerl]

        predsmetm = abs(preds[2][0][0])

        return predsmeac, predsmeac_prob, predsmerl, predsmerl_prob, predsmetm



    def _predict_next_event_confermance_checking(self, parameters, serie_predict_ac, serie_predict_rl, serie_predict_tm, serie_predict_intr, vectorizer, y_serie_predict_ac, y_serie_predict_rl, y_serie_predict_tm, y_serie_predict_intr):

        results = list()

        for i, _ in enumerate(serie_predict_ac):

                x_ac_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(serie_predict_ac[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))
                x_rl_ngram = (np.append(
                    np.zeros(parameters['dim']['time_dim']),
                    np.array(serie_predict_rl[i]),
                    axis=0)[-parameters['dim']['time_dim']:]
                              .reshape((1, parameters['dim']['time_dim'])))

                times_attr_num = (serie_predict_tm[i].shape[1])
                x_t_ngram = np.array(
                    [np.append(np.zeros(
                        (parameters['dim']['time_dim'], times_attr_num)),
                        serie_predict_tm[i], axis=0)
                     [-parameters['dim']['time_dim']:]
                         .reshape((parameters['dim']['time_dim'], times_attr_num))]
                )
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    inter_attr_num = (serie_predict_intr[i].shape[1])
                    x_inter_ngram = np.array(
                        [np.append(np.zeros((
                            parameters['dim']['time_dim'], inter_attr_num)),
                            serie_predict_intr[i], axis=0)
                         [-parameters['dim']['time_dim']:]
                             .reshape((parameters['dim']['time_dim'], inter_attr_num))]
                    )
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                # predict
                preds = self.model.predict(inputs)

                if self.imp == 'arg_max':

                    pos = np.argmax(preds[0][0])
                    pos_prob = preds[0][0][pos]

                    pos1 = np.argmax(preds[1][0])
                    pos1_prob = preds[1][0][pos1]

                    predictions = [[pos], [pos1], [abs(preds[2][0][0])], [pos_prob], [pos1_prob]]

                elif self.imp == 'multi_pred':

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

                    predictions = [pos, pos1, [abs(preds[2][0][0])], pos_prob, pos1_prob]

                if not parameters['one_timestamp']:
                    predictions.extend([preds[2][0][1]])
                results.append(
                    self._conf_create_result_record_next(i, serie_predict_ac, serie_predict_rl, serie_predict_tm,
                                                         predictions, parameters, y_serie_predict_ac,
                                                         y_serie_predict_rl, y_serie_predict_tm))

        return results


    def _conf_create_result_record_next(self, index, serie_predict_ac, serie_predict_rl, serie_predict_tm, preds, parms, y_serie_predict_ac, y_serie_predict_rl, y_serie_predict_tm):
        _fltr_idx = parms['nextcaseid_attr']["filter_index"] + 1
        record = dict()

        record['conf_ac_prefix'] = serie_predict_ac[index]
        record['conf_ac_expect'] = y_serie_predict_ac[index]
        record['conf_ac_pred'] = preds[0]
        record['conf_ac_prob'] = preds[3]
        record['conf_rl_prefix'] = serie_predict_rl[index]
        record['conf_rl_expect'] = y_serie_predict_rl[index]
        record['conf_rl_pred'] = preds[1]
        record['conf_rl_prob'] = preds[4]

        if parms['one_timestamp']:
            record['conf_tm_prefix'] = [self.rescale(
               x, parms, parms['scale_args'])
               for x in serie_predict_tm[index]]
            record['conf_tm_expect'] = self.rescale(
                y_serie_predict_tm[index][0],
                parms, parms['scale_args'])

            record['conf_tm_pred'] = [self.rescale(x, parms, parms['scale_args'])
                                 for x in preds[2]]

        return record