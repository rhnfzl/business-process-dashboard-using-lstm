# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:24:38 2020

@author: Manuel Camargo
"""
from model_prediction import next_event_samples_creator as nesc
from model_prediction import suffix_samples_creator as ssc


from model_prediction import next_event_predictor_batch_mode as nep
from model_prediction import suffix_predictor as sp
from model_prediction import event_log_predictor as elp
from model_prediction import next_event_predictor_single_mode_evaluation as nepsmeva #Evalaution
from model_prediction import next_event_predictor_single_mode_execution as nepsmexe #Executuon
from model_prediction import next_event_predictor_single_mode_whatif as nepsmwi #What-if


class SamplesCreator:
    def create(self, predictor, activity):
        sampler = self._get_samples_creator(activity)
        predictor.sampling(sampler)

    def _get_samples_creator(self, activity):
        if activity == 'predict_next':
            return nesc.NextEventSamplesCreator()
        elif activity == 'pred_sfx':
            return ssc.SuffixSamplesCreator()
        else:
            raise ValueError(activity)


class PredictionTasksExecutioner:
    def predict(self, predictor, activity, mode, next_mode):
        executioner = self._get_predictor(activity, mode, next_mode)
        predictor.predict(executioner, mode)

    def _get_predictor(self, activity, mode, next_mode):
        if activity == 'predict_next':
            if mode == 'next':
                if next_mode == 'history_with_next':
                    return nepsmexe.NextEventPredictor()
                elif next_mode == 'next_action':
                    return nepsmeva.NextEventPredictor()
                elif next_mode == 'what_if':
                    return nepsmwi.NextEventPredictor()
            elif mode == 'batch':
                return nep.NextEventPredictor()
        elif activity == 'pred_sfx':
            return sp.SuffixPredictor()
        elif activity == 'pred_log':
            return elp.EventLogPredictor()
        else:
            raise ValueError(activity)
