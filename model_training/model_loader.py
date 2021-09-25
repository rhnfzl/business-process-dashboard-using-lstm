# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:53:54 2020

@author: Manuel Camargo
"""
import tensorflow as tf


from model_training.models import model_concatenated_base as mcatnlbbs
from model_training.models import model_shared_cat_base as mshcatbs

from model_training.models import model_concatenated_inter as mcatnlb
from model_training.models import model_shared_cat_inter as mshcat

class ModelLoader():

    def __init__(self, parms):
        self.parms = parms
        self._trainers = dict()
        self.trainer_dispatcher = {'concatenated_base': mcatnlbbs._training_model,
                                   'shared_cat_base': mshcatbs._training_model,
                                   'concatenated_inter': mcatnlb._training_model,
                                   'shared_cat_inter': mshcat._training_model}

    def train(self, model_type, examples, ac_weights, rl_weights, output_folder):
        loader = self._get_trainer(model_type)
        tf.compat.v1.reset_default_graph()
        loader(examples, ac_weights, rl_weights, output_folder, self.parms)

    def register_model(self, model_type, trainer):
        try:
            self._trainers[model_type] = self.trainer_dispatcher[trainer]
        except KeyError:
            raise ValueError(trainer)

    def _get_trainer(self, model_type):
        trainer = self._trainers.get(model_type)
        if not trainer:
            raise ValueError(model_type)
        return trainer