# pylint: disable=no-self-use,invalid-name

import torch
from torch.autograd import Variable
from allennlp.common.testing import ModelTestCase


class OpenIETest(ModelTestCase):
    def setUp(self):
        super(OpenIETest, self).setUp()
        self.set_up_model('tests/fixtures/openie/experiment.json',
                          'tests/fixtures/openie/open_ie.gold_conll')

    def test_coref_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
