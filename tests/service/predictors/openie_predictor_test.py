# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from processes.models.propara_baseline_model import ProParaBaselineModel
from processes.service.predictors.propara_prediction import ProParaPredictor


class TestOpenIEPredictor(TestCase):

    def test_sentence_inputs_with_multiple_predicates(self):
        # sentence_tokens
        inputs = {"sentence_tokens": "Angela Merkel met and spoke to her EU counterparts during the climate submit in Paris."}
        archive = load_archive('tests/fixtures/openie_models/toy_model.tar.gz')
        predictor = Predictor.from_archive(archive, 'openie_predictor')
        result = predictor.predict_json(inputs)
        # one each for met and spoke.
        # FIXME add an example from dev set to match performance with non-AllenNLP implementation.
        assert(len(result['outputs'] == 2))
