# pylint: disable=no-self-use,invalid-name
from unittest import TestCase

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor


class TestOpenIEPredictor(TestCase):

    def test_sentence_inputs_with_multiple_predicates(self):
        # sentence_tokens
        inputs = {"sentence_tokens": "Angela Merkel met and spoke to her EU counterparts during the climate submit in Paris ."}
        # archive = load_archive('tests/fixtures/openie/toy_model.tar.gz')
        archive = load_archive('/Users/nikett/Downloads/model_openie_temp.tar.gz')
        predictor = Predictor.from_archive(archive, 'openie_predictor')
        result = predictor.predict_json(inputs)
        # FIXME add an example from dev set to match performance
        # with non-AllenNLP implementation.
        # one each for met and spoke.
        assert(len(result['outputs']) == 2)
        assert len(result['outputs'][0]) == len(inputs['sentence_tokens'].split(' '))
        assert len(result['outputs'][1]) == len(inputs['sentence_tokens'].split(' '))
        assert result['tokens'] == inputs['sentence_tokens'].split(" ")
        assert len(result['hierplane_inputs']) == len(result['outputs'])
        assert len(result['hierplane_inputs_merged']) > 0
        # the current model is not trained enough to get this output.
        # assert result['tag_spans'][0]['A0'] == inputs['Angela Merkel']

