# pylint: disable=no-self-use,invalid-name
import pytest
import pdb

from allennlp.data.dataset_readers.open_ie import OpenIEDatasetReader
from allennlp.common.util import ensure_list
from allennlp.data.dataset_readers.semantic_role_labeling import SrlReader
from allennlp.common.util import ensure_list



def get_verb_indicators(instance):
    """
    Get all indices where verb indicator
    is positive in the given instance.
    """
    return [word_ind
            for (word_ind, label)
            in enumerate(instance.fields["verb_indicator"].labels)
            if label == 1]


class TestOpenIEDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = SrlReader(lazy=False)

        instances = conll_reader.read('tests/fixtures/openie/')
        instances = ensure_list(instances)

        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]

        # Verify tokens and labels
        assert tokens == ["These", "events", ",", "as", "well", "as", "her", \
                          "sense", "that", "she", "destroyed", "the", "bond", \
                          "between", "Dick", "and", "Bruce", ",", "caused", "Barbara", "'s",\
                          "relationship", "with", "Dick", "to", "disintegrate", "and", "eventually", \
                          "led", "her", "to", "marry", "Sam", "Young", "."]

        assert fields["tags"].labels == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', \
                                         'B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', \
                                         'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O', 'O', 'O', \
                                         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', \
                                         'O', 'O', 'O', 'O']

        # Verify verb indicators
        assert get_verb_indicators(instances[0]) == [10]
        assert get_verb_indicators(instances[1]) == [18]
        assert get_verb_indicators(instances[2]) == [28]

        # Multi-word predicates
        assert get_verb_indicators(instances[3]) == [30, 31]
