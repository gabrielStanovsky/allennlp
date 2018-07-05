# pylint: disable=no-self-use,invalid-name
import pytest
import pdb

from allennlp.data.dataset_readers.open_ie import OpenIEDatasetReader
from allennlp.common.util import ensure_list

class TestOpenIEDatasetReader:
    def test_read_from_file(self):
        open_ie_reader = OpenIEDatasetReader()
        instances = open_ie_reader.read('tests/fixtures/data/open_ie.txt')
        instances = ensure_list(instances)
        assert len(instances) == 2


        fields = instances[0].fields
        tokens = [t.text for t in fields['tokens'].tokens]

        # Check tokens of the first sentence
        assert tokens == ["QVC", "Network", "Inc.", "said", "it", "completed", "its", "acquisition", \
                          "of", "CVN", "Cos.", "for","about", "$" , "423", "million", "."]

        # Check the tags of the second sentence
        assert list(instances[1].fields["tags"].labels) == ["A1-B","A1-I", "A1-I", "O", "O", "P-B", "A0-B", "A0-I",
                                                            "A0-I", "A0-I", "A0-I", "O", "A2-B", "A2-I", "A2-I", "A2-I", "O"]


        # fields = instances[1].fields
        # tokens = [t.text for t in fields['tokens'].tokens]
        # assert tokens == ['AI2', 'engineer', 'Joel', 'lives', 'in', 'Seattle', '.']
        # assert fields["tags"].labels == expected_labels
