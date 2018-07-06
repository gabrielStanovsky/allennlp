from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides
import pandas

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import iob1_to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("openie")
class OpenIEDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD-ID WORD PREDICATE-IND-LIST HEAD-PRED-IND SENT-ID SAMPLE-IND WORD-LABEL

    with a blank line indicating the end of each sample
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``,
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.

    separator : ```str``,
        optional (default = '\t')
        Field separator.
    """
    def __init__(self,
                 separator = "\t",
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.separator = separator

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # Read CSV with pandas with the class separator
        df = pandas.read_csv(file_path,
                             sep = self.separator,
                             header = 0,
                             keep_default_na = False)

        # Split according to instances
        sents = [df[df.run_id == run_id]
                 for run_id
                 in sorted(set(df.run_id.values))]

        #TODO-question: Should this be a generator? It is in conll2013.py
        #TODO: not sure if the casting of head_pred_id to int is necessary
        for sent in sents:
            yield self.text_to_instance(sentence_tokens = sent.word.values,
                                        head_pred_ids = list(map(int,
                                                                 sent.head_pred_id.values)),
                                        open_ie_tags = sent.label.values)

    #TODO-question: Is this a must? Should this override? Why?
    #               It seems to have a different function in sciencesrl_dataset_reader
    #               vs. the conll2013 reader.
    @overrides
    def text_to_instance(self,
                         sentence_tokens: List[str],
                         head_pred_ids: List[int],
                         open_ie_tags: List[str] = None) -> Instance:  # type: ignore
        """
        Create an instance from a tokenized sentence and Open-IE BIO tags
        """
        # pylint: disable=arguments-differ
        tokens = TextField([Token(w)
                            for w in sentence_tokens],
                           token_indexers=self._token_indexers)

        # Construct an instance
        return Instance({'tokens': tokens,
                         'pred_id': SequenceLabelField(labels = head_pred_ids,
                                                       sequence_field = tokens),
                         'tags': SequenceLabelField(labels = open_ie_tags,
                                                    sequence_field = tokens)})

    @classmethod
    def from_params(cls, params: Params) -> 'OpenIEDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        separator = params.pop('separator', '\t')
        params.assert_empty(cls.__name__)
        return OpenIEDatasetReader(token_indexers = token_indexers,
                                   separator = separator)
