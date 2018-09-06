import logging
import pdb
from pprint import pprint

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from typing import Tuple
from allennlp.data.tokenizers import Token
from collections import defaultdict

logger = logging.getLogger(__name__)

@Predictor.register('openie_predictor')
class OpenIePredictor(Predictor):
    """
    Predictor for the :class: `models.OpenIEModel` model.
    Used by online demo and for prediction on an input file using command line.
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader
                 ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))

    def _generate_tag_spans(self, tokens, sentence_bio_tags):
        """
        Converts a list of model outputs (i.e., a list of lists of bio tags, each
        pertaining to a single predicate), return a list of dictionaries,
        representing the linearized form of each of the corresponding predicate roles:
        [{"ARG0": "...",
          "V": "...",
          "ARG1": "...",
          ...}]
        """
        sentence_tag_spans = []
        for bio_tags in sentence_bio_tags:
            pred_tag_spans = defaultdict(lambda: '')
            for word_index, tag in enumerate(bio_tags):
                # Strip BIO to get just the tag label
                tag_label = tag.split('-')[-1]
                if tag_label != 'O':
                    if tag_label.endswith('V'):
                        tag_label = 'V'

                    # Concatenate this word to its label
                    pred_tag_spans[tag_label] += ' ' + tokens[word_index]
            sentence_tag_spans.append(pred_tag_spans)
        return sentence_tag_spans

    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"sentence": "...", "predicate_index": "..."}``.
        Assumes sentence is tokenized, and that predicate_index points to a specific
        predicate (word index) within the sentence, for which to produce Open IE extractions.
        """
        tokens = json_dict["sentence"]
        predicate_index = int(json_dict["predicate_index"])
        verb_labels = [0 for _ in tokens]
        verb_labels[predicate_index] = 1
        return self._dataset_reader.text_to_instance(tokens, verb_labels)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        """
        Create instance(s) after predicting the format. One sentence containing multiple verbs 
        will lead to multiple instances.

        Expects JSON that looks like ``{"sentence": "..."}``

        Returns a JSON that looks like ``{"tokens": [...],
                                          "tag_spans": [{"ARG0": "...",
                                                         "V": "...",
                                                         "ARG1": "...",
                                                         ...}]}
        """
        sent_tokens = self._tokenizer.tokenize(inputs["sentence_tokens"])
        sentence_token_text = [t.text for t in sent_tokens]

        # Find all verbs in the input sentence
        pred_ids = [i for (i, t)
                             in enumerate(sent_tokens)
                             if t.pos_ == "VERB"]

        # Create instances
        instances = [self._json_to_instance({"sentence": sent_tokens,
                                             "predicate_index": pred_id})
                     for pred_id in pred_ids]

        # Run model
        try:
            outputs = [self._model.forward_on_instance(instance)["tags"]
                       for instance in instances]
        except:
            # We do not want notorious GUI sentences to break the system.
            logger.error(f"Exception in OpenIE input: {inputs}")
            outputs = None

        # Build and return output dictionary
        return {"tokens": sentence_token_text,
                "tag_spans": self._generate_tag_spans(sentence_token_text, outputs)}
