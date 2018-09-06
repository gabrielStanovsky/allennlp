import logging
import pdb
from pprint import pprint

from typing import List

from allennlp.common.util import JsonDict, sanitize
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

    @staticmethod
    def join_mwp(tags: List[str]):
        """
        Join multi-word predicates to a single
        predicate ('V') token.
        """
        ret = []
        verb_flag = False
        for tag in tags:
            if "V" in tag:
                # Create a continuous 'V' BIO span
                prefix, suffix = tag.split("-")

                if verb_flag:
                    # Continue a verb label across the different predicate parts
                    prefix = 'I'

                ret.append(f"{prefix}-V")
                verb_flag = True
            else:
                ret.append(tag)
                verb_flag = False

        return ret

    @staticmethod
    def make_oie_string(tokens: List[Token], tags: List[str]):
        """
        Converts a list of model outputs (i.e., a list of lists of bio tags, each
        pertaining to a single word), returns an inline bracket representation of
        the prediction.
        """
        frame = []
        chunk = []

        words = [token.text for token in tokens]

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    frame.append("[" + " ".join(chunk) + "]")
                    chunk = []

                if tag.startswith("B-"):
                    chunk.append(tag[2:] + ": " + token)
                elif tag == "O":
                    frame.append(token)

        if chunk:
            frame.append("[" + " ".join(chunk) + "]")

        return " ".join(frame)

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
        sent_tokens = self._tokenizer.tokenize(inputs["sentence"])
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
        results = {"verbs": [], "words": sent_tokens}

        for tags, pred_id in zip(outputs, pred_ids):
            # Join multi-word predicates
            tags = self.join_mwp(tags)

            # Create description text
            description = self.make_oie_string(sent_tokens, tags)

            # Add a predicate prediction to the return dictionary.
            results["verbs"].append({
                    "verb": sent_tokens[pred_id].text,
                    "description": description,
                    "tags": tags,
            })

        return sanitize(results)

