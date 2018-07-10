import logging
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from overrides import overrides
from typing import Tuple

### for comparison's sake
from allennlp.service.predictors.prop_extraction import prop_extraction
###

logger = logging.getLogger(__name__)


def gen_tag_spans(tokens, all_bio_tags):
    # [
    #   "B-A0",
    #   "B-A0",
    #   "B-A0",
    #   "B-A0",
    #   "I-A0",
    #   "B-A0",
    #   "B-A0",
    #   "B-A0",
    #   "B-A0",
    #   "B-A0"
    # ]
    # BIO tags for A0-A5,P,O
    all_tag_spans = []
    for bio_tags in all_bio_tags:
        tag_spans = dict()
        for i, t in enumerate(bio_tags):
            if '-' in t:
                tag_label = ''.join(t[t.index('-')+1:])
                if tag_label not in tag_spans:
                    tag_spans[tag_label] = tokens[i]
                else:
                    tag_spans[tag_label] += " " + tokens[i]
        all_tag_spans.append(tag_spans)
    return all_tag_spans


@Predictor.register('openie_predictor')
class OpenIEPredictor(Predictor):
    """
    Predictor for the `models.OpenIEModel` model.
    Used by online demo and for prediction on an input file using command line 
    # demo command: 
    python -m allennlp.service.server_simple 
            --archive-path  model.tar.gz 
            --predictor openie_predictor 
            --include-package processes 
            --port 8008  
            --static-dir demo/openie/        
    """

    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader
                 ) -> None:
        super().__init__(model, dataset_reader)
        self.tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))
        self.pe = prop_extraction(include_id = False)

    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        pass

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        """
        Create instance(s) after predicting the format. One sentence containing multiple v verbs 
        will lead to multiple v instances. This function gets the model prediction for the instance(s).
        :param inputs:
            sentence_tokens: List[str] (cannot be empty or a sentence or CoNLL format). 
                When this is not in CoNLL format, we use SpaCy tokenizer.
            head_pred_ids: List[int] (can be empty). 
                When this is empty, we use SpaCy POS tagger.
        :param cuda_device:
            by default, no gpu.
        :return:
            model output in a JSON dictionary, that also indicates outputs.
        """

        # Non-empty sentence text-tokens can be assumed as in "conll format".
        in_conll_format = "sentence_tokens" in inputs and "head_pred_ids" in inputs and inputs["sentence_tokens"]

        if not in_conll_format:
            sent_tokens = self.tokenizer.tokenize(inputs["sentence_tokens"])

        sentence_token_text = inputs.get("sentence_tokens") \
                              if in_conll_format \
                                 else [t.text for t in sent_tokens]

        print(f"sentence_token_text = {sentence_token_text}")

        head_pred_ids_all = inputs.get("head_pred_ids",
                                       [i for (i, t)
                                        in enumerate(sent_tokens)
                                        if t.pos_ == "VERB"])

        instances = [self._dataset_reader.text_to_instance(sentence_tokens = sentence_token_text,
                                                           head_pred_ids=[h] * len(sentence_token_text))
                     for h in head_pred_ids_all]

        try:
            # forward returns: {"logits": logits, "mask": mask, "tags": predicted_tags}
            outputs = [self._model.forward_on_instance(instance, cuda_device)["tags"] for instance in instances]
        except:
            # We do not want notorious GUI sentences to break the system.
            logger.error(f"Exception in OpenIE input: {inputs}")
            outputs = None

        # The purpose of json_outputs is to wrap any other field necessary for the demo
        # e.g., spo format instead of BIO like tags.
        json_outputs = {"outputs": outputs}

        # For comparison's sake:
        extractions = self.pe.get_extractions(' '.join(sentence_token_text))
        json_outputs["tag_spans"] = [dict([("P", ex.template)] + \
                                          [("A{}".format(arg_id), val)
                                           for (arg_id, val) in ex.roles_dict.items()])
                                     for ex in extractions] #gen_tag_spans(sentence_token_text, outputs)

        json_outputs["tokens"] = sentence_token_text

        print(f"OpenIE predictor\t{len(instances)} instances:\ninput = {inputs}\noutput =\n{json_outputs}")
        return {**inputs, **json_outputs}
