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
                tag_label = ''.join(t[t.index('-') + 1:])
                if tag_label not in tag_spans:
                    tag_spans[tag_label] = tokens[i]
                else:
                    tag_spans[tag_label] += " " + tokens[i]
        all_tag_spans.append(tag_spans)
    return all_tag_spans


#            {
#               nodeType: 'entity',
#               word: 'Sam',
#               link: 'subject',
#               spans: [
#                 {
#                   start: 0,
#                   end: 3
#                 }
#               ]
#             }
def create_node(nodeType, word, span_start, span_end, link):
    d = dict()
    d["nodeType"] = nodeType
    d["word"] = word
    if link:
        d["link"] = link
    d["spans"] = [{"start": span_start, "end": span_end}]
    return d


# {
#         text: 'Sam likes bananas',
#         root: {
#           nodeType: 'event',
#           word: 'like',
#           spans: [
#             {
#               start: 4,
#               end: 9
#             }
#           ],
#           children: [
#             {
#               nodeType: 'entity',
#               word: 'Sam',
#               link: 'subject',
#               spans: [
#                 {
#                   start: 0,
#                   end: 3
#                 }
#               ]
#             },
#             {
#               nodeType: 'entity',
#               word: 'banana',
#               link: 'object',
#               attributes: [ '>1'],
#               spans: [
#                 {
#                   start: 10,
#                   end: 17
#                 }
#               ]
#             }
#           ]
#         }
def hierplane_input_from(spans, tokens, top_level_link=''):
    d = dict()
    sentence = " ".join(tokens)
    root_word = spans.get('P', "")
    # Multiple predicates indicated by top_level_link not being blank
    # For each such predicate, d[text] is the P value (i.e. a pattern)
    d["text"] = sentence if (not top_level_link or len(top_level_link) == 0) else "_"+root_word
    root_node = create_node(
        nodeType='event',
        word=root_word,  # AO => value
        span_start=sentence.index(root_word) if root_word in sentence else 0,
        span_end=len(root_word),
        link=top_level_link
    )
    children = []
    for o in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']:
        if o in spans:
            children.append(
                create_node(
                    nodeType='entity',
                    word=spans[o],
                    span_start=sentence.index(spans[o]) if spans[o] in sentence else 0,
                    span_end=len(spans[o]),
                    link=o
                )
            )
    root_node['children'] = children
    d['root'] = root_node
    return d


# github.com/allenai/hierplane#web
# example of a tree: https://github.com/allenai/allennlp/blob/5352a198e1831e63c76ee8e51694c1bf6b63b1b1/allennlp/tests/predictors/constituency_parser_test.py
# for css selectors: https://www.w3schools.com/cssref/trysel.asp
def hierplane_merged_inputs_from(spans_all, tokens):
    d = dict()
    sentence = " ".join(tokens)
    d["text"] = sentence
    root_word = sentence
    root_node = create_node(
        nodeType='event',
        word=root_word,  # AO => value
        span_start=sentence.index(root_word) if root_word in sentence else 0,
        span_end=len(root_word),
        link=''
    )
    children = []
    for tree_number, spans in enumerate(spans_all):
        temp_dict = hierplane_input_from(spans, tokens, 'Extraction_' + str(tree_number + 1))
        children.append(temp_dict['root'])
    root_node['children'] = children
    d['root'] = root_node
    print(f"\nDebug: the hierplane tree is: ")
    pprint(d)
    return d


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

        tokens = [Token(t) for t in sentence_token_text]
        instances = [self._dataset_reader.text_to_instance(tokens,
                                                           [1 if (word_ind ==h) \
                                                            else 0
                                                            for word_ind
                                                            in range(len(tokens))])
                     for h in head_pred_ids_all]

        try:
            outputs = [self._model.forward_on_instance(instance,
                                                       cuda_device)["tags"]
                       for instance in instances]
        except:
            # We do not want notorious GUI sentences to break the system.
            logger.error(f"Exception in OpenIE input: {inputs}")
            outputs = None

        # The purpose of json_outputs is to wrap any other field necessary for the demo
        # e.g., spo format instead of BIO like tags.
        json_outputs = {"outputs": outputs}

        json_outputs["tag_spans"] = gen_tag_spans(sentence_token_text, outputs)
        pdb.set_trace()
        json_outputs["tokens"] = sentence_token_text
        json_outputs["hierplane_inputs"] = [
            hierplane_input_from(spans, json_outputs["tokens"])
            for spans in json_outputs["tag_spans"]
        ]
        json_outputs["hierplane_inputs_merged"] = hierplane_merged_inputs_from(
            json_outputs["tag_spans"], json_outputs["tokens"])

        print(f"OpenIE predictor\t{len(instances)} instances:\ninput = {inputs}\noutput =\n{json_outputs}")
        return {**inputs, **json_outputs}
