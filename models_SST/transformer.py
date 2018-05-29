# --------------------------------------------------------
# --------------------------------------------------------
# code by Shane Steinert-Threlkeld (ILLC, UvA)
# unless otherwise noted
# February 2018
# --------------------------------------------------------
# --------------------------------------------------------

import os
import util
import numpy as np
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

# TODO: modify hparams
# TODO: make ThreeSentence problem


""" How to use TransformerEncoder models!

pip install tensor2tensor

t2t-datagen --t2t_usr_dir /path/to/fill-in-the-quant/models/ \
        --problem txt_class_one_sentence_quantifier \
        --data_dir ../data/

t2t-trainer --t2t_usr_dir /path/to/fill-in-the-quant/models/ \
        --data_dir ../data/ \
        --output_dir /tmp/t2t \
        --problems txt_class_one_sentence_quantifier \
        --model transformer_encoder \
        --hparams_set transformer_tiny \
        --train_steps=1000 \
        --eval_steps=100
"""


def to_int_list(narr):
    return [int(item) for item in list(narr)]


def to_generator(inputs, targets):
    assert len(inputs) == len(targets)
    for idx in range(len(inputs)):
        # 'target' needs to be list of one element, but targets[idx] is a
        # one-hot 1-d nparray, so get the corresponding index out
        label = np.nonzero(targets[idx])[0][0]
        # inputs and targets need to be list of ints, not numpy arrays
        yield {'inputs': to_int_list(inputs[idx]),
               'target': [int(label)]}


def to_native(s):
    return s.decode('latin_1')


def to_one_hot(label, num_classes):
    ls = [0] * num_classes
    ls[label] = 1
    return ls


@registry.register_problem()
class TxtClassOneSentenceQuantifier(problem.Problem):

    DATA_PATH = '../data/'
    SETTING = 'starget'
    SETS = ['train', 'val', 'test']
    LABELS = ['none of ', 'a few of ', 'few of ', 'some of ',
              'many of ', 'most of ', 'more than half of ',
              'almost all of ', 'all of ']
    MAX_WORDS = 50000
    MAX_LEN = 50

    @property
    def batch_size_means_tokens(self):
        return True

    @property
    def vocab_file(self):
        return TxtClassOneSentenceQuantifier.SETTING + '.vocab'

    def generator(self, data_dir, texts, labels):

        # generate vocab
        encoder = generator_utils.get_or_generate_vocab_inner(
            data_dir,
            self.vocab_file,
            TxtClassOneSentenceQuantifier.MAX_WORDS,
            (to_native(text) for text in texts))  # generator

        for idx in range(len(texts)):
            # labels is list of ints
            yield {'inputs': encoder.encode(to_native(texts[idx])),
                   'target': [labels[idx]]}

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        """
        texts, labels, word_index = util.generate_datasets(
            TxtClassOneSentenceQuantifier.DATA_PATH,
            TxtClassOneSentenceQuantifier.SETTING,
            TxtClassOneSentenceQuantifier.SETS,
            TxtClassOneSentenceQuantifier.LABELS,
            TxtClassOneSentenceQuantifier.MAX_WORDS,
            TxtClassOneSentenceQuantifier.MAX_LEN)
        """
        texts, labels = util.read_data_from_txt(
            TxtClassOneSentenceQuantifier.DATA_PATH,
            TxtClassOneSentenceQuantifier.SETTING,
            TxtClassOneSentenceQuantifier.SETS,
            TxtClassOneSentenceQuantifier.LABELS)
        generator_utils.generate_dataset_and_shuffle(
            self.generator(data_dir, texts['train'], labels['train']),
            self.training_filepaths(data_dir, 1, shuffled=False),
            self.generator(data_dir, texts['val'], labels['val']),
            self.dev_filepaths(data_dir, 1, shuffled=False))

    # inspired by sentiment_imdb tensor2tensor example:
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/imdb.py
    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        # TODO: is there a way to get vocab size from generate_data?
        # source_vocab_size = TxtClassOneSentenceQuantifier.MAX_WORDS
        source_vocab_size = self._encoders["inputs"].vocab_size
        p.input_modality = {
            "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
        }
        p.target_modality = (registry.Modalities.CLASS_LABEL,
                             len(TxtClassOneSentenceQuantifier.LABELS))
        p.input_space_id = problem.SpaceID.EN_TOK
        p.target_space_id = problem.SpaceID.GENERIC

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_file)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        return {
            'inputs': encoder,
            'targets':
            text_encoder.ClassLabelEncoder(TxtClassOneSentenceQuantifier.LABELS),
        }


@registry.register_problem()
class TxtClassThreeSentenceQuantifier(problem.Problem):

    DATA_PATH = '../data/'
    SETTING = 's3'
    SETS = ['train', 'val', 'test']
    LABELS = ['none of ', 'a few of ', 'few of ', 'some of ',
              'many of ', 'most of ', 'more than half of ',
              'almost all of ', 'all of ']
    MAX_WORDS = 50000
    MAX_LEN = 150

    @property
    def batch_size_means_tokens(self):
        return True

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        texts, labels, word_index = util.generate_datasets(
            TxtClassOneSentenceQuantifier.DATA_PATH,
            TxtClassOneSentenceQuantifier.SETTING,
            TxtClassOneSentenceQuantifier.SETS,
            TxtClassOneSentenceQuantifier.LABELS,
            TxtClassOneSentenceQuantifier.MAX_WORDS,
            TxtClassOneSentenceQuantifier.MAX_LEN)
        self._vocab_size = word_index
        generator_utils.generate_dataset_and_shuffle(
            to_generator(texts['train'], labels['train']),
            self.training_filepaths(data_dir, 1, shuffled=False),
            to_generator(texts['val'], labels['val']),
            self.dev_filepaths(data_dir, 1, shuffled=False))

    # inspired by sentiment_imdb tensor2tensor example:
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/imdb.py
    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        # TODO: is there a way to get vocab size from generate_data?
        source_vocab_size = TxtClassThreeSentenceQuantifier.MAX_WORDS
        p.input_modality = {
            "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
        }
        p.target_modality = (registry.Modalities.CLASS_LABEL,
                             len(TxtClassThreeSentenceQuantifier.LABELS))
        p.input_space_id = problem.SpaceID.EN_TOK
        p.target_space_id = problem.SpaceID.GENERIC
