# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '3/5/19'

import os
import pandas as pd
from bert.run_classifier import DataProcessor
from bert import tokenization
from bert.extract_features import InputExample

class SelfProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            print(split_line)
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = tokenization.convert_to_unicode(split_line[2])
            label = split_line[3]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'val.csv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = tokenization.convert_to_unicode(split_line[2])
            label = split_line[3]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir, 'test.csv')
        with open(file_path, 'r', encoding="utf-8") as f:
            reader = f.readlines()
        examples = []
        for index, line in enumerate(reader):
            guid = 'train-%d' % index
            split_line = line.strip().split("\t")
            text_a = tokenization.convert_to_unicode(split_line[1])
            text_b = tokenization.convert_to_unicode(split_line[2])
            label = split_line[3]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[2])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[2])
                label = tokenization.convert_to_unicode(line[4])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


if __name__ == "__main__":
    pass