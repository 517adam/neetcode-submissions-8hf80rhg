import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentence = positive + negative
        all_word = []
        for sentence in all_sentence:
            for word in sentence.split():
                all_word.append(word)
        all_word = list(set(all_word))
        all_word.sort()       
        all_sentence = []
        for words in positive:
            positive_encoded = []
            for word in words.split():
                positive_encoded.append(all_word.index(word)+1.0)
            all_sentence.append(torch.tensor(positive_encoded))
        for words in negative:
            negative_encoded = []
            for word in words.split():
                negative_encoded.append(all_word.index(word)+1.0)
            all_sentence.append(torch.tensor(negative_encoded))
        return nn.utils.rnn.pad_sequence(all_sentence, batch_first=True)
        

