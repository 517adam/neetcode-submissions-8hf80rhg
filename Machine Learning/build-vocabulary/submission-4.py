from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        tokens = sorted(list(set(text)))
        stoi = {}
        itos = {}
        for i, token in enumerate(tokens):
            stoi[token] = i
            itos[i] = token
        return (stoi, itos)
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        return [stoi[s] for s in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        char_list = [itos[index] for index in ids]
        return "".join(char_list)
