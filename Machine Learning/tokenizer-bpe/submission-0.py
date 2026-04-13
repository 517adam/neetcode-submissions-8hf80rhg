from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            pairs = []
            if len(tokens) < 2:
                break;
            for i in range(len(tokens)-1):
                pairs.append((tokens[i],tokens[i+1]))
            counts = Counter(pairs)
            max_freq = max(counts.values())
            best_pairs = [p for p, freq in counts.items() if freq == max_freq]  
            best_pair = min(best_pairs)       
            merges.append([best_pair[0],best_pair[1]])

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2 
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens   
        return merges