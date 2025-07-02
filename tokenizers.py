from collections import defaultdict
import os
import json

from tqdm import tqdm


class BytePairEncoding:
    def __init__(self):
        self.token_mapping: defaultdict[str, int] = defaultdict(lambda: -1)
        self.reverse_token_mapping: defaultdict[int, str] = defaultdict(lambda: "")
        self.merges: defaultdict[tuple[int, int], int] = defaultdict(lambda: -1)

    def setup_by_dataset(self, dataset: str, merging_iterations: int = 5000):
        for i in range(256):
            self.token_mapping[chr(i)] = i
            self.reverse_token_mapping[i] = chr(i)

        dataset_tokens = list(map(int, dataset.encode('utf-8')))
        latest_token = int(2 ** 8)

        for cnt in tqdm(range(merging_iterations)):
            pair_map_count = self._get_pair_map_count(dataset_tokens)

            max_pair = max(pair_map_count, key=pair_map_count.get)
            max_pair_count = pair_map_count[max_pair]

            if max_pair_count <= 1:
                break

            self.token_mapping[
                self.reverse_token_mapping[max_pair[0]] + self.reverse_token_mapping[max_pair[1]]] = latest_token
            self.reverse_token_mapping[latest_token] = self.reverse_token_mapping[max_pair[0]] + \
                                                       self.reverse_token_mapping[max_pair[1]]
            self.merges[max_pair] = latest_token

            new_dataset_tokens = self._merge(dataset_tokens, max_pair, latest_token)
            latest_token += 1
            dataset_tokens = new_dataset_tokens.copy()

    @staticmethod
    def _merge(dataset_tokens: list[str], max_pair: tuple[str, str], latest_token: int):
        new_dataset_tokens = []
        idx = 0
        while idx < len(dataset_tokens) - 1:
            if (dataset_tokens[idx], dataset_tokens[idx + 1]) == max_pair:
                new_dataset_tokens.append(latest_token)
                idx += 2
            else:
                new_dataset_tokens.append(dataset_tokens[idx])
                idx += 1
        return new_dataset_tokens

    @staticmethod
    def _get_pair_map_count(text_tokens: list[int]) -> defaultdict[tuple]:
        pair_map_count = defaultdict(lambda: 0)
        for i in range(len(text_tokens) - 1):
            pair = tuple(text_tokens[i: i + 2])
            pair_map_count[pair] += 1
        return pair_map_count

    def tokenize(self, text: str) -> list[int]:
        text_tokens = list(map(int, text.encode('utf-8')))
        while True:
            pair_map_count = self._get_pair_map_count(text_tokens)

            min_pair = max(pair_map_count, key=lambda p: self.merges.get(p, float('inf')))
            if min_pair not in self.merges:
                break
            text_tokens = self._merge(text_tokens, min_pair, self.merges[min_pair]).copy()

        return text_tokens

    def detokenize(self, arr: list[int]) -> str:
        return "".join([self.reverse_token_mapping[int(idx)] for idx in arr])

    @property
    def vocab_size(self) -> int:
        return len(self.token_mapping)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/token_mapping.json", "w") as file:
            json.dump(self.token_mapping, file, indent=4)

        with open(f"{path}/reverse_token_mapping.json", "w") as file:
            json.dump(self.reverse_token_mapping, file, indent=4)

        with open(f"{path}/merges.json", "w") as file:
            write_merges = dict()
            for k, val in self.merges.items():
                write_merges[f'{k[0]},{k[1]}'] = val
            json.dump(write_merges, file, indent=4)

    def load(self, path: str):
        with open(f"{path}/token_mapping.json", "r") as file:
            self.token_mapping = json.load(file)

        with open(f"{path}/reverse_token_mapping.json", "r") as file:
            self.reverse_token_mapping = {int(k): val for k, val in json.load(file).items()}

        with open(f"{path}/merges.json", "r") as file:
            write_merges = json.load(file)
            for k, val in write_merges.items():
                self.merges[tuple(map(int, k.split(',')))] = val

