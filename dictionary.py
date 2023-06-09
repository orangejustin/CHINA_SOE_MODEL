import json

class Dictionary:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.dict_data = json.load(f)
            self.dict_data_ch_en = self.dict_data  # Chinese to English
            self.dict_data_en_ch = {v: k for k, v in self.dict_data.items()}  # English to Chinese

    def translate_en_ch(self, word):
        return self.dict_data_en_ch.get(word, "Word not found")

    def translate_ch_en(self, word):
        return self.dict_data_ch_en.get(word, "Word not found")
