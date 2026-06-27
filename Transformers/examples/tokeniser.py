import re


class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:v for v, i in vocab.items()}

    def encode(self, text):
        tokens = []
        reg = ',-.?!;:()[]{}"\'\s'
        for work in re.split(f'[{reg}]+', text):
            tokens.append(self.str_to_int.get(work, self.str_to_int['<unk>']))
        return tokens
    def decode(self, tokens):
        text = ' '.join([self.int_to_str.get(token, '<unk>') for token in tokens])
        # Remove spaces before punctuation
        text = re.sub(r'\s+([,.?!;:()"\'])', r'\1', text)
        return text
