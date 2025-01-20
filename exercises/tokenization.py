import os
import re


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.vocab = vocab
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}


    def vocab_from_text(text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        unique_tokens = sorted(set(preprocessed))
        unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        return vocab

    @classmethod
    def from_text(cls, text):
        """Create a tokenizer from raw text input."""
        vocab = cls.vocab_from_text(text)
        return cls(vocab)

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def test_tokenizer():
    # Get the directory containing this file
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.path.join(CURRENT_DIR, "../samples/the-verdict.txt"), "r", encoding="utf-8"
    ) as f:
        raw_text = f.read()

    vocab = SimpleTokenizerV2.vocab_from_text(raw_text)
    # Create tokenizer from the raw text
    tokenizer = SimpleTokenizerV2(vocab)

    # Test the tokenizer
    test_text = """"It's the last he painted, you know,"
    Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(test_text)
    print(ids)
    print(tokenizer.decode(ids))
    print()


def test_tokenizer2():
    test_text = "The brown dog playfully chased the swift fox"

    # Create tokenizer from the raw text
    tokenizer = SimpleTokenizerV2.from_text(test_text)

    # Test the tokenizer
    ids = tokenizer.encode(test_text)
    print(ids)
    print(tokenizer.decode(ids))
    print()


def test_tokenizer3():
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    test_text = " <|endoftext|> ".join((text1, text2))
    print(test_text)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.path.join(CURRENT_DIR, "../samples/the-verdict.txt"), "r", encoding="utf-8"
    ) as f:
        raw_text = f.read()

    vocab = SimpleTokenizerV2.vocab_from_text(raw_text)

    # Create tokenizer from the raw text
    tokenizer = SimpleTokenizerV2(vocab)

    # Test the tokenizer
    ids = tokenizer.encode(test_text)
    print(ids)
    print(tokenizer.decode(ids))
    print()


if __name__ == "__main__":
    test_tokenizer()
    test_tokenizer2()
    test_tokenizer3()
