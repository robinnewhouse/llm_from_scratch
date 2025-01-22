import tiktoken
import os

tokenizer = tiktoken.get_encoding("gpt2")

CURRENT_DIR = os.path.dirname(__file__)
with open(os.path.join(CURRENT_DIR, "../samples/the-verdict.txt"), "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

print(f"Number of tokens: {len(enc_text)}")
print()
enc_sample = enc_text[50:]


# x contains input tokens and y contains targets, inputs shifted by 1

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

print(f"x: {x}")
print(f"y:      {y}")
print()

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)
print()
