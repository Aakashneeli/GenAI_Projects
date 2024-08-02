
#Text Generator
from transformers import pipeline

generator = pipeline("text-generation", model = "distilgpt2", temperature = 1)

res = generator(
        "My guy arnav is a Indian who", 
        max_length = 40, 
        num_return_sequences = 3
    )

print('-----------------------------------')
for r in res:
    print(r['generated_text'])
