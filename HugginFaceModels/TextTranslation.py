#Text transaltion 

# from transformers import pipeline

# translator = pipeline("translation_en_to_fr", model="t5-base", min_length = 233)
# res = translator("By following these steps, you can easily set up and use translation pipelines with Hugging Face's transformers library to translate text between various languages. Adjust the model and parameters as needed for your specific use case.")
# print(res)


## Japanese to EN
# # Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en", min_length = 25)

res = pipe("私の夢は海賊王になることです。私の名前はモンキー・D・ルフィ、あなたを徹底的に叩きのめします")

print(res)
