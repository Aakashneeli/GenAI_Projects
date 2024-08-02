#Sentiment analysis(classifier)
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("Im going to haircut")

print(res)
