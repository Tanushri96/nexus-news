from transformers import pipeline
sa=pipeline("sentiment-analysis")
print(sa("This food quality is excellent!"))

ner=pipeline("ner",grouped_entities=True)
print(ner("Infosys is based in Bengaluru,Karnataka."))

qa=(pipeline("question-answering"))
context="Python was created by Guido van Rossum in 1991.It is widely used in AI"
summ=pipeline("summarization")
long_text="Generative Ai refers to AI systems that can generate new content..."
print(summ(long_text,max_length=60,min_length=20))