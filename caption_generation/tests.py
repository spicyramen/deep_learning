from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
texts = ["The sun is shining in June!", "September is grey.", "Life is beautiful in August.", "I like it",
         "This and other things?"]
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)
print tokenizer.texts_to_sequences(["June is beautiful and I like it!"])


