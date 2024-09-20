import nltk
nltk.download('wordnet')


import json
import os
print(os.getcwd())


from flask import Flask, request, jsonify
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
with open('data.json', 'r') as file:
    qa_dict = json.load(file)
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
preprocessed_questions = [' '.join([lemmatizer.lemmatize(word.lower()) for word in question.split()]) for question in qa_dict.keys()]
question_vectors = vectorizer.fit_transform(preprocessed_questions)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json["input"]
        logging.debug(f"Original user input: {user_input}")
        user_input = ' '.join([lemmatizer.lemmatize(word.lower()) for word in user_input.split()])
        logging.debug(f"Lemmatized user input: {user_input}")
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, question_vectors).flatten()
        logging.debug(f"Similarities: {similarities}")
        most_similar_index = similarities.argmax()
        logging.debug(f"Most similar question index: {most_similar_index}")
        answer = qa_dict[list(qa_dict.keys())[most_similar_index]]

        return jsonify({"response": answer})
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)



