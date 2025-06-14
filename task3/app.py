
from flask import Flask, request, render_template
import random
import re

app = Flask(__name__)

class MarkovChainTextGenerator:
    def __init__(self):
        self.model = {}

    def train(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        for i in range(len(words) - 1):
            curr, next_word = words[i], words[i + 1]
            if curr not in self.model:
                self.model[curr] = []
            self.model[curr].append(next_word)

    def generate(self, start_word, max_words=50):
        word = start_word.lower()
        output = [word]
        for _ in range(max_words - 1):
            next_words = self.model.get(word)
            if not next_words:
                break
            word = random.choice(next_words)
            output.append(word)
        return " ".join(output).capitalize() + "."

# Load extended dataset from external file
with open("extended_dataset.txt", "r", encoding="utf-8") as f:
    sample_text = f.read()

generator = MarkovChainTextGenerator()
generator.train(sample_text)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    if request.method == "POST":
        prompt = request.form["prompt"]
        generated_text = generator.generate(prompt, 50)
    return render_template("index.html", generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
