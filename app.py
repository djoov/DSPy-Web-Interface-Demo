from flask import Flask, request, render_template_string
import dspy
import requests
import warnings

# Configure warnings and DSPy LM
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic._internal._config"
)
model_name = "llama3.2:3b"
lm = dspy.LM(
    "ollama_chat/" + model_name,
    api_base="http://localhost:11434",
    api_key="",
    cache=False,
)
dspy.configure(lm=lm)

app = Flask(__name__)

# HTML template
TEMPLATE = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>DSPy Web Interface</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 20px; }
      h1 { text-align: center; }
      form { margin-bottom: 40px; }
      label { display: block; margin-top: 10px; }
      input[type=text], textarea, select { width: 100%; padding: 8px; margin-top: 4px; }
      button { margin-top: 12px; padding: 8px 16px; }
      .result { background: #f4f4f4; padding: 10px; margin-top: 10px; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <h1>DSPy Demo Interface</h1>
    <form method="post" action="/run">
      <label for="function">Choose function:</label>
      <select name="function" id="function">
        <option value="count_letter">Count Letter</option>
        <option value="summarize">Summarize Text</option>
        <option value="translate">Translate Text</option>
        <option value="qa">Basic Q&A</option>
      </select>

      <div id="inputs">
        <!-- Inputs will be replaced via JavaScript -->
      </div>

      <button type="submit">Run</button>
    </form>

    {% if result %}
      <h2>Result:</h2>
      <div class="result">{{ result }}</div>
    {% endif %}

    <script>
      const inputsDiv = document.getElementById('inputs');
      const functionSelect = document.getElementById('function');
      
      function renderInputs() {
        const fn = functionSelect.value;
        let html = '';
        if (fn === 'count_letter') {
          html += '<label>Word:<input type="text" name="word" required></label>';
          html += '<label>Letter:<input type="text" name="letter" maxlength="1" required></label>';
        } else if (fn === 'summarize') {
          html += '<label>Text to Summarize:<textarea name="text" rows="4" required></textarea></label>';
        } else if (fn === 'translate') {
          html += '<label>Text to Translate:<textarea name="text" rows="4" required></textarea></label>';
          html += '<label>Target Language:<input type="text" name="target_language" required></label>';
        } else if (fn === 'qa') {
          html += '<label>Question:<input type="text" name="question" required></label>';
        }
        inputsDiv.innerHTML = html;
      }
      functionSelect.addEventListener('change', renderInputs);
      window.onload = renderInputs;
    </script>
  </body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(TEMPLATE)

@app.route('/run', methods=['POST'])
def run():
    fn = request.form['function']
    result = ''

    if fn == 'count_letter':
        word = request.form['word']
        letter = request.form['letter']
        # ReAct example
        def count_letter(w, l):
            return w.lower().count(l.lower()) if w and l and len(l) == 1 else 0
        class LetterCounter(dspy.Signature):
            word = dspy.InputField()
            letter = dspy.InputField()
            answer = dspy.OutputField(format='int')
        react = dspy.ReAct(LetterCounter, tools=[count_letter], max_iters=1)
        res = react(word=word, letter=letter)
        result = f"Letter '{letter}' appears {res.answer} times in '{word}'."

    elif fn == 'summarize':
        text = request.form['text']
        model = dspy.ChainOfThought('text -> summary')
        res = model(text=text)
        result = res.summary

    elif fn == 'translate':
        text = request.form['text']
        target = request.form['target_language']
        model = dspy.ChainOfThought('text, target_language -> translation')
        res = model(text=text, target_language=target)
        result = res.translation

    elif fn == 'qa':
        question = request.form['question']
        predictor = dspy.Predict('question -> answer')
        res = predictor(question=question)
        result = res.answer

    return render_template_string(TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)
