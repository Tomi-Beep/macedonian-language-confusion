# Macedonian Language Confusion

Can a small language model stick to Macedonian when the prompt mixes in English? I tested Llama 3.2 (1B) on questions from the Macedonian Natural Questions dataset, using Ollama to run the model locally.

The script compares four ways of prompting the model: a direct question, examples in Macedonian, examples with some English words mixed in, and a simple request to repeat a Macedonian sentence. It uses fastText to check whether each response is identified as Macedonian.

[Read the project report (PDF)](%5B%D0%92%D0%98%5D%20%D0%98%D0%B7%D0%B2%D0%B5%D1%88%D1%82%D0%B0%D1%98%20%D0%BD%D0%B0%20%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%20-%20%D0%A2%D0%BE%D0%BC%D0%B8%20%D0%91%D0%BE%D0%B3%D0%BE%D0%B5%D1%81%D0%BA%D0%B8%20236024.pdf) · [View the Python script](main.py)

## Run the experiment

With Ollama running, install the Python packages and download the model:

```bash
pip install ollama datasets deep-translator fasttext
ollama pull llama3.2:1b
```

Place the fastText language identification file `lid.176.bin` next to `main.py`, then run `python main.py` from that folder.
