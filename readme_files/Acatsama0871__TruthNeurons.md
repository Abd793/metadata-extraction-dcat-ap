# Truth Neurons

## Dependencies

The dependencies are managed with `uv`. The packages required are listed in `pyproject.toml` and `uv.lock`.

## Functionality

- **Process TriviaQA Data to Generate Adverbial Examples**

```bash
python process_trivia_qa_data.py
```

- **Get the Identified Neurons**
Replace the config file with the ones listed in `configs/`

```bash
# 1. Process the truthful-qa data
python main.py truthful-qa-data
# 2. Generate prompt template
python main.py truthful-qa-qa
# 3. Attribute the neurons
python main.py attribute
# 4. Get the truth neurons
python main.py stats-test
```

- **Suppress Neurons Experiment**

```bash
python main.py suppress-neurons-exp
```

- **Benchmark**
Running argument:
- `-m`: `raw` or `suppressed`, where `raw` is the original model and `suppressed` is the model with the truth neurons suppressed.
- `-t`: `mmlu` or `trivia_qa`, where `mmlu` is the MMLU dataset and `trivia_qa` is the TriviaQA dataset.

```bash
python main.py benchmark
```
