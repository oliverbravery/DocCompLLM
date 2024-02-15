# Document Comprehension with LLMs
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)



This project provides a convenient way to feed a PDF document to the LLM for comprehension and questioning. You can either use a local model with Ollama or the OpenAI API to process the PDF document.

With the local model approach, you can install and configure [Ollama](https://ollama.com), an accessible way to run LLMs locally.

Alternatively, you can utilize the [OpenAI API](https://openai.com/blog/openai-api) providing a cloud-based solution. You will need to set up an account and obtain an API key to use the OpenAI API.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

To install the project, first clone the repository:

```bash
git clone https://github.com/oliverbravery/DocCompLLM.git
```

Next, navigate to the project directory, set up the environment and install the dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After the installation, you will need to set up the environment variables. I recommend using a `.env` file. Copy and rename the `.env.template` file and fill in the necessary values:

```bash
cp .env.template .env
```

Finally, you can run the project:

```bash
python main.py
```

## Usage

The project provides two options for utilizing the LLM: either by using a local model with Ollama or by leveraging the OpenAI API. By default, the project uses the OpenAI API. To switch to the local model, you need to use the optional `--local` flag:

```bash
python main.py --local
```

You can also use the optional `--clean` flag to clear the terminal before the LLM's output is displayed:

```bash
python main.py --clean
```

## License

This project is licensed under the [MIT License](LICENSE).
