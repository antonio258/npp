# NLP Pre-Processing (npp)

This project is a collection of Python scripts for Natural Language Processing (NLP) pre-processing tasks. It provides a variety of functions to clean and prepare text data for further NLP analysis.

## Features

- Text cleaning: Remove unnecessary characters, numbers, and specific word forms from the text.
- Part-of-speech filtering: Ability to remove adverbs, adjectives, verbs, and other parts of speech from the text.
- Entity removal: Remove specific entities like money, date, person, etc.
- Infinitive form removal: Remove the infinitive form of verbs from the text.
- IDF filtering: Filter words based on their Inverse Document Frequency (IDF) values.

## Installation

This project requires Python and pip installed. Clone the project and install the dependencies:

```bash
pip install git+https://github.com/antonio258/npp.git
```

## Usage
Import the pre-processing module in your Python script and create an instance of the pre-processing class. Then, call the desired pre-processing methods on your text data.

```python

from pre_processing import PreProcessing

pp = PreProcessing()
text = "Your text data here"
processed_text = pp.lowercase_unidecode(text)
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.