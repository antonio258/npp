import spacy
import re
import os
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
from typing import Callable
from sklearn.feature_extraction.text import TfidfVectorizer


class PreProcessing:
    """
    Class for performing text preprocessing operations.

    Args:
        noadverbs (bool, optional): Flag to remove adverbs from the text. Defaults to False.
        noadjectives (bool, optional): Flag to remove adjectives from the text. Defaults to False.
        noverbs (bool, optional): Flag to remove verbs from the text. Defaults to False.
        noentities (bool, optional): Flag to remove named entities from the text. Defaults to False.
        language (str, optional): Language for the Spacy model. Defaults to 'en'.
        remove_list (bool, optional): Flag to remove a list of words from the text. Defaults to False.

    Attributes:
        noadverbs (bool): Flag to remove adverbs from the text.
        noadjectives (bool): Flag to remove adjectives from the text.
        noverbs (bool): Flag to remove verbs from the text.
        noentities (bool): Flag to remove named entities from the text.
        language (str): Language for the Spacy model.
        remove_list (bool): Flag to remove a list of words from the text.
        punctuation (str): Regular expression pattern for removing punctuation.
        nlp (spacy.Language): Spacy language model.
        stopwords (list): List of stopwords.

    Methods:
        lowercase_unidecode: Converts text to lowercase and removes diacritics.
        remove_urls: Removes URLs from the text.
        remove_tweet_marking: Removes Twitter mentions and hashtags from the text.
        remove_punctuation: Removes punctuation from the text.
        remove_repetion: Removes repeated words from the text.
        append_stopwords_list: Appends additional stopwords to the existing list.
        remove_stopwords: Removes stopwords from the text.
        spacy_processing: Performs Spacy processing on a list of documents.
        remove_n: Removes words with length less than or equal to n from the text.
        remove_numbers: Removes or filters out numbers from the text.
        remove_gerund: Removes gerund endings from verbs in the text.
        remove_infinitive: Removes infinitive endings from verbs in the text.
        filter_by_idf: Filters out words based on their inverse document frequency.

    """

    def __init__(self, noadverbs: bool = False, noadjectives: bool = False, noverbs: bool = False,
                 noentities: bool = False, language: str = 'en', remove_list: bool = False):
        """
        Initialize the PreProcessing object.

        Args:
            noadverbs (bool, optional): Flag to indicate whether to remove adverbs. Defaults to False.
            noadjectives (bool, optional): Flag to indicate whether to remove adjectives. Defaults to False.
            noverbs (bool, optional): Flag to indicate whether to remove verbs. Defaults to False.
            noentities (bool, optional): Flag to indicate whether to remove named entities. Defaults to False.
            language (str, optional): Language code for the Spacy model. Defaults to 'en'.
            remove_list (bool, optional): Flag to indicate whether to remove stopwords. Defaults to False.
        """
        self.noadverbs = noadverbs
        self.noadjectives = noadjectives
        self.noverbs = noverbs
        self.noentities = noentities
        self.remove_list = remove_list
        self.punctuation = (
                r'\(|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/'
                r'|:|;|<|=|>|\?|\@|\[|\]|\^|_|`|\{|\}|~|\|'
                r'|\r\n|\n|\r|\\\)'
        )
        self.nlp = self._load_spacy_model(language)
        self.stopwords = [unidecode(x).lower() for x in list(self.nlp.Defaults.stop_words)]

    @staticmethod
    def _load_spacy_model(language: str = 'en'):
        """
        Load a Spacy language model based on the specified language.

        Args:
            language (str): The language code for the model. Defaults to 'en'.

        Returns:
            spacy.Language: The loaded Spacy language model.

        Raises:
            OSError: If the specified language model is not found and cannot be downloaded.
        """
        model_list = {
            'en': 'en_core_web_sm',
            'pt': 'pt_core_news_sm'
        }
        try:
            return spacy.load(model_list[language])
        except OSError:
            os.system(f'python -m spacy download {model_list[language]}')
        return spacy.load(model_list[language])

    @staticmethod
    def _process_text(text: str | list, function: Callable) -> str | list:
        """
        Process the given text using the provided function.

        Args:
            text (str | list): The text to be processed. It can be either a string or a list of strings.
            function (Callable): The function to be applied to the text.

        Returns:
            str | list: The processed text. If the input is a string, the output will be a string. If the input is a list,
            the output will be a list of processed strings. If the input is neither a string nor a list, an empty string
            will be returned.
        """
        if isinstance(text, str):
            return function(text)
        elif isinstance(text, list):
            return [function(x) for x in text]
        return ''

    def lowercase_unidecode(self, text: str | list) -> str | list:
        """
        Convert the given text to lowercase and remove any diacritical marks (accents).

        Args:
            text (str | list): The text to be processed. It can be either a string or a list of strings.

        Returns:
            str | list: The processed text. If the input is a string, the output will be a string. If the input is a list,
            the output will be a list of strings.

        Example:
            >>> pre_processor = PreProcessor()
            >>> text = "Café"
            >>> pre_processor.lowercase_unidecode(text)
            'cafe'
        """
        text = self._process_text(text, lambda value: value.lower())
        text = self._process_text(text, unidecode)
        return text

    def remove_urls(self, text: str | list) -> str | list:
        """
        Removes URLs from the given text or list of texts.

        Args:
            text (str | list): The text or list of texts from which to remove URLs.

        Returns:
            str | list: The text or list of texts with URLs removed.

        """
        return self._process_text(text, lambda value: re.sub(r'http\S+ *', '', value).strip())

    def remove_tweet_marking(self, text: str | list) -> str | list:
        """
        Removes tweet markings (e.g., @mentions and #hashtags) from the given text.

        Args:
            text (str | list): The text or list of texts to process.

        Returns:
            str | list: The processed text or list of processed texts with tweet markings removed.
        """
        return self._process_text(text, lambda value: re.sub(r'(@|#)\S+ *', '', value).strip())

    def remove_punctuation(self, text: str | list) -> str | list:
        """
        Removes punctuation from the given text.

        Args:
            text (str | list): The text from which punctuation needs to be removed.

        Returns:
            str | list: The text with punctuation removed.
        """
        text = self._process_text(text, lambda value: re.sub(self.punctuation, ' ', value))
        text = self._process_text(text, lambda value: re.sub(' {2,}', ' ', value).strip())
        return text

    def remove_repetion(self, text: str | list) -> str | list:
        """
        Removes repeated words in the given text.

        Args:
            text (str | list): The input text or list of words.

        Returns:
            str | list: The processed text with repeated words removed.

        """
        return self._process_text(text, lambda value: re.sub(r'\b(\w+)\s+\1\b', r'\1', value))

    def append_stopwords_list(self, stopwords: list) -> None:
        """
        Appends additional stopwords to the existing list of stopwords.

        Parameters:
        stopwords (list): A list of stopwords to be appended.

        """
        self.stopwords.extend(stopwords)

    def remove_stopwords(self, text: str | list) -> str | list:
        """
        Removes stopwords from the given text.

        Args:
            text (str | list): The input text from which stopwords need to be removed.

        Returns:
            str | list: The processed text with stopwords removed.

        """
        return self._process_text(text, lambda value: re.sub(rf'\b({"|".join(self.stopwords)})\b *', '', value).strip())

    def spacy_processing(self, docs: list, n_process: int = -1, lemma: str = False) -> list:
        """
        Preprocesses a list of documents using spaCy.

        Args:
            docs (list): List of documents to be processed.
            n_process (int, optional): Number of processes to use for parallel processing. Defaults to -1.
            lemma (str, optional): Flag indicating whether to lemmatize the tokens. Defaults to False.

        Returns:
            list: List of preprocessed documents.

        """
        all_docs = self.nlp.pipe(docs, n_process=n_process)
        pp_docs = []
        for doc in tqdm(all_docs):
            pp_doc = [token for token in doc if token.is_ascii]  # remove no ascii
            if self.noadverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADV']  # remove adverbs
            if self.noadjectives:
                pp_doc = [token for token in pp_doc if token.pos_ != 'ADJ']  # remove adjectives
            if self.noverbs:
                pp_doc = [token for token in pp_doc if token.pos_ != 'VERB']  # remove verbs
            pp_doc = [token for token in pp_doc if not token.is_space]  # remove whitespace

            # Remove Entities
            if self.noentities:
                pp_doc = [
                    token for token in pp_doc if
                    token.ent_type_ not in
                    ['MONEY', 'DATE', 'PERSON', 'PERCENT', 'ORDINAL', 'CARDINAL',
                     'QUANTITY', 'GPE', 'NORP', 'LANGUAGE']
                ]

            pp_doc = [
                self.remove_infinitive(token.lemma_).lower() if token.pos_ == 'VERB' else
                token.lemma_.lower() if lemma else token.lower_ for token in pp_doc
            ]
            pp_docs.append(' '.join(pp_doc))
        return pp_docs

    def remove_n(self, text: str | list, n: int) -> str | list:
        """
        Removes words of length 1 to n followed by the word 'pri' from the given text.

        Args:
            text (str | list): The input text or list of texts to process.
            n (int): The maximum length of words to remove.

        Returns:
            str | list: The processed text or list of processed texts.

        """
        return self._process_text(text, lambda value: re.sub(rf'\b\w{{1,{n}}}\b ?pri', '', value).strip())

    def remove_numbers(self, text: str | list, mode: str = 'replace') -> str | list:
        """
        Removes or replaces numbers in the given text.

        Args:
            text (str | list): The input text or list of texts.
            mode (str, optional): The mode of operation. Defaults to 'replace'.
                - 'filter': Removes the numbers from the text.
                - 'replace': Replaces the numbers with an empty string.

        Returns:
            str | list: The processed text or list of processed texts.
        """
        if mode == "filter":
            return self._process_text(text, lambda value: '' if re.search('[0-9]', value) else value)
        elif mode == "replace":
            return self._process_text(text, lambda value: re.sub('[0-9] *', '', value))

    def remove_gerund(self, text: str | list) -> str | list:
        """
        Removes the gerund form '-ndo' from the given text.

        Args:
            text (str | list): The input text or list of texts to process.

        Returns:
            str | list: The processed text with the gerund form removed.

        """
        return self._process_text(text, lambda value: re.sub(r'ndo\b', '', value))

    def remove_infinitive(self, text: str | list) -> str | list:
        """
        Removes the infinitive form of verbs from the given text.

        Args:
            text (str | list): The input text or list of texts to process.

        Returns:
            str | list: The processed text with infinitive forms removed.

        """
        return self._process_text(text, lambda value: re.sub(r'r\b', '', value))

    @staticmethod
    def filter_by_idf(text: list) -> list:
        """
        Filters the input text by removing words with IDF (Inverse Document Frequency) values below a certain threshold.

        Args:
            text (list): The input text to be filtered.

        Returns:
            list: The filtered text with words removed based on IDF values.
        """
        vectorizer = TfidfVectorizer()
        vectorizer.fit(text)
        idfs = vectorizer.idf_
        min_value = np.percentile(idfs, 0.1)
        words = np.where(idfs <= min_value)
        vocab = vectorizer.get_feature_names_out()
        words = [vocab[x] for x in words][0]
        return [" ".join([y for y in x.split(" ") if y not in words]) for x in text]