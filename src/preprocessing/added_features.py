"""
Filename: added_features.py
Author: Beck Chamberlain
Version: 0.04
Description: Added extra features to the csv
Resources:
    https://informationr.net/ir/20-1/paper663.html
    https://www.geeksforgeeks.org/python/ssl-certificate-verification-python-requests/
    https://woteq.com/how-to-check-certificate-expiry-using-openssl-in-python
"""
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import ssl
import socket
from datetime import datetime
import dns.resolver
import nltk
from nltk.tokenize import sent_tokenize
from spellchecker import SpellChecker
spell = SpellChecker()

#Load pyspellchecker

#Load sentence tokeniser
nltk.download("punkt_tab")

def dmarc_check(domain):
    """
    Checks whether a domain has a DMARC record in its DNS

    :param domain: The domain to strip & check
    :return: If the domain has a DMARC record or not
    """
    # Strip www. from the domain name
    domain = domain.replace("www.", "", 1)
    #
    try:
        texts = dns.resolver.resolve(f"_dmarc.{domain}", "TXT")
        for record in texts:
            if "DMARC1" in record.to_text():
                return 1
    except Exception:
        pass
    return 0


def http_certification_check(url):
    """
    Scores a webpage based on if it's encrypted with a valid ssl certificate.
    This is done by checking the scheme of the URL, fetching the SSL Certificate from hostname and port.

    Resource: https://woteq.com/how-to-check-certificate-expiry-using-openssl-in-python

    :return: A score between 0 and 3 representing the certification level of the webpage
    """
    certification_score = 0
    # Gets the parsed url
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # Check HTTPS certification
    try:
        # Checks scheme
        if parsed_url.scheme == 'https':
            certification_score += 1
        # Creates a default SSL context
        context = ssl.create_default_context()
        # Wraps a socket connection with the SSL context to get the certificate
        with socket.create_connection((domain, 443), timeout=10) as sock_connection:
            with context.wrap_socket(sock_connection, server_hostname=domain) as sock_wrapper:
                certificate = sock_wrapper.getpeercert()
                # Adds to score because certificate exists
                certification_score += 1
        # Gets the expiry date
        certificate_expiry_date = datetime.strptime(certificate['notAfter'], "%b %d %H:%M:%S %Y %Z")
        # Compares with the date when code is run
        days_remaining = (certificate_expiry_date - datetime.now()).days
        if days_remaining > 0:
            certification_score += 1
    except Exception:
        pass
    return certification_score


def html_w3c_compliance(url, soup = None):
    """
    Uses the W3C API to verify a sites compliance with HTML standards

    :param soup:
    :param url: The URL of the webpage to be investigated
    :return: A score between 0 and 5 indictive of how many errors there are per line
    """
    if soup is None:
        try:
            webpage = requests.get(url, timeout=20)
            soup = BeautifulSoup(webpage.text, "html.parser")
        except Exception as exception:
            return 0
    # The URL of the w3c tool
    w3c_validator = "https://validator.w3.org/nu/"

    # Sets the url to be checked and the format for the response
    parameters = {
        "doc": url,
        "out": "json"
    }

    # Identify to the site that its being accessed by an automated tool
    headers = {"User-Agent": "WebsiteCredibilityChecker/0.3 (fake-news-detector; bec68@aber.ac.uk)"}

    # Stores the response in JSON format then extrapolates the messages
    response = (requests.get(w3c_validator, headers=headers, params=parameters)).json()
    messages = response.get("messages", [])
    # Counts the number of error and warning messages
    parser_stopped = any("Cannot recover after last error" in message.get("message", "") for message in messages)
    if parser_stopped:
        return 5
    else:
        number_of_errors = sum(1 for message in messages if message["type"] == "error")
        number_of_warnings = sum(1 for message in messages if message["type"] == "warning")

    #Counts number of tags
    number_of_tags = len(soup.find_all())
    # If no tags found then return 0
    if number_of_tags == 0:
        return 0
    else:
        # Calculates the issues per tag using errors and warnings per tags with errors weighted higher
        issues_per_tag = (number_of_errors + number_of_warnings/2) / number_of_tags

    return round(issues_per_tag,3)

def website_credibility_tests(url):
    """
    A function for testing the extra function features
    :return:
    """
    credibility_score = 0
    # Gets the webpage info using beautiful soup
    try:
        webpage = requests.get(url, timeout=20)
        soup = BeautifulSoup(webpage.text, "html.parser")
    except Exception as exception:
        return 0

    # Gets the domain and parsed url of the site
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    # Adds to score if http has a certificate and if its valid
    credibility_score += http_certification_check(domain, url)
    # Adds to score if the site has a digital signature
    if dmarc_check(domain):
        credibility_score += 3

    # Adds to score if the site's html is compliant with w3c standards
    html_w3c_compliance(url,soup)
    # Checks how recently the site was updated

    return credibility_score

def spelling_check(body_of_text):
    """
    Chose language_tool_python not pyspellchecker because of how it handles proper nouns

    Checks the main body of text in a webpage for any spelling errors.
    :param body_of_text: The body of the webpage
    :return: The number of spelling errors that occurred
    """
    #Gets the individual words and how many there are
    words = body_of_text.split()
    number_of_words = len(words)

    # Gets number of unknown words, discounting words with capital letters at start as they may be proper nouns or
    # abbreviations and discounting words with any non alphabet characters
    spellchecked_words = [word for word in words if word.isalpha() and word.islower()]


    #Gets number of mistakes per word
    if number_of_words > 0:
        mistakes_per_word = round(len(spell.unknown(spellchecked_words)) / number_of_words,3)
    else:
        return 0
    return mistakes_per_word

def textual_analysis(body_of_text):
    # Gets the individual words
    words = body_of_text.split()

    # Get number of sentences & words in the text
    number_of_sentences = len(sent_tokenize(body_of_text))
    number_of_words = len(words)
    # Calculates average sentence length and exclamation/question marks per sentences
    if number_of_sentences > 0:
        average_sentence_length = round(number_of_words / number_of_sentences,3)
        exclamation_marks_per_sentences = round(body_of_text.count("!") / number_of_sentences,3)
        question_marks_per_sentences = round(body_of_text.count("?") / number_of_sentences,3)
    else:
        exclamation_marks_per_sentences = 0
        question_marks_per_sentences = 0
        average_sentence_length = 0

    #Gets how many spelling mistakes there are per word
    mistakes_per_word = spelling_check(body_of_text)

    #Gets number of full upper case words excluding one-letter words eg "I"
    if number_of_words > 0:
        number_of_uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        uppercase_words_frequency = round(number_of_uppercase_words / number_of_words,3)
    else:
        uppercase_words_frequency = 0

    #Calulates how many words are unique out of all the words
    diversity_in_language = round(len(set(words)) / number_of_words, 3)

    return exclamation_marks_per_sentences, question_marks_per_sentences, uppercase_words_frequency, average_sentence_length, diversity_in_language, mistakes_per_word


