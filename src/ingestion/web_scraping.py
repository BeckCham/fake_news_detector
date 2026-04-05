"""
Filename: web_scraping.py
Author: Beck Chamberlain
Version: 0.03
Description: This script gathers certain data from a given news site and compacts it into a singular string that
             can be embedded
References:
    https://stackoverflow.com/questions/68087698/how-to-scrape-author-name-and-author-url-from-a-webpage-using-python
    https://dev.to/leapcell/scrape-like-a-pro-beautifulsoup-python-full-tutorial-nj4
AI Declaration:
    Claude code was used for a large part of the get_author_from_url function:
    'Write a function that gets the author from a URL using Article'
    Claude code was also used to get the MetaTags:
    'How can I use beautiful soup to get the Meta keywords from a URL'
"""
from newspaper import Article
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from src.preprocessing import added_features


def get_author_from_url(url):
    """
    Uses Article library to go through the given webpage and extract the author/s if they can be found.

    :param url: URL of news article with desired information
    :return: Either the author/s of the news article or "Not found"
    """
    try:
        article = Article(url)
        article.download()
        article.parse()

        # Returns list of authors
        authors = article.authors
        return ', '.join(authors) if authors else "Unknown"

    except Exception as e:
        return "Not found"


def url_to_data(url):
    """
    Uses BeautifulSoup to extract relevant data from the given url compiling them into one text consistent with the
    combining of textual features done in csv_cleaning.py

    :param url: URL of news article with desired information
    :return: A string combining all the extracted data from the webpage
    """
    # Gets the html of the given url & parses it
    try:
        webpage = requests.get(url, timeout=20)
        soup = BeautifulSoup(webpage.text, "html.parser")
    except Exception:
        return None

    # Creates a text variable to be embedded
    text = ''
    # Adds the domain x2
    domain = urlparse(url).netloc
    if domain:
        text += domain + ' ' + domain + ' '

    # Adds the title x2
    if soup.title:
        title = soup.title.get_text()
        text += title + ' ' + title + ' '

    # Adds meta keywords x2
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords:
        text += meta_keywords.get('content') + ' ' + meta_keywords.get('content') + ' '

    # Adds author x2
    author = soup.find("meta", attrs={"name": "author"})
    if author:
        text += author.get('content') + ' ' + author.get('content') + ' '
    else:
        author = get_author_from_url(url)
        if author != "Not found":
            text += author + ' ' + author + ' '

    # Adds content
    content = soup.find("article")
    if content:
        text += content.get_text(strip=True)
    else:
        content = soup.find_all("p")
        text += " ".join(paragraph.get_text(strip=True) for paragraph in content)

    # Adds metadescription
    meta_description = soup.find('meta', attrs={'name': 'description'})
    if meta_description:
        text += (meta_description.get('content'))

    #####   Adds extra features
    # Adds feature that represents if webpage has DMARC
    dmarc_present = added_features.dmarc_check(domain)
    # Adds feature that holds a score between 0-3 reflecting the SSL/TLS security configuration of a website
    ssl_score = added_features.http_certification_check(domain,url)
    #Adds feature that represents how w3c compliant a webpage is
    if soup is not None:
        w3c_score = added_features.html_w3c_compliance(url,soup)
    else:
        w3c_score = None
    # Gets textual analysis information
    textual_features = added_features.textual_analysis(content)

    return text, dmarc_present, ssl_score, w3c_score,textual_features
