'''
pet6r
May 12 2025
Week 16 - Webscraping Final w/ bs4 and nltk
'''

# Standard lib imports
import os
import re
import requests
from datetime import datetime

# Third party lib imports
from bs4 import BeautifulSoup
from nltk import pos_tag, wordpunct_tokenize
from nltk.corpus import stopwords
from PIL import Image

# Check for NLTK data, download if not found
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Constants
URL = 'https://casl.website'
IMAGE_DIR = 'images'
STOPWORDS = set(stopwords.words('english'))

# Regex patterns
ZIPCODE_PATTERN = r'\b\d{5}(?:-\d{4})?\b'
PHONE_PATTERN = r'\(?\d{3}\)?-? *\d{3}-? *-?\d{4}'

# Create the image directory if it doesn't exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def scrape_website(url):
    '''
    Scrapes the target website for:
    - Unique page URLs
    - Unique image URLs
    - All text content
    - Phone numbers
    - Zip codes

    Parameters:
        url (str): The target website URL.

    Returns:
        tuple: A tuple containing:
            - page_links (set): Unique URLs of pages.
            - image_links (set): Unique URLs of images.
            - text_content (str): All text content from the website.
            - phone_numbers (set): Extracted phone numbers.
            - zip_codes (set): Extracted zip codes.
    '''
    page_links = set()
    image_links = set()
    text_content = ""
    phone_numbers = set()
    zip_codes = set()

    # Send GET request to the URL to retrieve HTML source
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get all anchor tags (links)
    for link in soup.find_all('a', href=True):
        href = link['href']
        if not href.startswith('http'):
            href = URL + href  # Append base URL for relative links
        if URL in href:  # Ensure the link stays within CyberApolis
            page_links.add(href)

    # Get all image tags
    for img in soup.find_all('img', src=True):
        src = img['src']
        # Convert URLs to absolute URLs
        if not src.startswith('http'):
            src = URL + src  # Append base URL for full image links
        image_links.add(src)

    # Extract text content
    text_content += soup.get_text(separator=' ', strip=True)

    # Find all phone numbers and zip codes
    phone_numbers.update(re.findall(PHONE_PATTERN, text_content))
    zip_codes.update(re.findall(ZIPCODE_PATTERN, text_content))

    return page_links, image_links, text_content, phone_numbers, zip_codes

def download_images(image_links):
    '''
    Downloads all images from the provided image URLs and saves them to the `images` directory.

    Parameters:
        image_links (set): A set of image URLs to download.
    '''
    for image_url in image_links:
        try:
            response = requests.get(image_url)
            image_name = os.path.basename(image_url)
            image_path = os.path.join(IMAGE_DIR, image_name)
            with open(image_path, 'wb') as img_file:
                img_file.write(response.content)
            print(f"Image saved: {image_path}")
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")

def process_text_with_nltk(text):
    '''
    Processes the provided text using NLTK to extract:
    - Unique vocabulary
    - Nouns
    - Verbs

    Parameters:
        text (str): The text content to process.

    Returns:
        tuple: A tuple containing:
            - unique_vocab (set): Unique vocabulary words.
            - nouns (set): Extracted nouns.
            - verbs (set): Extracted verbs.
    '''
    tokens = wordpunct_tokenize(text)
    filtered_tokens = [word for word in tokens if word.isalpha() and word.lower() not in STOPWORDS]
    unique_vocab = set(filtered_tokens)
    pos_tags = pos_tag(filtered_tokens)

    # Extract nouns and verbs
    nouns = {word for word, tag in pos_tags if tag.startswith('NN')}
    verbs = {word for word, tag in pos_tags if tag.startswith('VB')}

    return unique_vocab, nouns, verbs

def generate_report(page_links, image_links, phone_numbers, zip_codes, unique_vocab, nouns, verbs):
    '''
    Generates a report containing the results of the website scraping and text processing.
    The report is saved to a timestamped text file and also printed to the terminal.

    Parameters:
        page_links (set): Unique URLs of pages.
        image_links (set): Unique URLs of images.
        phone_numbers (set): Extracted phone numbers.
        zip_codes (set): Extracted zip codes.
        unique_vocab (set): Unique vocabulary words.
        nouns (set): Extracted nouns.
        verbs (set): Extracted verbs.
    '''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"report_{timestamp}.txt"

    with open(report_filename, 'w') as report_file:
        # Write the target URL at the top of the report
        report_file.write(f"Target Website: {URL}\n\n")
        print(f"Target Website: {URL}\n")

        # Write URLs to the file and print to terminal
        report_file.write(f"Unique URLs Found ({len(page_links)}):\n")
        print(f"Unique URLs Found ({len(page_links)}):")
        report_file.write(", ".join(page_links) + "\n\n")
        print(", ".join(page_links))

        # Write Image URLs to the file and print to terminal
        report_file.write(f"\nUnique Image URLs Found ({len(image_links)}):\n")
        print(f"\nUnique Image URLs Found ({len(image_links)}):")
        report_file.write(", ".join(image_links) + "\n\n")
        print(", ".join(image_links))

        # Write phone numbers to the file and print to terminal
        report_file.write(f"\nPhone Numbers Found ({len(phone_numbers)}):\n")
        print(f"\nPhone Numbers Found ({len(phone_numbers)}):")
        report_file.write(", ".join(phone_numbers) + "\n\n")
        print(", ".join(phone_numbers))

        # Write zip Codes to the file and print to terminal
        report_file.write(f"\nZip Codes Found ({len(zip_codes)}):\n")
        print(f"\nZip Codes Found ({len(zip_codes)}):")
        report_file.write(", ".join(zip_codes) + "\n\n")
        print(", ".join(zip_codes))

        # Write unique vocab to the file and print to terminal
        report_file.write(f"\nUnique Vocabulary Found ({len(unique_vocab)}):\n")
        print(f"\nUnique Vocabulary Found ({len(unique_vocab)}):")
        report_file.write(", ".join(unique_vocab) + "\n\n")
        print(", ".join(unique_vocab))

        # Write Nouns to the file and print to terminal
        report_file.write(f"\nNouns Found ({len(nouns)}):\n")
        print(f"\nNouns Found ({len(nouns)}):")
        report_file.write(", ".join(nouns) + "\n\n")
        print(", ".join(nouns))

        # Write Verbs to the file and print to terminal
        report_file.write(f"\nVerbs Found ({len(verbs)}):\n")
        print(f"\nVerbs Found ({len(verbs)}):")
        report_file.write(", ".join(verbs) + "\n\n")
        print(", ".join(verbs))

    print(f"\nReport saved to {report_filename}")

def main():
    '''Main function to call the scraping, processing, and reporting funcs.'''

    print(f"Target website: {URL}")
    page_links, image_links, text_content, phone_numbers, zip_codes = scrape_website(URL)

    print("\nDownloading images...")
    download_images(image_links)

    print("\nProcessing text with NLTK...")

    unique_vocab, nouns, verbs = process_text_with_nltk(text_content)

    # Generate and save the report
    print("\nGenerating report...")
    generate_report(page_links, image_links, phone_numbers, zip_codes, unique_vocab, nouns, verbs)

    print("\nCompleted")

if __name__ == '__main__':
    main()
