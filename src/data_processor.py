# @Project      : taskcraft
# @File         : data_processor.py
# @Author       : Jingyi Cao <224040283@link.cuhk.edu.cn>
# @LastUpdated  : 2025/6/11
# @LICENSE      : Apache License 2.0

import logging
import os
import re
from bs4 import BeautifulSoup
import json
import requests
import langid
import multiprocessing
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import unquote
import pandas as pd
import pdfplumber
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import run_llm_prompt, write_json

logging.getLogger('pdfminer').setLevel(logging.ERROR)


def process_news_csv(csv_path):
    """
    Read a CSV file and set the 'title' column as the identifier and the 'body' column as the content for each row.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        list: A list of dictionaries, each containing 'identifier' and 'content' keys.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure the CSV file contains the required columns
        if 'title' not in df.columns or 'body' not in df.columns:
            raise ValueError("CSV file must contain 'title' and 'body' columns")

        # Process the data
        processed_data = []
        for _, row in df.iterrows():
            item = {
                'identifier': str(row['title']).strip(),
                'content': str(row['body']).strip()
            }
            processed_data.append(item)

        return processed_data

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def extract_page_text(args):
    """
    Extract text from a specific page of a PDF file.

    Args:
        args: Tuple(pdf_path, page_num)

    Returns:
        str: Extracted text from the specified page.
    """
    pdf_path, page_num = args
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_num].extract_text()


def multiprocess_load_pdf(pdf_path, max_num=10):
    """
    Load and extract text from multiple pages of a PDF file using multiprocessing.

    Args:
        pdf_path (str): Path to the PDF file.
        max_num (int): Maximum number of pages to process.

    Returns:
        list: List of extracted text from each page.
    """
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = min(len(pdf.pages), max_num)

    with multiprocessing.Pool(max_num) as pool:
        results = pool.map(extract_page_text, [(pdf_path, page) for page in range(num_pages)])
    return results


def identify_address_type(content) -> Tuple[str, str]:
    """
    Identify the type of address (URL, file path, or text content).

    Args:
        content: The content to be identified, can be a URL, file path, or text content.
    Returns:
        Tuple[str, str]: (content_type, addr_type)
            content_type: 'pdf', 'web', 'text', or 'html'
            addr_type: 'url', 'file', or 'unknown'
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    txt_exts = {'.txt', '.csv', '.json', '.xml'}
    pdf_exts = {'.pdf'}
    file_exts = {
        '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar', '.7z', '.tar', '.gz',
        '.gif', '.mp3', '.mp4', '.avi', '.mkv', '.docx', '.doc'
    }
    web_exts = {'.html', '.htm', '.php', '.asp', '.aspx', '.jsp'}

    # special case
    if 'arxiv.org/pdf' in content:
        return 'pdf', 'file'

    ext = os.path.splitext(content)[1].lower()

    if ext.startswith('.'):
        if ext in img_exts:
            return 'image', 'file'
        elif ext in txt_exts:
            return 'text', 'file'
        elif ext in pdf_exts:
            return 'pdf', 'file'
        elif ext in file_exts:
            return ext, 'file'

    # Check if it is a URL
    if content.startswith(('http://', 'https://')) or ext in web_exts:
        return 'web', 'url'

    return 'text', 'command'


def extract_html_content(file_path):
    """
    Extract title and main content from an HTML file

    Args:
        file_path (str): Path to the HTML file

    Returns:
        dict: Contains 'title' and 'content' of the webpage
    """
    try:
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        title = soup.title.string.strip() if soup.title else "No title found"

        # Remove unwanted elements (scripts, styles, etc.)
        for element in soup(['script', 'style', 'nav', 'footer', 'head', 'iframe', 'noscript']):
            element.decompose()

        # Extract main content - several strategies combined
        content = ""

        # Strategy 1: Look for common content containers
        main_tags = ['main', 'article', 'div.content', 'div.main', 'div.post', 'div.entry']
        for tag in main_tags:
            if soup.select(tag):
                content = ' '.join([p.get_text().strip() for p in soup.select(f'{tag} p')])
                if content:
                    break

        # Strategy 2: If no main content found, use body text
        if not content:
            content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

        # Clean up content
        content = re.sub(r'\s+', ' ', content)  # Replace multiple whitespace with single space
        content = content.strip()

        return title, content

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return "", ""


def find_main_title_enhanced(pdf_path, model, max_num=10):
    """
    Extract the main title from a PDF file using LLM and language detection, processing up to max_num pages.

    Args:
        pdf_path (str): Path to the PDF file.
        model: The model to be used for title extraction.
        max_num (int): Maximum number of pages to process.

    Returns:
        tuple: (main_title, all_pages_text)
            main_title (str): The extracted main title.
            all_pages_text (str): Concatenated text from all processed pages.
    """
    all_pages = multiprocess_load_pdf(pdf_path, max_num=max_num)

    all_pages = [each for each in all_pages if each]
    # Detect the language of the input text
    if len(all_pages) == 0:
        return "", ""

    first_page_text = all_pages[0]
    all_pages_text = "".join(all_pages)

    language, _ = langid.classify(first_page_text)

    # Extract the main title
    main_title = ""
    for _ in range(3):
        try:
            # Use the first 200 characters of the first page to extract the title
            input_text = first_page_text[:200] if len(first_page_text) > 200 else first_page_text
            response = run_llm_prompt(
                model,
                f"Input content: {input_text}",
                "Please find the main title from the given text. The main title can be: title, web page title, website name, dataset name, etc."
                f", return as JSON format with key 'title', and the language must be {language}",
                return_json=True
            )
            main_title = response.get("title", "").strip()
            if main_title:
                break
        except Exception as e:
            print(f"[find_main_title_enhanced]: {e}")

    return main_title, all_pages_text


def get_content_identifier(content: str, content_type: str = "text") -> str:
    """
    Get the identifier for the content

    Args:
        content: content text
        content_type: could be 'pdf', 'web', or 'text'
    Returns:
        str: content identifier
    """
    if not content:
        return "Unknown Content"
    elif content_type == "web":
        # Try to extract the title from the web content
        # The web page title is usually in the <title> tag
        return extract_html_title(content)

    # Default: use the first line of text as identifier
    first_line = content.split('\n')[0].strip()
    return first_line if first_line else "Unknown Content"


def extract_html_title(path: str) -> str:
    """
    Extract the title from HTML content using multiple matching strategies

    Args:
        path: HTML path
    Returns:
        str: Extracted title, returns an empty string if not found
    """
    html_content = get_html_content(path)
    if not html_content:
        return ""
    # 1. Standard <title> tag
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    if title_match:
        return title_match.group(1).strip()
    # 2. Open Graph title
    og_title_match = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\'](.*?)["\']',
                               html_content, re.IGNORECASE)
    if og_title_match:
        return og_title_match.group(1).strip()
    # 3. Twitter card title
    twitter_title_match = re.search(r'<meta[^>]*name=["\']twitter:title["\'][^>]*content=["\'](.*?)["\']',
                                    html_content, re.IGNORECASE)
    if twitter_title_match:
        return twitter_title_match.group(1).strip()
    # 4. h1 tag (as a fallback)
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
    if h1_match:
        return h1_match.group(1).strip()
    # 5. Standard first heading tag (h1-h6)
    h_tag_match = re.search(r'<h[1-6][^>]*>(.*?)</h[1-6]>', html_content, re.IGNORECASE | re.DOTALL)
    if h_tag_match:
        return h_tag_match.group(1).strip()
    # 6. First strong or b tag
    strong_match = re.search(r'<strong[^>]*>(.*?)</strong>|<b[^>]*>(.*?)</b>',
                             html_content, re.IGNORECASE | re.DOTALL)
    if strong_match:
        return (strong_match.group(1) or strong_match.group(2)).strip()
    # 7. First paragraph
    p_match = re.search(r'<p[^>]*>(.*?)</p>', html_content, re.IGNORECASE | re.DOTALL)
    if p_match:
        return p_match.group(1).strip()
    # 8. Cleaned first line of text after removing HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_content)
    first_line = clean_text.split('\n')[0].strip()
    if first_line:
        return first_line
    return ""


def get_html_content(source: str, request_kwargs=None) -> str:
    """
    Retrieve online HTML content.

    Args:
        source: The source, can be a URL or file path.
        request_kwargs: Request parameters for HTTP requests.
    Returns:
        str: HTML content.
    Raises:
        ValueError: If the content cannot be retrieved.
    """
    if not source:
        raise ValueError("Source cannot be empty")

    # prepare request parameters
    if request_kwargs is None:
        request_kwargs = {}
    request_kwargs.setdefault("timeout", 30)
    request_kwargs.setdefault("headers", {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    try:
        # process URL
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, **request_kwargs)
            response.raise_for_status()
            return response.text
        # process local file
        elif source.startswith('file://'):
            file_path = unquote(source[7:])
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        # process local file path
        elif os.path.exists(source):
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported source type: {source}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch content from URL: {str(e)}")
    except IOError as e:
        raise ValueError(f"Failed to read file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


def split_text_intelligently(text: str, chunk_size: int = 2048, chunk_overlap: int = 50) -> List[str]:
    """
    Split text into chunks intelligently while maintaining semantic integrity.

    Args:
        text: The text to be split into chunks
        chunk_size: Maximum number of characters per chunk
        chunk_overlap: Number of overlapping characters between adjacent chunks
    Returns:
        List[str]: List of split text chunks
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Define separator priority: first by paragraphs, then by sentences, finally by spaces
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Split the text
    chunks = text_splitter.split_text(text)
    return chunks


def is_black_image(pix, threshold=1, sample_step=10):
    """
    Determine if an image is almost completely black.

    Args:
        pix: The Pixmap object.
        threshold: The maximum RGB value to consider as black.
        sample_step: The step size for sampling pixels.
    Returns:
        bool: True if the image is black, False otherwise.
    """
    if pix.n not in [3, 4]:
        return False

    for y in range(0, pix.height, sample_step):
        for x in range(0, pix.width, sample_step):
            if pix.n == 3:
                r, g, b = pix.pixel(x, y)
            else:
                r, g, b, _ = pix.pixel(x, y)
            if r > threshold or g > threshold or b > threshold:
                return False
    return True


def get_image_from_pdf(input_file, tmp_dir):
    """
    Extract images from a PDF file, save them, and extract surrounding text.

    Args:
        input_file: Path to the PDF file.
        tmp_dir: Directory to save extracted images and JSON.
    Returns:
        list: List of dictionaries containing image path, description, and text.
    """
    doc = fitz.open(input_file)
    image_data_list = []

    for page in doc:
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if is_black_image(pix):
                pix = None
                continue

            img_rect = page.get_image_rects(xref)[0]  # Get image rectangle (x0, y0, x1, y1)

            # Ensure (img_rect + (0, 0, 50, 50)) does not exceed page bounds
            text_in_area = page.get_textbox(
                fitz.Rect(
                    img_rect.x0,
                    img_rect.y0,
                    min(img_rect.x1 + 50, page.rect.width),  # Limit x1 to page width
                    min(img_rect.y1 + 50, page.rect.height)  # Limit y1 to page height
                )
            )

            # Ensure (img_rect + (-100, -500, 500, 500)) stays within page bounds
            adjusted_rect = fitz.Rect(
                max(0, img_rect.x0 - 100),  # x0 not less than 0
                max(0, img_rect.y0 - 500),  # y0 not less than 0
                min(page.rect.width, img_rect.x1 + 500),  # x1 not more than page width
                min(page.rect.height, img_rect.y1 + 500)  # y1 not more than page height
            )
            text_ = page.get_textbox(adjusted_rect)

            # Clean up text (remove extra spaces)
            text = ' '.join(text_.split())
            description = ' '.join(text_in_area.split())

            # Set image save path
            image_path = os.path.join(tmp_dir, f"image_{xref}.png")
            if pix.n < 5:
                pix.save(image_path)
            else:
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(image_path)
                pix1 = None

            image_data = {
                "image_path": image_path,
                "description": description,
                "text": text,
            }
            image_data_list.append(image_data)
            pix = None

    # 修改json保存路径
    # output_file = os.path.join(tmp_dir, "image_descriptions.json")
    # write_json(output_file, image_data_list, "w")
    return image_data_list
