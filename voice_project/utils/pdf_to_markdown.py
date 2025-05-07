"""
PDF to Markdown conversion module for the Voice Assistant.

This module provides functionality to convert PDF research papers into markdown
format for easier processing by the language models. It extracts text from PDF
files and saves them as markdown files with sequential numbering.

Dependencies:
    - pdfplumber: For PDF text extraction
    - os: For file system operations
"""

import pdfplumber
import os

def pdf_to_markdown(pdf_path, article_number):
    """
    Convert a PDF file to markdown format.
    
    This function:
    1. Opens the PDF file
    2. Extracts text from each page
    3. Combines the text
    4. Saves it as a markdown file
    
    Args:
        pdf_path (str): Path to the input PDF file
        article_number (int): Number to use in the output filename
    
    Note:
        The output file will be saved as '{article_number}.md' in the
        datasets/research_papers directory
    """
    content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content += page.extract_text()
    with open(f"datasets/research_papers/{article_number}.md", "w") as f:
        f.write(content)

# Convert all PDFs in the research papers directory
path = "datasets/research_papers"
files = os.listdir(path)

for idx, file in enumerate(files):
    pdf_to_markdown(os.path.join(path, file), (idx+1))
