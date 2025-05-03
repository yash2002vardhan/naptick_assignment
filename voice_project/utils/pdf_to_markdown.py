import pdfplumber
import os

def pdf_to_markdown(pdf_path, article_number):
    content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content += page.extract_text()
    with open(f"datasets/research_papers/{article_number}.md", "w") as f:
        f.write(content)


path = "datasets/research_papers"
files = os.listdir(path)

for idx, file in enumerate(files):
    pdf_to_markdown(os.path.join(path, file), (idx+1))
