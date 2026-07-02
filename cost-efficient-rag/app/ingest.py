import fitz
import os
from html.parser import HTMLParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

DOCUMENT_PATH = "data/documents"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.ignore = False

    def handle_starttag(self, tag, attrs):
        if tag in ["script", "style"]:
            self.ignore = True

    def handle_endtag(self, tag):
        if tag in ["script", "style"]:
            self.ignore = False

    def handle_data(self, data):
        if not self.ignore:
            self.result.append(data)

    def get_text(self):
        return "".join(self.result)


def read_pdf(file_path):
    """Read text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def read_html(file_path):
    """Read text from an HTML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text()


def read_md(file_path):
    """Read text from a Markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents():
    documents = []

    if not os.path.exists(DOCUMENT_PATH):
        os.makedirs(DOCUMENT_PATH)
        return documents

    for file in os.listdir(DOCUMENT_PATH):
        ext = os.path.splitext(file)[1].lower()
        if ext not in [".pdf", ".html", ".htm", ".md"]:
            continue

        full_path = os.path.join(DOCUMENT_PATH, file)
        print(f"Reading: {file}")

        try:
            if ext == ".pdf":
                text = read_pdf(full_path)
            elif ext in [".html", ".htm"]:
                text = read_html(full_path)
            elif ext == ".md":
                text = read_md(full_path)
            else:
                continue

            chunks = splitter.split_text(text)

            documents.append({
                "filename": file,
                "chunks": chunks
            })

            print(f"[OK] Successfully read {file}")

        except Exception as e:
            print(f"[ERROR] Error reading {file}")
            print(e)

    return documents