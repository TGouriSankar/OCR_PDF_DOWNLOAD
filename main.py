import logging
import time
from pathlib import Path
import contextlib
import gradio as gr
import nltk
import torch
from pdf2text import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_here = Path(__file__).parent

nltk.download("stopwords")  # TODO=find where this requirement originates from

def load_uploaded_file(file_obj, temp_dir: Path = None):
    """
    load_uploaded_file - process an uploaded file
    Args:
        file_obj (POTENTIALLY list): Gradio file object inside a list
    Returns:
        str, the uploaded file contents
    """

    # check if mysterious file object is a list
    if isinstance(file_obj, list):
        file_obj = file_obj[0]
    file_path = Path(file_obj.name)

    if temp_dir is None:
        _temp_dir = _here / "temp"
    _temp_dir.mkdir(exist_ok=True)

    try:
        pdf_bytes_obj = open(file_path, "rb").read()
        temp_path = temp_dir / file_path.name if temp_dir else file_path
        # save to PDF file
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes_obj)
        logging.info(f"Saved uploaded file to {temp_path}")
        return str(temp_path.resolve())

    except Exception as e:
        logging.error(f"Trying to load file with path {file_path}, error: {e}")
        print(f"Trying to load file with path {file_path}, error: {e}")
        return None

def convert_PDF(
    pdf_obj,
    language: str = "en",
    max_pages=20,
):
    """
    convert_PDF - convert a PDF file to text
    Args:
        pdf_bytes_obj (bytes): PDF file contents
        language (str, optional): Language to use for OCR. Defaults to "en".
    Returns:
        str, the PDF file contents as text
    """
    # clear local text cache
    rm_local_text_files()
    global ocr_model
    st = time.perf_counter()
    if isinstance(pdf_obj, list):
        pdf_obj = pdf_obj[0]
    file_path = Path(pdf_obj.name)
    if not file_path.suffix == ".pdf":
        logging.error(f"File {file_path} is not a PDF file")

        html_error = f"""
        <div style="color: red; font-size: 20px; font-weight: bold;">
        File {file_path} is not a PDF file. Please upload a PDF file.
        </div>
        """
        return "File is not a PDF file", html_error, None

    conversion_stats = convert_PDF_to_Text(
        file_path,
        ocr_model=ocr_model,
        max_pages=max_pages,
    )
    converted_txt = conversion_stats["converted_text"]
    num_pages = conversion_stats["num_pages"]
    was_truncated = conversion_stats["truncated"]
    # if alt_lang: # TODO: fix this

    rt = round((time.perf_counter() - st) / 60, 2)
    print(f"Runtime: {rt} minutes")
    html = ""
    if was_truncated:
        html += f"<p>WARNING - PDF was truncated to {max_pages} pages</p>"
    html += f"<p>Runtime: {rt} minutes on CPU for {num_pages} pages</p>"

    _output_name = f"RESULT_{file_path.stem}_OCR.txt"
    with open(_output_name, "w", encoding="utf-8", errors="ignore") as f:
        f.write(converted_txt)

    return converted_txt, html, _output_name

if __name__ == "__main__":
    logging.info("Starting app")

    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU status: {use_GPU}")
    logging.info("Loading OCR model")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )

    pdf_obj = _here / "try_example_file.pdf"
    pdf_obj = str(pdf_obj.resolve())
    _temp_dir = _here / "temp"
    _temp_dir.mkdir(exist_ok=True)

    logging.info("starting demo")
    demo = gr.Blocks()

    with demo:
        gr.Markdown("# PDF to Text")
        gr.Markdown(
            "A basic demo of pdf-to-text conversion using OCR from the [doctr](https://mindee.github.io/doctr/index.html) package"
        )
        gr.Markdown("---")

        with gr.Column():
            gr.Markdown("## Load Inputs")
            gr.Markdown("Upload your own file & replace the default. Files should be < 10MB to avoid upload issues - search for a PDF compressor online as needed.")
            gr.Markdown(
                "_If no file is uploaded, a sample PDF will be used. PDFs are truncated to 20 pages._"
            )

            uploaded_file = gr.File(
                label="Upload a PDF file",
                file_count="single",
                type="filepath",
                value=str(_here / "try_example_file.pdf"),
            )

            gr.Markdown("---")

        with gr.Column():
            gr.Markdown("## Convert PDF to Text")
            convert_button = gr.Button("Convert PDF!", variant="primary")
            out_placeholder = gr.HTML("<p><em>Output will appear below:</em></p>")
            gr.Markdown("### Output")
            OCR_text = gr.Textbox(
                label="OCR Result", placeholder="The OCR text will appear here"
            )
            text_file = gr.File(
                label="Download Text File",
                file_count="single",
                type="filepath",
                interactive=False,
            )

        convert_button.click(
            fn=convert_PDF,
            inputs=[uploaded_file],
            outputs=[OCR_text, out_placeholder, text_file],
        )
    demo.launch()
