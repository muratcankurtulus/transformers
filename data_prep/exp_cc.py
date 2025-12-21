import gzip
import os
import re
import sys
import unicodedata

from fasttext import load_model  # From fasttext-langdetect

# --- Configuration ---
MODEL_PATH = "lid.176.ftz"
TARGET_LANG = "__label__en"
CONFIDENCE_THRESHOLD = 0.9
OUTPUT_CORPUS_FILE = "my_corpus_en.txt"
WET_FILES = [
    "./wet/CC-MAIN-20251005114239-20251005144239-00000.warc.wet.gz",
    #    "./wet/CC-MAIN-20251005114239-20251005144239-00001.warc.wet.gz",
]

# --- Pre-compile regex for efficiency ---
bracket_re = re.compile(r"\[.*?\]")
whitespace_re = re.compile(r"\s+")
boilerplate_phrases = [
    "copyright ©",
    "all rights reserved",
    "lorem ipsum",
    "subscribe to our newsletter",
    "sign in",
    "log in",
    "javascript must be enabled",
    "enable javascript",
    "view our privacy policy",
    "terms of use",
]


def clean_text(text):
    """
    Applies final cleaning to a block of text *after* it has been
    confirmed to be in the target language.
    """
    # 1. Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # 2. Lowercase (optional, but good for most tokenizers)
    text = text.lower()
    # 3. Remove text between square brackets
    text = bracket_re.sub("", text)

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 4. Filter out boilerplate lines
        if any(phrase in line for phrase in boilerplate_phrases):
            continue

        # 5. Filter out very short lines
        if len(line.split()) < 5:
            continue

        # 6. Filter lines that are mostly non-alphabetic
        # This is a good heuristic for "junk" lines (e.g., symbol lists)
        alnum_chars = sum(c.isalnum() for c in line)
        if len(line) > 0 and (alnum_chars / len(line)) < 0.6:
            continue

        cleaned_lines.append(line)

    if not cleaned_lines:
        return None

    # 7. Collapse all lines and whitespace into a single line of text
    text = " ".join(cleaned_lines)
    text = whitespace_re.sub(" ", text).strip()

    return text


def process_wet_file_stream(input_gz_file, output_txt_file, ft_model):
    """
    Reads a .wet.gz file line-by-line (streaming) to save memory.
    Detects language and cleans text for each document.
    """
    print(f"Streaming from {input_gz_file}...")
    doc_count = 0
    eng_count = 0

    with gzip.open(input_gz_file, "rt", encoding="utf-8", errors="ignore") as f_in:
        with open(output_txt_file, "a", encoding="utf-8") as f_out:
            current_doc_lines = []
            in_document = False

            for line in f_in:
                # WARC headers mark the *start* of a new document
                if line.startswith("WARC-Type: conversion"):
                    if current_doc_lines:
                        doc_count += 1
                        text_content = "\n".join(current_doc_lines)

                        # --- 1. Language Detection ---
                        pred_text = text_content.replace("\n", " ").replace("\r", " ")

                        if len(pred_text) > 25:  # Don't check tiny fragments
                            prediction = ft_model.predict(pred_text, k=1)
                            lang = prediction[0][0]
                            confidence = prediction[1][0]

                            # --- 2. Filtering ---
                            if lang == TARGET_LANG and confidence >= CONFIDENCE_THRESHOLD:
                                eng_count += 1

                                # --- 3. Cleaning ---
                                cleaned = clean_text(text_content)

                                # --- 4. Writing ---
                                if cleaned and len(cleaned) > 150:  # Min length for a *clean* doc
                                    f_out.write(cleaned)
                                    f_out.write("\n")  # Write one clean document per line

                        if doc_count % 1000 == 0:
                            print(f"  ...processed {doc_count} docs, saved {eng_count} English.", end="\r")

                    current_doc_lines = []
                    in_document = True

                elif in_document and not line.startswith("WARC/"):
                    current_doc_lines.append(line.strip())

            if current_doc_lines:
                text_content = "\n".join(current_doc_lines)
                pred_text = text_content.replace("\n", " ").replace("\r", " ")
                if len(pred_text) > 25:
                    prediction = ft_model.predict(pred_text, k=1)
                    lang = prediction[0][0]
                    confidence = prediction[1][0]
                    if lang == TARGET_LANG and confidence >= CONFIDENCE_THRESHOLD:
                        eng_count += 1
                        cleaned = clean_text(text_content)
                        if cleaned and len(cleaned) > 150:
                            f_out.write(cleaned)
                            f_out.write("\n")

    print(f"\nFinished {input_gz_file}. Found {doc_count} total docs, wrote {eng_count} clean English docs.")


def main():
    # 1. Check for and load the FastText model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: FastText model not found at {MODEL_PATH}")
        print("Please download it from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
        sys.exit(1)

    print("Loading language detection model...")
    try:
        load_model.SUPPRESS_WARNINGS = True
        ft_model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have the correct model file and 'fasttext-langdetect' is installed.")
        sys.exit(1)

    print("Model loaded. Starting processing...")

    # 2. Clear the output file first
    open(OUTPUT_CORPUS_FILE, "w").close()

    # 3. Process each file
    for wet_file in WET_FILES:
        if not os.path.exists(wet_file):
            print(f"Warning: File not found, skipping: {wet_file}")
            continue
        process_wet_file_stream(wet_file, OUTPUT_CORPUS_FILE, ft_model)

    print("\n--- All done! ---")
    print(f"Clean English corpus is ready in: {OUTPUT_CORPUS_FILE}")


if __name__ == "__main__":
    main()
