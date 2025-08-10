# ADGM-Compliant Corporate Agent with Document Intelligence

## 📌 Overview
This is an AI-powered legal assistant that helps review, validate, and prepare documentation for business incorporation and compliance within the Abu Dhabi Global Market (ADGM) jurisdiction.

The Corporate Agent:
- Accepts uploaded `.docx` legal documents.
- Verifies completeness against ADGM rules & checklist.
- Highlights red flags and inconsistencies.
- Inserts contextual comments in the `.docx` file.
- Generates a reviewed document and a structured JSON summary.

## 🚀 Features
- Upload and parse `.docx` legal documents.
- Detect missing mandatory documents based on the legal process.
- Identify legal red flags such as:
  - Wrong jurisdiction references
  - Missing signatory sections
  - Ambiguous clauses
- Add inline comments with ADGM regulation citations.
- Output both a **reviewed `.docx`** and **JSON report**.
- Simple Streamlit interface for demonstration.

## 📂 Project Files
- **examples/** → Example before/after `.docx` files and output JSON  
- **demo/** → Screenshot of app working

## 📥 How to Run
1. Install requirements:
2. Run the app:
3. Steps in the app:
   - Upload `adgm_reference.docx` in the reference upload section
   - Upload `examples/before_review.docx` in the review upload section
   - Click Process to see results and download outputs

## 🖼 Demo Screenshot
![Demo Screenshot](demo/screenshot.png)
