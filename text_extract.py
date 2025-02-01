import fitz 
import os
from pathlib import Path

def clean_text(text):
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    text = ' '.join(text.split())
    
    text = text.replace('\n', '\n\n')
    
    return text

def process_pdf(pdf_path, output_dir):
    try:
        output_path = output_dir / f"{pdf_path.stem}.txt"
        
        doc = fitz.open(pdf_path)
        
        text = []
        for page in doc:
            text.append(page.get_text())
        
        full_text = '\n'.join(text)
        cleaned_text = clean_text(full_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        doc.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return False

def batch_process_pdfs(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob('*.pdf'))
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to process")
   
    successful = 0
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"Processing {i}/{total_files}: {pdf_file.name}")
        if process_pdf(pdf_file, output_path):
            successful += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful}/{total_files} files")
    print(f"Text files saved to: {output_path}")

if __name__ == "__main__":
    input_directory = "female-nominees"
    output_directory = "female-nominees-processed"
    
    batch_process_pdfs(input_directory, output_directory)