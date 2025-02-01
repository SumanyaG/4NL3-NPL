This dataset has been curated using publicly available data on the [US Supreme Court Nomination Hearings](https://www.govinfo.gov/collection/supreme-court-nomination-hearings). It contains two main categories - female and male supreme court nominees.

Three women justices, namely, Justice Sandra Day O'Connor, Justice Ruth Bader Ginsburg and Justics Sonia Sotomayor have been included in the former category and four men justices, namely, Justice Samuel Alito, Justice Anthony Kennedy, Justice John Roberts and Justice Antonin Scalia have been included in the latter category.

For each nomination hearing, the following files have been included as individual documents:
- Statements of Committee Members (including Prepared Statements)
- Statements and/or Testimony by the Nominee
- Questions and Answers
All of the aforementioned files can be found at https://www.govinfo.gov/collection/supreme-court-nomination-hearings.

This folder contains the following files and subfolders:
- **text_extract.py**: Python script to extract data from the PDF files and store them as TXT files
- **female-nominees**: folder containing 136 PDFs related to the nomination hearings of the three women justices
- **female-nominees-processed**: folder containing 136 TXT files each of which corresponds to an individual document
- **male-nominees**: folder containing 147 PDFs related to the nomination hearings of the four men justices
- **male-nominees-processed**: folder containing 147 TXT files each of which corresponds to an individual document