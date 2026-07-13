# User Guide

## 1. Upload document

Open the app at [http://localhost:8501](http://localhost:8501), go to the
**1_Upload** page, and choose a PDF, JPG, JPEG, or PNG scan of a court
attachment order. Click **Upload**. Note the **Document ID** shown -- it
identifies this document for every later step.

## 2. Start processing

Go to **2_Processing_Status** and click **Start Processing**. This runs
OCR, text normalization, rule-based extraction, and local Arabic NER. The
page shows: which OCR engine was used, OCR status and quality, whether the
Tesseract fallback was needed, NER status and engine, and how many person
and organization entities were found. If OCR quality is low, a warning in
Arabic ("منخفضة جودة قراءة النص، يرجى مراجعة البيانات بعناية.") tells you
to review the data carefully.

## 3. Review extracted information

Go to **3_Review_Validation**. The document-level fields (court name, case
number, document number, date) appear as editable text boxes, with a
warning icon next to any field that needs review. The persons/companies
table is editable: you can add rows, delete rows, or fix values directly.
Records that came from NER without a National ID are flagged with:
"تم اكتشاف هذا الاسم بواسطة NER ويحتاج مراجعة."

## 4. Edit fields

Correct any wrong or missing values directly in the form and table. Split
rows manually if a grouped name wasn't separated correctly, or merge/edit
duplicate rows.

## 5. Save reviewed result

Click **Save Reviewed Result**. This is the mandatory human-review step --
nothing is considered "approved" until this is saved. You'll see a success
message once it's stored.

## 6. Export JSON

Go to **4_Export** and click **Prepare JSON Export**, then
**Download JSON**. If you saved a reviewed result, the export uses it;
otherwise it uses the raw extracted data and is marked as not reviewed.

## 7. Export Excel

On the same page, click **Prepare Excel Export**, then **Download Excel**.
The spreadsheet has one row per person/company, with the document-level
fields repeated on every row.
