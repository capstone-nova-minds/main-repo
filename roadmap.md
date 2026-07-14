# Working Block Agendas

## 4. 8-July / Wednesday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 45 min | Review the stakeholder’s court-order extraction brief | Confirmed that the solution must process Jordanian Arabic court attachment orders locally | ✅ |
| 45 min | Define the project problem and target user | Target users identified as Operations employees who manually review court documents | ✅ |
| 60 min | Define the required extracted fields | Document fields and person/company fields were documented | ✅ |
| 45 min | Review OCR, NER, and rule-based extraction concepts | Discussed the role of OCR, Arabic NER, regex validation, and human review | ✅ |
| 30 min | Define the purpose of Human Review | Confirmed that extracted data must be reviewed and corrected before export | ✅ |
| 45 min | Define input document types | Added support requirements for PDF, JPG, PNG, printed Arabic, handwritten Arabic, and multi-page documents | ✅ |
| 45 min | Define output formats | Confirmed that approved results will be exported to JSON and Excel | ✅ |
| 30 min | Define included and excluded project scope | Excluded customer lookup, wallet-freezing APIs, dashboards, and external integrations | ✅ |
| 30 min | Review data-privacy requirements | Confirmed that names and National IDs must be handled locally without cloud AI APIs | ✅ |
| 30 min | Draft the initial AI project components | Divided the solution into upload, OCR, extraction, validation, review, and export components | ✅ |

## 5. 9-July / Thursday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 45 min | Finalize the project concept and description | Defined the project as an AI-powered Jordanian Court Order Extraction system | ✅ |
| 60 min | Convert the stakeholder brief into technical requirements | Mapped each business requirement to a technical AI component | ✅ |
| 45 min | Define the local-system constraints | Confirmed no OpenAI, Gemini, Claude, or other cloud AI APIs inside the application | ✅ |
| 60 min | Design the initial technical architecture | Defined Streamlit frontend, FastAPI backend, local OCR, Arabic NER, validation, and export flow | ✅ |
| 45 min | Define the document-processing pipeline | Upload → preprocessing → OCR → extraction → validation → human review → export | ✅ |
| 45 min | Assign initial team responsibilities | Backend, OCR, extraction, and frontend responsibilities were divided among team members | ✅ |
| 60 min | Create the initial project folder structure | Added backend, frontend, data, tests, Docker, and shared configuration directories | ✅ |
| 45 min | Prepare the first implementation plan | Created a daily implementation plan beginning with upload and OCR functionality | ✅ |
| 30 min | Prepare a request for sample stakeholder documents | Requested realistic samples to guide development and validation | ✅ |
| 30 min | Review synthetic-data alternatives | Planned to generate fictional court documents if real samples were delayed | ✅ |

## 6. 10-July / Friday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 45 min | Review the Git collaboration workflow | Discussed clone, pull, branch creation, commit, push, and pull-request steps | ✅ |
| 45 min | Create a separate collaboration repository | Prepared a working repository that would not affect the original submission repository | ✅ |
| 60 min | Define team branches | Created separate responsibility areas for backend, extraction, frontend, and OCR | ✅ |
| 45 min | Assign files to each team member | Mapped project files to each branch to reduce merge conflicts | ✅ |
| 30 min | Confirm branch responsibilities | Hadeel: Backend, Omar: Extraction, Rawan: Frontend, Ibrahim: OCR | ✅ |
| 60 min | Prepare project setup instructions | Documented virtual environment, dependencies, folder structure, and startup commands | ✅ |
| 45 min | Prepare a Claude Code implementation prompt | Instructed Claude to continue the team’s existing work without changing setup files or ports | ✅ |
| 30 min | Protect project configuration files | Confirmed that Docker, ports, requirements, and setup files should not be modified without approval | ✅ |
| 45 min | Review pull and update procedures | Confirmed how each team member should download the latest team changes before continuing work | ✅ |
| 30 min | Review conflict-prevention practices | Agreed to keep responsibilities separated and merge through pull requests | ✅ |

## 7. 11-July / Saturday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 45 min | Review the upload and processing workflow | Confirmed that uploaded documents were stored and processed successfully | ✅ |
| 60 min | Improve document-level extraction | Improved Court Name, Case Number, Document Number, and Document Date extraction | ✅ |
| 45 min | Fix Case Number and Document Number mixing | Confirmed separate values such as `1202/2026` and `UW-2026-0002` | ✅ |
| 60 min | Improve the Processing Status frontend | Added extraction score, required fields, review flags, and person count | ✅ |
| 45 min | Add extracted document and person tables | Displayed document fields and extracted person/company records clearly | ✅ |
| 30 min | Add Advanced Technical Details | Moved OCR and NER summaries into a hidden expander | ✅ |
| 45 min | Improve OCR quality display | Changed the OCR result display to High, Medium, and Low quality levels | ✅ |
| 60 min | Investigate person/company extraction errors | Identified missing companies, mixed names, and incorrect fallback-generated records | ✅ |
| 45 min | Review structured target-list extraction | Confirmed that the structured people/entities section should have the highest extraction priority | ✅ |
| 60 min | Generate synthetic testing documents | Created fictional court documents with Arabic names, 11-digit National IDs, and company registration numbers | ✅ |

## 8. 12-July / Sunday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 30 min | Review current OCR engine routing | Confirmed EasyOCR as primary and Tesseract as fallback | ✅ |
| 45 min | Inspect the PaddleOCR placeholder implementation | Confirmed that `paddleocr_engine_stub.py` was not yet a real OCR engine | ✅ |
| 60 min | Create a Python 3.11 Paddle environment | Created and activated `.venv-paddle` because Python 3.14 was not supported | ✅ |
| 45 min | Install and verify PaddlePaddle | Installed PaddlePaddle and successfully verified CPU execution | ✅ |
| 45 min | Investigate Paddle GPU compatibility | Detected the NVIDIA RTX 3050 GPU and reviewed CUDA compatibility | ✅ |
| 30 min | Resolve installation storage issues | Cleaned disk space after the installation failed due to insufficient storage     | ✅ |
| 45 min | Install NVIDIA CUDA runtime packages | Installed CUDA runtime, cuBLAS, and cuDNN packages in the Paddle environment      | ✅ |
| 45 min | Troubleshoot the missing CUDA DLL | Investigated the `cublas64_13.dll` runtime error       | ✅ |
| 30 min | Configure the correct VS Code interpreter | Changed VS Code from `.venv-1` to `.venv-paddle`      | ✅ |
| 45 min | Run the application on CPU | Set `PADDLEOCR_DEVICE=cpu` while GPU configuration remained under investigation         | ✅ |
| 45 min | Start the FastAPI backend and Streamlit frontend | Successfully launched the backend and Streamlit application         | ✅ |
| 60 min | Test OCR, NER, extraction, and review workflow | Processed sample documents and inspected OCR quality and extracted JSON results       | ✅ |

## 9. 13-July / Sunday’s Working Block Agenda

| Time | Activity | Notes / Updates | Completed |
|---|---|---|---|
| 45 min | Review the PaddleOCR implementation status                 | Confirmed that PaddleOCR was successfully loading the Arabic detection and recognition models locally on CPU                                        | ✅ |
| 60 min | Resolve PaddlePaddle runtime compatibility issues          | Investigated Windows oneDNN/PIR errors and installed a compatible PaddlePaddle version in the Python 3.11 environment                               | ✅ |
| 45 min | Replace the PaddleOCR placeholder with a functional engine | Updated `paddleocr_engine_stub.py` while preserving the existing class name required by the OCR router                                              | ✅ |
| 45 min | Add the missing OCR base interface                         | Created and verified `base_ocr_engine.py` to provide a common interface for local OCR engines                                                       | ✅ |
| 60 min | Test Arabic PaddleOCR on a synthetic court document        | Successfully extracted Arabic court names, document labels, person names, organization names, and document text                                     | ✅ |
| 45 min | Analyze missing structured numeric fields                  | Confirmed that the Arabic recognition model was not reliably detecting case numbers, dates, National IDs, and Latin document codes                  | ✅ |
| 60 min | Test the Latin and numeric PaddleOCR model                 | Successfully detected values including `1201/2026`, `W-2026-0101`, and `10/07/2026` using `latin_PP-OCRv5_mobile_rec`                               | ✅ |
| 60 min | Implement Arabic and Latin dual-pass OCR                   | Configured the system to run Arabic and Latin OCR passes and merge results using bounding-box positions                                             | ✅ |
| 45 min | Improve OCR field-label normalization                      | Separated joined labels such as `رقمالكتاب`, `الرقمالوطني`, and `نوع الشخصفرد` into readable structured text                                        | ✅ |
| 45 min | Investigate duplicate OCR identifiers                      | Identified duplicate variants such as `12012026`, `20260101`, `10/07`, and `10072026` generated from the same document rows                         | ✅ |
| 45 min | Confirm the hybrid extraction architecture                 | Confirmed that PaddleOCR provides document evidence, CAMeLBERT extracts people and organizations, and regex rules validate structured fields        | ✅ |
| 45 min | Prepare the complete Claude Code implementation prompt     | Prepared detailed instructions to add identifier deduplication, targeted numeric crops, OCR routing, validation, testing, and human-review accuracy | ✅ |

