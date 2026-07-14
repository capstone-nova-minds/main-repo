# Git Workflow

This project uses a standard feature-branch workflow with `main` and `develop`
as the two long-lived branches.

## Branch roles

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code. Only updated via reviewed pull requests from `develop`. |
| `develop` | Integration branch. All feature branches merge here first. This is where day-to-day implementation happens. |
| `feature/*` | One branch per component (e.g. `feature/ocr-router-fallback`, `feature/streamlit-review-screen`). Each has a single, clear responsibility. |

## Rules

- **Never push directly to `main`.** All changes to `main` come from a pull
  request out of `develop`, after `develop` has been tested.
- **Always pull the latest `develop` before starting new work**, so your
  feature branch is based on the most current integration state.
- **Each feature branch owns one component.** Don't mix unrelated changes
  (e.g. don't fix the OCR router in a branch meant for the Streamlit review
  screen).
- **Open a pull request into `develop`** when a feature branch is ready for
  review. Do not merge your own feature branches directly without review
  when working in a team.
- Once `develop` is stable and tested, it is merged into `main` for a release.

## Feature branches in this project

- `feature/project-foundation`
- `feature/upload-api`
- `feature/streamlit-upload-screen`
- `feature/pdf-processing`
- `feature/image-preprocessing`
- `feature/ocr-router-fallback`
- `feature/arabic-ocr`
- `feature/ocr-storage`
- `feature/text-normalization`
- `feature/document-extraction-rules`
- `feature/person-extraction-rules`
- `feature/company-detection`
- `feature/name-splitting`
- `feature/validation-confidence`
- `feature/local-arabic-ner`
- `feature/entity-merge-service`
- `feature/streamlit-processing-status`
- `feature/streamlit-review-screen`
- `feature/streamlit-export-screen`
- `feature/export-json-excel`
- `feature/testing-evaluation`
- `feature/docs-demo`

## Typical flow for a contributor

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-component
# ... do the work, commit ...
git push origin feature/my-component
# open a pull request: feature/my-component -> develop
```
