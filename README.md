# trait-weaver-NLP

- clone repo 
- install uv
- install uv dependencies with uv sync

Everything from Repo path


**N.B.:** for compatibility with windows systems, aws cli is used to fetch data from OTAR.

Get data and models
- sh scripts/get_data.sh
- uv run python scripts/get_models.py

In case of import error it could be beneficial to run:
uv sync --reinstall
especially if python scripts are runned without 'python' keyword before path
