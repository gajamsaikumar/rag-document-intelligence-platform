import importlib.metadata
packages = [
    "langchain",
    "python-dotenv",
    "ipykernel",
    "langchain_groq",
    "langchain_google_genai",
    "langchain-community",
    "faiss-cpu",
    "structlog",
    "PyMuPDF",
    "pylint",
    "langchain-core",
    "pytest",
    "streamlit",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "docx2txt",
    "cfn-lint",
    "pre-commit",
    "deepeval",
    "openai",
    "unstructured",
    "pillow",
    "opencv-python-headless",
    "pytesseract",
    "fastapi-users[sqlalchemy]",
    "fastapi-users-db-sqlalchemy",
    "sqlalchemy>=2",
    "aiosqlite",
    "passlib[bcrypt]",
    "pyjwt",
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")

# # serve static & templates
# app.mount("/static", StaticFiles(directory="../static"), name="static")
# templates = Jinja2Templates(directory="../templates")