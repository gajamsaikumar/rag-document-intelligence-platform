import os
from pathlib import Path

from fastapi import FastAPI,UploadFile,File,Form,HTTPException,Request
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import Dict,Any,List,Optional

from src.document_ingestion.data_ingestion import (DocHandler,
                                                   DocCompare,
                                                   ChatIngestor,
                                                   )
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparision import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter,read_pdf_via_handler

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
app=FastAPI(title="Document Portal API",version="0.1")
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#serve static and templates
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")#To check my api is working or not
def health() -> Dict[str,str]:
    return {"status":"ok","service":"document-portal"}


@app.post("/analyze")
async def analyze_document(file:UploadFile=File(...)) -> Any:
    try:
        doc_handler=DocHandler()
        #save the pdf in the current dir in the form of session
        saved_path=doc_handler.save_pdf(FastAPIFileAdapter(file))
        text=read_pdf_via_handler(doc_handler,saved_path)

        doc_analyzer=DocumentAnalyzer()
        result=doc_analyzer.analyze_document(text)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        #status_code=500(which means internal server errror)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/compare")
async def compare_documents(reference:UploadFile=File(...),actual:UploadFile=File(...)) -> Any:
    try:
        doc_compare=DocCompare()
        ref_path,act_path=doc_compare.save_uploaded_files(FastAPIFileAdapter(reference),FastAPIFileAdapter(actual))
        _=ref_path,act_path
        combined_text=doc_compare.combine_docs()

        comp=DocumentComparatorLLM()
        df=comp.compare_documents(combined_text)
        return {"rows":df.to_dict(orient="records"),"session_id":doc_compare.session_id}
    except HTTPException:
        raise
    except Exception as e:
        #status_code=500(which means internal server errror)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
    
@app.post("/chat/index")
async def chat_build_index(
                            files: List[UploadFile] = File(...),
                            session_id: Optional[str] = Form(None),
                            use_session_dirs: bool = Form(True),
                            chunk_size: int = Form(1000),
                            chunk_overlap: int = Form(200),
                            k: int = Form(5),
                        )->Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriever(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        #status_code=500(which means internal server errror)
        raise HTTPException(status_code=500, detail=f"Chat built index failed: {e}")

@app.post("/chat/query")
async def chat_query(question: str = Form(...),
                    session_id: Optional[str] = Form(None),
                    use_session_dirs: bool = Form(True),
                    k: int = Form(5),
                    ) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        #status_code=500(which means internal server errror)
        raise HTTPException(status_code=500, detail=f"Chat query failed: {e}")
    
#command for executing the fast api
#1. cd api/
#2. uvicorn main:app --reload