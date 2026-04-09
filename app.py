from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from chatbot import ask_bot  
from chatbot import init_bot  

# -----------------------------------------------------
# 🚀 FastAPI App Initialization
# -----------------------------------------------------
app = FastAPI(
    title="Finance RAG Chatbot API",
    description="Ask tax questions — Retrieval-Augmented LLM Agent",
    version="1.0.0",
)

# -----------------------------------------------------
# 🌐 CORS (Allow front-end / HF demo UI / apps)
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# 🛠 Logger
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance-api")

# -----------------------------------------------------
# 🟢 Request Model
# -----------------------------------------------------
class Query(BaseModel):
    question: str

# -----------------------------------------------------
# 🚦 Startup — warm the bot (optional)
# -----------------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("🔄 Initializing Finance RAG bot...")
        init_bot()  # loads embeddings/vectorstore once
        logger.info("✅ Finance bot ready!")
    except:
        logger.exception("❌ Failed to initialize bot")


# -----------------------------------------------------
# 🏠 Root endpoint
# -----------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Finance RAG Chatbot API 💰",
        "endpoints": {
            "/ask": "POST — ask tax-related questions",
            "/health": "GET — system health check"
        }
    }

# -----------------------------------------------------
# ❤️ Health Check
# -----------------------------------------------------
@app.get("/health")
async def health():
    return {
        "service": "finance-rag-bot",
        "status": "OK",
        "version": "1.0.0"
    }

# -----------------------------------------------------
# ❓ Question → Answer
# -----------------------------------------------------
@app.post("/ask")
async def ask_api(payload: Query):
    try:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        logger.info(f"🙋 User asked: {question}")
        answer = ask_bot(question)

        return {
            "question": question,
            "answer": answer
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error while processing request")
        raise HTTPException(status_code=500, detail=str(e))
