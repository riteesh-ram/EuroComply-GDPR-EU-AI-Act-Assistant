from fastapi import FastAPI
from API.routes.appRoutes import router as complaince_bot_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Complaince RAG Bot",
    description="An API for GDPR and EU AI compliance using RAG.",
    version="1.0.0"
)

# Root endpoint to check if API is running
@app.get("/")
async def root():
    return {"message": "Complaince Bot API is running!"}

app.include_router(complaince_bot_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)