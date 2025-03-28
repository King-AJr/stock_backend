from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import app

app = FastAPI(title="Stock")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(app.router, prefix="/api", tags=["api"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Youtube Assistant API"} 
