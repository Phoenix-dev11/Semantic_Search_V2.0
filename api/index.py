from fastapi import FastAPI

# Create FastAPI app for Vercel
app = FastAPI(title="Semantic Search API",
              description="AI-powered semantic search for company data",
              version="1.0.0")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Semantic Search API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "semantic_search_api",
        "version": "1.0.0"
    }


# Create handler function for Vercel
def handler(request):
    """Vercel serverless function handler"""
    return app(request.scope, request.receive, request.send)
