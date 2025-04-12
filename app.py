from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("health_model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, height: int = Form(...), weight: int = Form(...)):
    prediction = model.predict([[height, weight]])
    result = "✅ Healthy" if prediction[0] == 1 else "❌ Not Healthy"
    detail = f" At height {height}cm, And Weight {weight}kg"
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "detail": detail})
