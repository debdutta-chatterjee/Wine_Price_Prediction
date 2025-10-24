from fastapi import FastAPI
import os 
import numpy as np
import pandas as pd
from src.pipeline.predict import PredictionPipeline
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.staticfiles import StaticFiles # Import for serving static files

app = FastAPI(title="Wine Quality Predictor")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/', response_class=HTMLResponse, summary="Home Page")
async def home_page(request: Request):
    """
    Route to display the home page (index.html).
    Equivalent to the Flask '@app.route('/', methods=['GET'])'.
    """
    # The dictionary passed to TemplateResponse contains the context variables, 
    # and 'request' is mandatory for Jinja2 in FastAPI.
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/train', summary="Run Training Pipeline")
async def training():
    """
    Route to trigger the training pipeline via a system call.
    Equivalent to the Flask '@app.route('/train', methods=['GET'])'.
    
    NOTE: In production FastAPI applications, using os.system() is generally
    discouraged. It's better to use a dedicated background task runner (like Celery)
    or the 'subprocess' module for better control and non-blocking execution.
    """
    try:
        print("--- Triggering training script: python main.py ---")
        # Executes the training script synchronously
        os.system("python main.py")
        return "Training Successful! Check server logs for output from main.py."
    except Exception as e:
        return f"Training Failed: {e}"


@app.get('/predict', response_class=HTMLResponse, summary="Prediction Form")
@app.post('/predict', response_class=HTMLResponse, summary="Submit Prediction Data")
async def predict_route(request: Request):
    """
    Handles both GET (display form) and POST (process form) requests on /predict.
    This replaces the combined Flask 'index()' function.
    """
    # If it is a GET request, just render the form (index.html)
    if request.method == 'GET':
        return templates.TemplateResponse("index.html", {"request": request})

    # If it is a POST request, process the prediction
    if request.method == 'POST':
        try:
            # Retrieve form data asynchronously
            form_data = await request.form()
            
            # Read and convert inputs (using .get() for safety)
            fixed_acidity = float(form_data.get('fixed_acidity', 0.0))
            volatile_acidity = float(form_data.get('volatile_acidity', 0.0))
            citric_acid = float(form_data.get('citric_acid', 0.0))
            residual_sugar = float(form_data.get('residual_sugar', 0.0))
            chlorides = float(form_data.get('chlorides', 0.0))
            free_sulfur_dioxide = float(form_data.get('free_sulfur_dioxide', 0.0))
            total_sulfur_dioxide = float(form_data.get('total_sulfur_dioxide', 0.0))
            density = float(form_data.get('density', 0.0))
            pH = float(form_data.get('pH', 0.0))
            sulphates = float(form_data.get('sulphates', 0.0))
            alcohol = float(form_data.get('alcohol', 0.0))

            # Prepare data for prediction
            data = [
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, 
                pH, sulphates, alcohol
            ]
            
            data_np = np.array(data).reshape(1, 11)
            
            # Run prediction
            obj = PredictionPipeline()
            predict_result = obj.predict(data_np)

            # Render the results page
            return templates.TemplateResponse(
                'results.html', 
                {"request": request, "prediction": predict_result}
            )

        except Exception as e:
            # Error handling for missing form fields or failed conversions
            print(f'The Exception message is: {e}')
            # Render an error template or return a simple error response
            # Note: Always return a TemplateResponse if the route is defined to return HTMLResponse
            return templates.TemplateResponse(
                'error.html', 
                {"request": request, "error_message": "An error occurred during prediction. Please check inputs."},
                status_code=500
            )


# if __name__ == "__main__":
# 	uvicorn.run(app, host="0.0.0.0", port=8080)