# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application source code and MLflow runs into the container
COPY ./src /app/src
COPY ./mlruns /app/mlruns

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable to tell MLflow where to find runs
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Run main.py when the container launches
# Use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
