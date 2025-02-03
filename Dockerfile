# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Ensure the model file is copied correctly
COPY grapevine_xception_model.keras /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 9696

# Command to run the Flask app
CMD ["python", "predict.py"]
