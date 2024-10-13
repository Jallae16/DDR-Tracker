# Use the official Python image from Docker Hub as the base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install system dependencies and libraries (including libGL for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's code into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "your_app.py"]
