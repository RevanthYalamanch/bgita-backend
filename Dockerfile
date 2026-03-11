# Use the official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code into the container
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]