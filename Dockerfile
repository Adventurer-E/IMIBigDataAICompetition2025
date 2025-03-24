# Use an official Python runtime as a parent image
FROM python:3.12.1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Copy the source code into the container (only the src/ directory)
COPY src/ /app/src

# Define the entrypoint to run the main script
ENTRYPOINT [ "python", "-u", "src/main.py" ]
