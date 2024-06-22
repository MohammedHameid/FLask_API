FROM python:3.11-slim

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and the post-install script
COPY requirements.txt .
COPY post_install.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the post-install script to download the Spacy model
RUN python post_install.py

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set the entrypoint for your application
CMD ["python", "your_script.py"]
