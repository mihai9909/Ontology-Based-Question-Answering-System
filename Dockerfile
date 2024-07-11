FROM nvcr.io/nvidia/pytorch:23.09-py3

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y

# Install the required packages
RUN pip install -r requirements.txt

CMD ["./start"]

