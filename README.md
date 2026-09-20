# ML-CD Lab:  ML Model Deployment with Flask + Docker

This project shows how to deploy a small machine learning model using **Flask**, **Docker**, and **pytest**.  
It includes a `/predict` API endpoint that returns a sentiment label for any text you send.

# What’s Inside

- A Flask API (`webapp/app.py`)
- A PyTorch model (`model_full.pt`)
- A tokenizer folder
- A Dockerfile to containerize everything
- Simple tests using `pytest`

#Prerequisites

Before running the project,  we installed:

- Python 3.10+
- pip
- Docker Desktop
- Git

 #Running the App Locally

1. Clone the repo:
git clone https://github.com/machancejoy-max/ml-cd-lab.git
cd ml-cd-lab
