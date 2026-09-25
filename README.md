# Emotion Classification with RNNs, LSTMs & GRUs

This project demonstrates how to classify text into six distinct emotions (sadness, joy, love, anger, fear, and surprise) using various Recurrent Neural Network (RNN) architectures. The project starts by comparing foundational models like Simple RNN, LSTM, and GRU, and then builds an advanced Bidirectional GRU (BiGRU) model that achieves over 92% accuracy. The final model is served via a REST API using FastAPI.

## Features

-   **Multi-Class Emotion Classification:** Accurately categorizes text into six common emotions.
-   **Comparative Model Analysis:** Provides a clear comparison between Simple RNN, LSTM, GRU, and Bidirectional GRU architectures.
-   **High-Performance Model:** The final Bidirectional GRU model achieves **92.4% accuracy** on the test set.
-   **Pre-trained Artifacts:** Includes a pre-trained Keras model (`BiGRU_Model.keras`) and a tokenizer (`tokenizer.pkl`).
-   **RESTful API:** A production-ready API built with FastAPI allows for easy integration and real-time predictions.
-   **Dockerized Application:** Comes with a `Dockerfile` for seamless setup and deployment.

## Methodology

The project follows a structured approach, from data exploration to model deployment.

### 1. Data Loading and EDA

-   **Dataset:** The project uses the `dair-ai/emotion` dataset from Hugging Face, which contains sentences labeled with one of six emotions.
-   **Exploratory Data Analysis (EDA):** The class distribution is analyzed to understand the prevalence of each emotion. The dataset is somewhat imbalanced, with "joy" and "sadness" being the most common labels.

### 2. Data Preprocessing

-   **Tokenization:** The text data is tokenized using `tf.keras.preprocessing.text.Tokenizer`, which converts sentences into sequences of integers.
-   **Padding:** All sequences are padded to a maximum length of 50 to ensure uniform input size for the models.
-   **Class Weights:** To handle the class imbalance, balanced class weights are computed and applied during training. This ensures the model does not become biased towards the majority classes.

### 3. Model Training and Comparison

Four different models were trained and evaluated:

1.  **Simple RNN:** A basic recurrent network that serves as a baseline.
2.  **Standard LSTM:** A Long Short-Term Memory network, designed to better handle long-range dependencies.
3.  **Standard GRU:** A Gated Recurrent Unit network, which is often more efficient than LSTM.
4.  **Advanced Bidirectional GRU:** The final, high-performing model. It uses bidirectional layers to process text from both directions, 300-dimensional embeddings, and dropout for regularization.

The **Bidirectional GRU** significantly outperformed the other models, achieving the highest accuracy.

### 4. API Deployment

The trained BiGRU model is deployed using a FastAPI server. The API exposes endpoints to check the server's health and predict emotions from input text.

## How to Use the API

You can interact with the API to get real-time emotion predictions.

### 1. Setup

**Using Docker (Recommended):**

1.  **Build the Docker image:**
    ```bash
    docker build -t emotion-predictor-app .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8000:8000 emotion-predictor-app
    ```

**Running Locally:**

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

The API will be available at `http://localhost:8000`.

### 2. API Endpoints

-   `GET /`: A welcome message to confirm the server is running.
-   `GET /health`: Checks the server status and whether the model is loaded.
-   `POST /predict`: Predicts the emotion of a given text.

**Example `POST /predict` Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "I was shocked and completely surprised by the unexpected gift!"
}'


Example Response:

{
  "text": "I was shocked and completely surprised by the unexpected gift!",
  "predicted_emotion": "surprise",
  "confidence": 0.999,
  "all_probabilities": {
    "sadness": 1.2e-06,
    "joy": 5.4e-06,
    "love": 2.1e-07,
    "anger": 8.9e-07,
    "fear": 2.3e-06,
    "surprise": 0.999
  }
}
