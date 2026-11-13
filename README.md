# 🧠 Real-Time AI Facial Recognition System

A real-time AI-powered facial recognition system built using **Python, OpenCV, and InsightFace's Buffalo model**.

The system uses a client-server architecture to perform facial recognition through a laptop's webcam. It can identify registered individuals using just **3–5 reference images per person**, leveraging pretrained deep learning models and facial embedding similarity matching.

## 🚀 Key Features

- **Real-Time Facial Recognition:** Recognizes individuals through a live webcam feed.
- **Buffalo AI Model:** Uses InsightFace's pretrained Buffalo model for facial analysis and recognition.
- **Few-Shot Face Enrollment:** Requires only 3–5 reference images per individual.
- **Client-Server Architecture:** Separates webcam capture and server-side recognition processing.
- **Facial Embedding Extraction:** Converts detected faces into numerical feature representations.
- **Similarity-Based Identification:** Matches live facial embeddings against registered identities.
- **Live Camera Processing:** Uses OpenCV to capture and process webcam frames.

## 🏗️ System Architecture

The project consists of two primary components:

### 1. Server Side – `opencv1.py`

Responsible for the server-side facial recognition operations.

- Loads the pretrained Buffalo model.
- Processes incoming image/frame data.
- Performs face detection and feature extraction.
- Compares facial embeddings against registered identities.
- Determines the recognized identity and returns the result to the client.

### 2. Client Side – `webi.py`

Runs on the laptop and handles webcam interaction.

- Accesses the laptop's webcam.
- Captures live video frames.
- Communicates with the server.
- Receives facial recognition results.
- Displays the results to the user.

## 🔄 Recognition Workflow

```text
        LAPTOP WEBCAM
              |
              v
       CLIENT SIDE
         webi.py
              |
              v
       Capture Frames
              |
              v
       Send Frames
              |
              v
      SERVER SIDE
        opencv1.py
              |
              v
      Buffalo AI Model
              |
              v
      Face Detection
              |
              v
     Feature Extraction
              |
              v
    Embedding Comparison
              |
              v
     Identity Recognition
              |
              v
       Return Results
              |
              v
       CLIENT DISPLAY
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| InsightFace | Facial recognition framework |
| Buffalo Model | Pretrained facial analysis model |
| NumPy | Numerical computation |
| Client-Server Communication | Communication between client and server |

## 📂 Project Structure

```text
AI-Facial-Recognition/
│
├── opencv1.py    # Server-side facial recognition
├── webi.py       # Client-side webcam application
│
└── README.md
```

## ⚙️ Running the Project

**Step 1 – Start the server**

Run the server-side application:

```bash
python opencv1.py
```

**Step 2 – Start the client**

On the laptop that will provide the webcam feed, run:

```bash
python webi.py
```

Ensure the client is configured with the correct server IP address and port.

## 🎯 Project Objective

The goal is to develop a practical, real-time facial recognition system that identifies individuals with minimal reference images while distributing processing between a client device and a server.

The architecture allows the webcam-equipped client device to communicate with a separate machine responsible for AI-based facial recognition.

## 🔮 Future Improvements

- Multi-client support.
- Optimized real-time frame transmission.
- Face recognition database management.
- Improved recognition under varying lighting conditions.
- Secure client-server communication.
- Scalable deployment across multiple devices.

## 🔐 Privacy & Responsible Use

This project is intended for educational and authorized applications. Obtain consent before collecting or processing facial data. Protect stored facial embeddings and provide a mechanism to remove registered identities.

## 👨‍💻 Author

**Sayan Ghosh**

AI/ML | Python | Computer Vision | Deep Learning

⭐ If you find this project interesting, consider starring the repository!
