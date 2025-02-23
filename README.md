# Krishna AI - Voice Assistant  

✨ Designed to be a smart and interactive classmate for students, Krishna AI responds to voice commands, answers questions, and engages in a meaningful, simple, and student-friendly way.  

##  Demo

  {video cooking}

## Features 🚀  

🔹 **Wake Word Activation** – Listens for "Hey Krishna" or "Hello Krishna" to start interacting.  
🔹 **Conversational AI** – Uses Google Gemini AI to generate responses tailored for students.  
🔹 **Text-to-Speech (TTS)** – Converts AI responses into natural-sounding speech.  
🔹 **Speech Recognition** – Understands and processes voice commands.  
🔹 **Kerala Context** – Provides region-specific answers when relevant.  
🔹 **Custom Responses** – Answers common queries about schools, colleges, and local figures.  
🔹 **Multi-Language Support** – Responds in **English and Malayalam** based on user preference.  
🔹 **Safe & Kid-Friendly** – Avoids inappropriate topics and promotes safe discussions.  

## How It Works 🎧  

🟢 **Step 1:** Run the Python script and let Krishna AI start listening.  
🟢 **Step 2:** Say **"Hey Krishna"** or **"Hello Krishna"** to wake it up.  
🟢 **Step 3:** Ask your question or give a command in **English or Malayalam**. Krishna AI will listen, process, and respond accordingly!  

## Setup 🛠️  

🔹 Install dependencies:  
```bash
pip install -r requirements.txt
```  
🔹 Create a `.env` file and add your API keys:  
```env
GEMINI_API_KEY=your_gemini_api_key  
PICOVOICE_ACCESS_KEY=your_picovoice_api_key  
```  
🔹 Run the assistant:  
```bash
python main.py
```  

## Technologies Used 💻  

🎤 **Porcupine** – Wake word detection  
🧠 **Google Gemini AI** – Response generation  
🗣 **Google Speech Recognition** – Converts speech to text  
🔊 **gTTS** – Text-to-speech conversion  
🌍 **Multi-Language Processing** – Supports **English & Malayalam** for input and output  
