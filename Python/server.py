import contextlib
import io
import os
import warnings
import speech_recognition as sr
from flask import Flask, request, jsonify
import logging
import time
from tqdm import tqdm
import torch

from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from crewai import Agent

app = Flask(__name__)

# Set up logging and warnings
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('fairseq').setLevel(logging.WARNING)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Classification prompt
classification_prompt = """
Classify the following task description into one of the categories: 
1. Code: For tasks related to programming or code generation.
2. Literacy: For tasks related to writing, storytelling, or blog posts.
3. Generic: For general knowledge, trends, or insights.
4. Complex: For complex tasks that don't fit into the other categories.
5. Other: For tasks that don't fit into any of the above categories.

Task Description: {}
"""

def initialize_ollama_model(model_name):
    try:
        logging.info(f"Initializing model: {model_name}...")
        with tqdm(total=100, desc=f"Loading {model_name}",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for _ in range(100):
                time.sleep(0.02)
                pbar.update(1)
            model = Ollama(model=model_name)
        logging.info(f"Successfully initialized model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Error initializing model {model_name}: {e}")
        return None

@app.route('/upload', methods=['POST'])
def handle_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        audio_file_path = audio_file.filename
        audio_file.save(audio_file_path)

        recognized_text = recognize_speech(audio_file_path)
        if recognized_text:
            response_text, _ = handle_response(recognized_text)
            if response_text:
                return jsonify({
                    "recognized_text": recognized_text,
                    "response_text": response_text
                })
            else:
                return jsonify({"error": "Error generating response text"}), 500
        else:
            return jsonify({"error": "Error recognizing speech"}), 500
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        logging.info(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        logging.error("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during speech recognition: {e}")
        return None

def handle_response(task_description):
    llm = agents['classifier'].llm
    if not llm:
        return "No LLM assigned to the classifier agent.", None

    prompts = [classification_prompt.format(task_description)]

    try:
        classification_result = llm.generate(prompts)
        classification = classification_result.text if hasattr(classification_result, 'text') else str(classification_result)
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return "Error during classification.", None

    category = classification.split(":")[-1].strip()

    if category == "Code":
        researcher = agents['researcher2']
    elif category == "Literacy":
        researcher = agents['researcher4']
    elif category == "Complex":
        researcher = agents['researcher3']
    elif category == "Generic":
        researcher = agents['researcher1']
    else:
        researcher = agents['researcher3']

    researcher_llm = researcher.llm
    if not researcher_llm:
        return "No LLM assigned to the selected researcher.", None

    try:
        response_result = researcher_llm.generate([task_description])
        response_text = response_result.generations[0][0].text if hasattr(response_result, 'generations') else str(response_result)
        response_text = response_text.split('generation_info=')[0].strip()
    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        return "Error during response generation.", None

    return response_text, None

def initialize_crewsai_agents():
    search_tool_func = DuckDuckGoSearchRun()
    search_tool = Tool(name="DuckDuckGoSearch", description="A search tool used to query DuckDuckGo for search results.",
                       func=search_tool_func.run)

    researcher1 = Agent(
        role='Conversational Specialist',
        goal='Engage in meaningful and insightful conversations on a wide range of topics.',
        backstory='A skilled conversationalist adept at navigating various topics with depth and clarity, ensuring engaging and informative dialogues.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_lmstudio
    )

    researcher2 = Agent(
        role='Code Specialist',
        goal='Assist with programming tasks, code generation, and debugging.',
        backstory='You are a highly skilled code specialist with expertise in various programming languages and tools. Your mission is to assist users with their coding needs, from writing and reviewing code to resolving programming issues.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_janai
    )

    researcher3 = Agent(
        role='Complex Task Specialist',
        goal='Solve intricate problems and provide nuanced insights for complex issues.',
        backstory='You are an expert in addressing multifaceted challenges and offering in-depth analysis. Your focus is on handling tasks that require sophisticated understanding and creative problem-solving.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_ollama
    )

    researcher4 = Agent(
        role='Creative Writer',
        goal='Craft engaging and insightful content on a variety of topics',
        backstory='As a skilled writer with a flair for creativity, you excel at producing captivating stories, articles, and other forms of written content. Your expertise lies in making complex topics accessible and interesting to a broad audience.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_textgen
    )

    classification_agent = Agent(
        role='Task Classifier',
        goal='Classify task descriptions into categories: Code, Literacy, Generic, Complex, or Other.',
        backstory='A specialized agent that categorizes tasks based on predefined categories to route them appropriately.',
        verbose=True,
        tools=[search_tool],
        allow_delegation=False,
        llm=llm_lmstudio
    )

    return {
        'researcher1': researcher1,
        'researcher2': researcher2,
        'researcher3': researcher3,
        'researcher4': researcher4,
        'classifier': classification_agent
    }

# Initialize Ollama models
llm_lmstudio = initialize_ollama_model("neural-chat")
llm_janai = initialize_ollama_model("codegemma")
llm_ollama = initialize_ollama_model("llama3.1")
llm_textgen = initialize_ollama_model("mistral")

agents = initialize_crewsai_agents()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
