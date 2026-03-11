import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import os

# Initialize Vertex AI with your project details
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") 
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

class GitaAIEngine:
    def __init__(self):
        # Using Gemini 1.5 Flash for low latency (good for voice mode)
        self.model = GenerativeModel("gemini-1.5-flash")
        
        # System Instruction: Repurposing Gita for Western Audience [cite: 1]
        self.instructions = (
            "You are a spiritual and psychological guide. Your task is to take "
            "teachings from the Bhagavad Gita and repurpose them for a Western "
            "audience using Cognitive Behavioral Therapy (CBT) frameworks. "
            "Focus on mindfulness, duty (Dharma), and detachment from outcomes. "
            "Avoid overly religious language; use psychological terms like "
            "'cognitive reframing' and 'process-oriented growth'."
        )

    def generate_response(self, user_text, lesson_id=None):
        # Contextual prompt based on user input or lesson progression [cite: 2]
        chat = self.model.start_chat()
        
        prompt = f"System Context: {self.instructions}\n\n"
        if lesson_id:
            prompt += f"Current Lesson Context: {lesson_id}. Focus on the specific principle of this chapter.\n"
        
        prompt += f"User Input: {user_text}"
        
        response = chat.send_message(prompt)
        return response.text