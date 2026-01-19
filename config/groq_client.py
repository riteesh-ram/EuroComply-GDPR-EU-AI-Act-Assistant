from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class Groq:
    def __init__(self, model="llama-3.3-70b-versatile"):
        self.model = model
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=self.model)

    async def get_llm(self):
        if not self.llm:
            raise ValueError("llm is required.")
        return self.llm