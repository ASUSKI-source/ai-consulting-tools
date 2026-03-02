from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

for m in client.models.list().data:
    print(m.id)