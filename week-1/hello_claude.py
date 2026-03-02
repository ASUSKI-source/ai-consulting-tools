import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables from the .env file in the current directory
load_dotenv()

# Read the Anthropic API key from the environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# If the API key is missing, print a clear error and exit the program
if not api_key:
    print("Error: ANTHROPIC_API_KEY is not set in the environment or .env file.")
    sys.exit(1)

# Create an Anthropic client instance using the API key
client = Anthropic(api_key=api_key)

# Define the message we want to send to Claude
user_message = "What are some practical ways to reduce product waste in the dairy section of a grocery store?"

# Send the message to Claude using the specified model and token limit
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    system="You are a retail operations assistant who specializes in grocery store efficiency. You give practical, specific advice. Keep all answers under 100 words.",
    messages=[
        {
            "role": "user",
            "content": user_message,
        }
    ],
)

# Print Claude's text response to the terminal
if response.content and len(response.content) > 0:
    print(response.content[0].text)
else:
    print("Claude returned an empty response.")

