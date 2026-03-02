import os
import sys
import datetime

import anthropic
from dotenv import load_dotenv


# Load environment variables from the .env file so we can access the API key
load_dotenv()

# Read the Anthropic API key from the environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# If the API key is missing, print an error message and exit the program
if not api_key:
    print("Error: ANTHROPIC_API_KEY not found in .env")
    sys.exit(1)

# Create an Anthropic client instance using the API key
client = anthropic.Anthropic(api_key=api_key)


# Ask the user to choose how long they want the summary to be
def get_summary_length_choice() -> str:
    """
    Prompt the user to choose a summary length and keep asking until they enter 1, 2, or 3.
    Returns the chosen option as a string: "1", "2", or "3".
    """

    while True:
        print("Choose summary length:")
        print("  1 = Short (2-3 sentences)")
        print("  2 = Medium (1 paragraph)")
        print("  3 = Detailed (3-4 bullet points)")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice in {"1", "2", "3"}:
            return choice

        print("Invalid choice. Please enter 1, 2, or 3.\n")


# Ask the user to paste the text they want summarized and collect all lines into one string
def get_input_text() -> str:
    """
    Prompt the user to paste text, ending input when they press Enter twice in a row.
    Returns the combined text as a single string.
    """

    print("\nPaste the text you want summarized.")
    print("When you are done, press Enter on an empty line twice in a row.\n")

    lines = []
    empty_count = 0

    while True:
        line = input()

        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            lines.append(line)

    text = "\n".join(lines).strip()

    if not text:
        print("No text was provided. Exiting.")
        sys.exit(1)

    return text


# Build the system prompt that tells Claude how to summarize, based on the user's choice
def build_system_prompt(choice: str) -> str:
    """
    Return an appropriate system prompt string based on the summary length choice.
    """

    if choice == "1":
        return (
            "You are a summarization assistant. Summarize the provided text "
            "in exactly 2-3 sentences. Be concise and capture the main point."
        )
    elif choice == "2":
        return (
            "You are a summarization assistant. Summarize the provided text "
            "in one clear paragraph of 4-6 sentences."
        )
    else:  # choice == "3"
        return (
            "You are a summarization assistant. Summarize the provided text "
            "as 3-4 bullet points. Each bullet should be one sentence."
        )


# Save the summary details to a timestamped text file
def save_summary_to_file(length_choice: str, input_text: str, summary_text: str) -> None:
    """
    Save the date/time, summary length choice, original text, and summary
    to a uniquely named file so results are not overwritten.
    """

    # Create a timestamp to make the filename unique
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = f"summary_{timestamp}.txt"

    # Provide a human-readable label for the summary length choice
    length_labels = {
        "1": "Short (2-3 sentences)",
        "2": "Medium (1 paragraph)",
        "3": "Detailed (3-4 bullet points)",
    }
    length_label = length_labels.get(length_choice, "Unknown")

    # Write all requested information into the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Date and time: {now.isoformat()}\n")
        f.write(f"Summary length choice: {length_choice} - {length_label}\n\n")
        f.write("Original text:\n")
        f.write(input_text + "\n\n")
        f.write("Summary:\n")
        f.write(summary_text + "\n")

    # Inform the user where the file was saved
    full_path = os.path.abspath(filename)
    print(f"\nSummary saved to {full_path}")


def main() -> None:
    """
    Main function that ties together user input, prompt construction, and the Claude API call.
    """

    # Step 1: Ask the user how long they want the summary to be
    length_choice = get_summary_length_choice()

    # Step 2: Ask the user for the text to summarize
    input_text = get_input_text()

    # Step 3: Build the system prompt based on the chosen summary length
    system_prompt = build_system_prompt(length_choice)

    # Step 4: Send the text and system prompt to Claude and request a summary
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": input_text,
            }
        ],
    )

    # Step 5: Extract the summary text from the response (or a fallback message)
    if response.content and len(response.content) > 0:
        summary_text = response.content[0].text
    else:
        summary_text = "Claude returned an empty response."

    # Step 6: Print the summary result to the terminal
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print(summary_text)

    # Step 7: Save the summary and related details to a timestamped file
    save_summary_to_file(length_choice, input_text, summary_text)


if __name__ == "__main__":
    main()

