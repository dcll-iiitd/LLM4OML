import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Load your JSON data
with open('./outputs/json/derived_optimizer_nvidia-llama-3.3-nemotron-super-49b-v1_20250602-103941.json', 'r') as f:
    data = json.load(f)

# Extract the 9 output strings
api_interactions = data['combinations'][0]['api_interactions']
outputs = [interaction['output'] for interaction in api_interactions]

# Split outputs into two groups
first_five = "\n\n".join(outputs[:5])
last_four = "\n\n".join(outputs[5:9])

# Create the prompts
prompt1 = f"""
I have an optimizer algorithm
I need you to create a latex pdf from the content I am providing
The structure should be:
1. Algorithm description
2. Pseudo code and dry run
3. Convergence theory
4. Convergence proofs (all 6 under different assumption settings)

Content is here:

{first_five}

IMPORTANT: DO NOT START GENERATING LATEX YET. ONLY RESPOND WITH "Waiting..." EXACTLY AS SHOWN.
"""

prompt2 = f"""
Here are the additional proofs - add them to the document:
{last_four}

NOW generate the complete LaTeX document with the following requirements:
1. Output should be of atleast 16-20 pages do not skip any part of content
2. Provide all the 6 convergence proofs exactly as given in the content in different proof sections
3. Do not alter any convergence proofs - keep all detailed steps
4. Include all content from both parts
5. Output ONLY the full LaTeX code between \\documentclass{{article}} and \\end{{document}} tags
6. Do not add any extra text outside these tags
"""
prompt3 = f"""
continue
"""

# Configure Gemini
genai.configure(api_key=os.getenv("aiml_api"))  # Replace with your actual API key
model = genai.GenerativeModel('gemini-2.0-flash')

# Create conversation history
chat = model.start_chat(history=[])

# Send first prompt and validate response
response1 = chat.send_message(prompt1)
if "waiting..." not in response1.text.lower():
    print("Warning: Unexpected first response")
    print(response1.text)

# Send second prompt and get final response
response2 = chat.send_message(prompt2)
full_response = response2.text

response3 = chat.send_message(prompt3)
full_response = full_response+response3.text

# Extract and save LaTeX content
start_tag = "\\documentclass{article}"
end_tag = "\\end{document}"

start_idx = full_response.find(start_tag)
end_idx = full_response.find(end_tag)

if start_idx != -1 and end_idx != -1:
    latex_content = full_response[start_idx:end_idx + len(end_tag)]
    with open('./outputs/tex/output.tex', 'w') as f:
        f.write(latex_content)
    print("LaTeX document saved to output.tex")
else:
    print("Error: Could not find LaTeX document tags in response")
    print("Full response saved to full_response.txt")
    with open('./outputs/full_response.txt', 'w') as f:
        f.write(full_response)