import numpy as np
import json
import gradio as gr
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import time
import faiss
import re


# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "facebook/opt-350m"
HF_API_KEY = "Your API KEY"

# Initialize models with API key
print("Loading models...")
start_time = time.time()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL,
    token=HF_API_KEY,
    trust_remote_code=True
)
generation_model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL,
    token=HF_API_KEY,
    trust_remote_code=True
)
print(f"Models loaded in {time.time() - start_time:.2f} seconds")


def load_knowledge_base(input_dir="medical_embeddings"):
    """Load knowledge base and create FAISS index from saved embeddings."""
    try:
        print(f"\nLoading knowledge base from {input_dir}...")
        start_time = time.time()

        # Load knowledge base
        with open(os.path.join(input_dir, 'knowledge_base.json'), 'r', encoding='utf-8') as f:
            knowledge_base_data = json.load(f)

        # Load embeddings
        embeddings = torch.load(os.path.join(input_dir, 'embeddings.pth'))
        embeddings_np = embeddings.numpy().astype('float32')

        # Create FAISS index
        print("Creating FAISS index...")
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
        index.add(embeddings_np)

        # Reconstruct knowledge base
        knowledge_base = []
        for i in range(len(knowledge_base_data['conditions'])):
            knowledge_base.append({
                "condition": knowledge_base_data['conditions'][i],
                "symptoms": knowledge_base_data['symptoms'][i],
                "description": knowledge_base_data['descriptions'][i],
                "embedding": embeddings_np[i].tolist()
            })

        print(f"Knowledge base and FAISS index loaded in {time.time() - start_time:.2f} seconds")
        return knowledge_base, index

    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        raise


def extract_symptoms(text):
    """Extract symptoms from user input text."""
    # Common symptom indicators
    indicators = ['suffering from', 'experiencing', 'having', 'feel', 'feeling', 'symptoms of', 'showing signs of']

    # Clean the text
    text = text.lower()

    # Remove indicators
    for indicator in indicators:
        text = text.replace(indicator, '')

    # Split by common separators
    delimiters = [',', ' and ', ' with ', ' plus ', ' along with ']
    for delimiter in delimiters:
        text = text.replace(delimiter, ',')

    # Extract symptoms
    symptoms = [s.strip() for s in text.split(',') if s.strip()]
    return symptoms


def retrieve_relevant_conditions(symptoms: List[str], knowledge_base: List[Dict], index, top_k: int = 5):
    """Retrieve relevant conditions using FAISS similarity search."""
    # Create query embedding
    query = " ".join(symptoms)
    query_embedding = embedding_model.encode(query)

    # Prepare query for FAISS
    query_vector = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_vector)

    # Search in FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Convert distances to similarities (FAISS returns L2 distances)
    similarities = 1 / (1 + distances[0])

    # Get matches and deduplicate conditions
    seen_conditions = set()
    unique_matches = []

    for idx, sim in zip(indices[0], similarities):
        condition = knowledge_base[idx]['condition']
        if condition not in seen_conditions:
            seen_conditions.add(condition)
            unique_matches.append((knowledge_base[idx], sim))

    return unique_matches


def generate_response(symptoms: List[str], matches: List[tuple]) -> str:
    """Generate a response using the language model."""
    # Prepare the context - limit to top 2 matches for faster processing
    context = f"Patient symptoms: {', '.join(symptoms)}\n"
    context += "Relevant medical conditions:\n"

    # Only use top 2 matches for faster processing
    for item, similarity in matches[:2]:
        context += f"- {item['condition']} (confidence: {similarity:.2f})\n"
        context += f"  Symptoms: {', '.join(item['symptoms'][:5])}\n"

    # Create a specific prompt with clear instructions
    prompt = f"""You are a medical assistant. Write a medical assessment that:
1. Identifies the most likely condition based on the confidence score
2. Explains what the condition means
3. Relates the patient's symptoms to the condition
4. Provides specific management advice
5. Lists warning signs that require medical attention

Patient Information:
{context}

Write your assessment:"""

    # Generate response with optimized parameters
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
        return_attention_mask=True
    )

    outputs = generation_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id
    )

    # Get the response and clean it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove any system instructions or tags that might have been included
    response = response.replace("You are a medical assistant. Write a medical assessment that:", "")
    response = response.replace("Patient Information:", "")
    response = response.replace("Write your assessment:", "")

    # Remove any duplicate context
    response = response.replace(context, "")

    # Remove any instructions that might have been included
    instructions = [
        "Example",
        "Write your assessment",
        "Patient Information",
        "You are a medical assistant",
        "Write a medical assessment",
        "following this example",
        "If you have trouble reading",
        "write it out as if writing from memory",
        "The first two points above",
        "make sense easily enough",
        "without any additional details",
        "being added at all"
    ]
    for instruction in instructions:
        response = response.replace(instruction, "")

    # Clean up any extra whitespace and newlines
    response = "\n".join(line.strip() for line in response.split("\n") if line.strip())

    # Remove any remaining instruction fragments
    response = re.sub(r'-\s*.*', '', response)  # Remove any remaining bullet points
    response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
    response = response.strip()

    # Remove any URLs or HTML tags
    response = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', response)
    response = re.sub(r'<[^>]+>', '', response)

    # Remove any asterisks and their content
    response = re.sub(r'\*.*?\*', '', response)

    # Validate response content
    def is_valid_response(text):
        # Check if response contains any of the conditions
        has_condition = any(item['condition'].lower() in text.lower() for item, _ in matches)
        # Check if response contains any of the symptoms
        has_symptoms = any(symptom.lower() in text.lower() for symptom in symptoms)
        # Check if response is not too short
        has_length = len(text.split()) >= 10
        # Check if response doesn't contain irrelevant content
        no_irrelevant = not any(phrase in text.lower() for phrase in [
            "example", "when asked", "the doctor responded", "thank goodness",
            "you may want to use", "no medication needed", "requires medical evaluation",
            "consult a healthcare professional", "for proper diagnosis and treatment",
            "if you have trouble reading", "write it out as if writing from memory",
            "the first two points above", "make sense easily enough",
            "without any additional details", "being added at all"
        ])
        return has_condition and has_symptoms and has_length and no_irrelevant

    # If response is invalid, generate a simple fallback response
    if not is_valid_response(response):
        primary_condition = matches[0][0]['condition'] if matches else "unknown condition"
        primary_symptoms = matches[0][0]['symptoms'][:3] if matches else symptoms
        response = f"Based on your symptoms of {', '.join(primary_symptoms)}, {primary_condition} appears to be the most likely condition. Please consult a healthcare professional for proper diagnosis and treatment."

    # Format the final response
    final_response = f"{context}\n{response}\n\nImportant Note: This is an AI-powered suggestion and should not replace professional medical advice. Please consult a healthcare professional for proper diagnosis and treatment."

    return final_response


def is_medical_query(query: str) -> bool:
    """Check if the query is related to medical or health concerns using the model's understanding."""
    prompt = f"""Determine if the following query is related to medical or health concerns. Answer with only 'yes' or 'no'.

Query: {query}

Is this a medical or health-related query?"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
        return_attention_mask=True
    )

    outputs = generation_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,  # Enable sampling since we're using temperature
        top_p=0.9,  # Add top_p for better sampling
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    return "yes" in response


def process_query(user_input: str, knowledge_base: List[Dict], index) -> str:
    """Process a user query and return a response."""
    # Check if the query is medically relevant
    if not is_medical_query(user_input):
        return "I am a medical assistant designed to help with health-related questions. I cannot provide information about non-medical topics. Please ask me about symptoms, conditions, or health concerns, and I'll be happy to help."

    # Extract symptoms from the query
    symptoms = extract_symptoms(user_input)

    # If no symptoms found, return a helpful message
    if not symptoms:
        return "I couldn't identify any specific symptoms in your query. Please describe your symptoms or health concerns, and I'll be happy to help."

    # Retrieve relevant conditions
    matches = retrieve_relevant_conditions(symptoms, knowledge_base, index)

    # If no matches found, return a helpful message
    if not matches:
        return "I couldn't find any conditions that match your symptoms. Please provide more details about your symptoms, and I'll try to help you better."

    # Generate response
    response = generate_response(symptoms, matches)

    return response


def create_gradio_interface(knowledge_base, index):
    """Create and launch Gradio interface."""

    def gradio_process_query(user_input):
        return process_query(user_input, knowledge_base, index)

    iface = gr.Interface(
        fn=gradio_process_query,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Describe your symptoms (e.g., 'I have fever and cough')...",
            label="Symptoms"
        ),
        outputs=gr.Textbox(
            label="Analysis Results",
            lines=15
        ),
        title="SympAI: AI-Powered Medical Assistant",
        description="Enter your symptoms to get a comprehensive medical analysis.",
        examples=[
            ["I have fever and cough"],
            ["Experiencing headache and nausea"],
            ["Feeling joint pain with swelling"]
        ]
    )

    return iface


def main():
    try:
        total_start_time = time.time()

        # Load knowledge base and create FAISS index
        knowledge_base, index = load_knowledge_base()

        total_time = time.time() - total_start_time
        print(f"\nTotal loading time: {total_time:.2f} seconds")

        # Create and launch Gradio interface
        iface = create_gradio_interface(knowledge_base, index)
        iface.launch()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure the medical_embeddings directory is accessible and contains the required files.")


if __name__ == "__main__":
    main()