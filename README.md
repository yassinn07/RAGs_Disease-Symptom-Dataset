## Features

* **Symptom Extraction:** Accurately extracts key symptoms from natural language input.
* **Knowledge Base Search:** Utilizes FAISS to quickly find relevant medical conditions from a pre-built knowledge base based on symptom similarity.
* **AI-Powered Assessment:** Generates detailed medical assessments, including the most likely condition, explanation, symptom relation, management advice, and warning signs, using a fine-tuned language model.
* **Medical Query Filtering:** Ensures that only medical or health-related queries are processed, redirecting non-medical inquiries appropriately.
* **User-Friendly Interface:** Provides an intuitive web interface built with Gradio for easy interaction.

---


### Knowledge Base Setup

SympAI requires a pre-built knowledge base and associated embeddings.

1.  **Create the `medical_embeddings` directory:**
    In the root of your project, create a directory named `medical_embeddings`.

    ```bash
    mkdir medical_embeddings
    ```

2.  **Populate the `medical_embeddings` directory:**
    You need two files inside this directory:
    * `knowledge_base.json`: A JSON file containing your medical conditions, symptoms, and descriptions. The structure should be:
        ```json
        {
          "conditions": ["Condition A", "Condition B"],
          "symptoms": [["Symptom 1A", "Symptom 2A"], ["Symptom 1B", "Symptom 2B"]],
          "descriptions": ["Description A", "Description B"]
        }
        ```
    * `embeddings.pth`: A PyTorch tensor file containing the embeddings of your medical data, corresponding to the `knowledge_base.json`. This file needs to be generated using the `EMBEDDING_MODEL` (`sentence-transformers/all-MiniLM-L6-v2`) on your `knowledge_base.json` content.
        *(A separate script would typically be used to generate these embeddings. For example, a Python script that loads `knowledge_base.json`, encodes the `conditions` or a combination of `conditions` and `symptoms` using `SentenceTransformer`, and then saves the resulting embeddings to `embeddings.pth`.)*


### Running the Application

Once the knowledge base is set up, you can run the Gradio application

Dataset : https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
