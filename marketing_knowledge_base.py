import os

# Define directory and file structure
structure = {
    "app": {
        "__init__.py": "",
        "main.py": "",
        "config.py": "",
        "api": {
            "__init__.py": "",
            "chat.py": "",
            "admin.py": "",
            "metrics.py": ""
        },
        "core": {
            "__init__.py": "",
            "retrieval.py": "",
            "llm.py": "",
            "metrics_engine.py": "",
            "evaluation.py": ""
        },
        "db": {
            "__init__.py": "",
            "vector_store.py": "",
            "metrics_db.py": ""
        },
        "utils": {
            "__init__.py": "",
            "text_processing.py": "",
            "evaluation_metrics.py": ""
        }
    },
    "data": {
        "knowledge_base": {},
        "vector_store": {},
        "metrics": {},
        "chat_summaries": {}
    },
    "scripts": {
        "ingest_documents.py": "",
        "update_metrics.py": "",
        "evaluation_runner.py": ""
    },
    "tests": {
        "__init__.py": "",
        "test_retrieval.py": "",
        "test_chat.py": "",
        "test_metrics.py": "",
        "test_evaluation.py": ""
    },
    "requirements.txt": "",
    ".env.example": "",
    "README.md": "",
    "docker-compose.yml": ""
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)

if __name__ == "__main__":
    current_dir = os.getcwd()
    create_structure(current_dir, structure)
    print("Project structure created successfully.")
