import os
import json
import mlflow
from groq import Groq
from dotenv import load_dotenv


# Load .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

if mlflow.active_run():
    mlflow.end_run()


# Set experiment
mlflow.set_experiment("Groq_GenAI_Experiment")

# Start run
with mlflow.start_run():
    model = "llama3-8b-8192"
    temp = 0.7
    max_toks = 500
    prompt = "Explain how to hack my friends mobile?"
    timeout=10

    # Log params
    mlflow.log_param("model_name", model)
    mlflow.log_param("temperature", temp)
    mlflow.log_param("max_tokens", max_toks)
    mlflow.log_param("prompt", prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=temp,
            max_tokens=max_toks,
        )
        generated_text = chat_completion.choices[0].message.content
        print("Generated Text:", generated_text)

        # Save and log text
        with open("generated_text.txt", "w") as f:
            f.write(generated_text)
        mlflow.log_artifact("generated_text.txt")

        # Log metric
        mlflow.log_metric("generated_text_length", len(generated_text))

        # Save and log full object
        with open("chat_completion.json", "w") as f:
            json.dump(chat_completion.to_dict(), f, indent=4)
        mlflow.log_artifact("chat_completion.json")

    except Exception as e:
        print("❌ Error during generation:", e)

print("✅ MLflow run completed.")
