import os
import json
import mlflow
from groq import Groq
from dotenv import load_dotenv

# --- 1. Configuration and Setup ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

client = Groq(api_key=api_key)

# Ensure no active MLflow run is left open from previous executions
if mlflow.active_run():
    mlflow.end_run()

# Set MLflow experiment name
mlflow.set_experiment("GenAI_Prompt_Optimization_Summarization")
print("MLflow Experiment set to: GenAI_Prompt_Optimization_Summarization")

# --- 2. Sample Data (Article to Summarize) ---
sample_article = """
India, a land steeped in history and vibrant with diversity, tells a story as old as time itself. From the ancient civilizations that thrived along the Indus Valley, building sophisticated cities millennia ago, to the spiritual wisdom enshrined in the Vedas, the subcontinent has been a cradle of human thought and innovation.

Imagine the dawn of empires: the Mauryas uniting much of the land under Ashoka's peaceful dharma, or the Guptas ushering in a "Golden Age" of science, mathematics (with the invention of the decimal system and the concept of zero), and art. Later, the intricate grandeur of the Mughal Empire brought new architectural marvels like the Taj Mahal and a rich tapestry of cultural exchange.

Yet, India's narrative isn't just about emperors and monuments. It's woven through the daily lives of its people, the kaleidoscope of festivals celebrated with fervent joy, the aromas of spices wafting from bustling markets, and the symphony of languages spoken across its vast landscape. It's the resilience of farmers tending fields passed down through generations, the rhythmic hum of handlooms creating exquisite textiles, and the vibrant energy of its megacities, pulsating with technological ambition.

The struggle for independence, led by figures like Mahatma Gandhi and his philosophy of non-violent resistance, became a global inspiration, culminating in the birth of the modern Indian republic. Today, India stands as the world's largest democracy, a land of stark contrasts where ancient traditions coexist with cutting-edge advancements in space technology and software. Its story continues to unfold, a complex, dynamic, and ever-evolving saga of a billion dreams.
"""

# --- 3. Mock LLM Judge Function (Simulates a real LLM for evaluation) ---
# In a real scenario, this would be another LLM API call
# that evaluates the summary based on criteria like conciseness, relevance, coherence.
# For simplicity, this mock judge assigns scores based on simple text properties.
def evaluate_summary_with_mock_llm_judge(original_text: str, prompt_used: str, generated_summary: str) -> dict:
    """
    A mock LLM judge to evaluate the quality of a generated summary.
    In a real application, this would involve calling another LLM (e.g., GPT-4, Claude)
    with a specific prompt to rate the summary.
    """
    evaluation_results = {
        "conciseness_score": 0.0,
        "relevance_score": 0.0,
        "coherence_score": 0.0,
        "overall_score": 0.0
    }

    # Simulate conciseness: shorter summaries (within reason) get higher scores
    original_word_count = len(original_text.split())
    summary_word_count = len(generated_summary.split())

    if summary_word_count < 0.2 * original_word_count:
        evaluation_results["conciseness_score"] = 9.0 # Very concise
    elif summary_word_count < 0.4 * original_word_count:
        evaluation_results["conciseness_score"] = 7.0 # Moderately concise
    else:
        evaluation_results["conciseness_score"] = 4.0 # Less concise

    # Simulate relevance: check for keywords from the original text
    keywords = ["AI", "healthcare", "finance", "challenges", "innovation"]
    relevant_keywords_found = sum(1 for kw in keywords if kw.lower() in generated_summary.lower())
    evaluation_results["relevance_score"] = (relevant_keywords_found / len(keywords)) * 10.0

    # Simulate coherence: check for basic sentence structure / length (very basic mock)
    # A real LLM judge would assess flow, grammar, etc.
    if len(generated_summary.split('.')) > 1 and len(generated_summary.split('.')) < 5:
        evaluation_results["coherence_score"] = 8.0
    else:
        evaluation_results["coherence_score"] = 5.0

    evaluation_results["overall_score"] = (evaluation_results["conciseness_score"] +
                                           evaluation_results["relevance_score"] +
                                           evaluation_results["coherence_score"]) / 3.0

    return evaluation_results

# --- 4. Define Different Prompts to Test ---
prompts_to_test = [
    {
        "name": "Standard Summary",
        "text": "Summarize the following article concisely: {article_text}"
    },
    {
        "name": "Bullet Point Summary",
        "text": "Provide a summary of the following article in 3-5 bullet points, focusing on key challenges and benefits: {article_text}"
    },
    {
        "name": "One Sentence Summary",
        "text": "Condense the following article into a single, comprehensive sentence: {article_text}"
    },
    {
        "name": "Detailed Summary",
        "text": "Write a detailed summary of the following article, ensuring all major aspects are covered: {article_text}"
    }
]

# --- 5. Run Experiments ---
for i, prompt_data in enumerate(prompts_to_test):
    prompt_name = prompt_data["name"]
    raw_prompt_template = prompt_data["text"]
    
    # Fill in the article text into the prompt template
    current_prompt = raw_prompt_template.format(article_text=sample_article)

    print(f"\n--- Running Experiment for Prompt: '{prompt_name}' ({i+1}/{len(prompts_to_test)}) ---")

    with mlflow.start_run(run_name=prompt_name):
        # --- Log Parameters ---
        mlflow.log_param("prompt_strategy", prompt_name)
        mlflow.log_param("raw_prompt_template", raw_prompt_template)
        mlflow.log_param("full_input_prompt", current_prompt)
        mlflow.log_param("model_name", "llama3-8b-8192")
        mlflow.log_param("temperature", 0.7)
        mlflow.log_param("max_tokens", 500)

        # --- Log Original Article as Artifact ---
        with open("original_article.txt", "w") as f:
            f.write(sample_article)
        mlflow.log_artifact("original_article.txt")

        try:
            # --- Call Groq API ---
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": current_prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=500,
            )
            generated_text = chat_completion.choices[0].message.content
            print(f"Generated Summary ({prompt_name}):\n{generated_text}\n")

            # --- Log Generated Text as Artifact ---
            with open(f"generated_summary_{prompt_name.replace(' ', '_')}.txt", "w") as f:
                f.write(generated_text)
            mlflow.log_artifact(f"generated_summary_{prompt_name.replace(' ', '_')}.txt")

            # --- Log Basic Metric ---
            mlflow.log_metric("generated_summary_length", len(generated_text.split()))

            # --- Evaluate with LLM Judge and Log Metrics ---
            evaluation_scores = evaluate_summary_with_mock_llm_judge(sample_article, current_prompt, generated_text)
            for metric_name, score in evaluation_scores.items():
                mlflow.log_metric(f"judge_{metric_name}", score)
            print(f"Judge Scores: {evaluation_scores}")

            # --- Log Full API Response as Artifact (Optional but useful) ---
            with open(f"chat_completion_{prompt_name.replace(' ', '_')}.json", "w") as f:
                json.dump(chat_completion.to_dict(), f, indent=4)
            mlflow.log_artifact(f"chat_completion_{prompt_name.replace(' ', '_')}.json")

        except Exception as e:
            print(f"❌ Error during generation for '{prompt_name}': {e}")
            mlflow.log_param("generation_error", str(e))
            mlflow.set_tag("status", "failed")
            continue # Continue to next prompt if one fails

print("\n✅ All MLflow runs completed for prompt optimization.")