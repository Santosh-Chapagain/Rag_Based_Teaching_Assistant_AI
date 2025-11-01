import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json
import sys
import traceback


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    r.raise_for_status()
    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "gpt-oss",
        "prompt": prompt,
        "stream": False
    })
    r.raise_for_status()
    return r.json()


def safe_write(path, text):
    # Always write in UTF-8 to avoid Windows cp1252 errors
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    try:
        df = joblib.load('embeddings.joblib')
    except Exception:
        print("Failed to load embeddings.joblib", file=sys.stderr)
        traceback.print_exc()
        return

    incoming_query = input("Ask a Question: ").strip()
    if not incoming_query:
        print("No query provided, exiting.")
        return

    try:
        question_embedding = create_embedding([incoming_query])[0]
    except Exception:
        print("Embedding creation failed", file=sys.stderr)
        traceback.print_exc()
        return

    try:
        similarities = cosine_similarity(np.vstack(df['embedding']), [
                                         question_embedding]).flatten()
    except Exception:
        print("Similarity computation failed", file=sys.stderr)
        traceback.print_exc()
        return

    top_results = 5
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]

    prompt = f'''I am teaching 100 days python course . Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "Number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

    # Save prompt with UTF-8
    try:
        safe_write("prompt.txt", prompt)
    except Exception:
        print("Failed to write prompt.txt", file=sys.stderr)
        traceback.print_exc()

    # Call inference and be defensive about output format
    try:
        response_obj = inference(prompt)
    except Exception:
        print("Inference call failed", file=sys.stderr)
        traceback.print_exc()
        response_obj = {"error": "inference failed"}

    # Inspect and extract a human-readable response
    response_text = ""
    try:
        if isinstance(response_obj, dict):
            # Common fields: "response", "text", "output"
            response_text = response_obj.get("response") or response_obj.get(
                "text") or response_obj.get("output")
            if response_text is None:
                # Not a simple text; pretty-print the whole JSON
                response_text = json.dumps(
                    response_obj, indent=2, ensure_ascii=False)
        else:
            response_text = str(response_obj)
    except Exception:
        response_text = str(response_obj)

    # Print to console
    print(response_text)

    # Write response safely (UTF-8)
    try:
        safe_write("response.txt", response_text)
    except Exception:
        # As a fallback, write with replacement to avoid crashes
        with open("response.txt", "w", encoding="utf-8", errors="replace") as f:
            f.write(response_text)


if __name__ == "__main__":
    main()
