from groq import Groq

client = Groq(api_key="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")

def generate_summary (prediction : str, confidence : float,  triage_message : str)->str:
    prompt = f"""You are a medical triage assistant. A skin lesion image was analyzed
by an AI classifier with these results:
- Predicted condition: {prediction}
- Confidence: {confidence * 100:.1f}%
- Triage guidance: {triage_message}

Write a short, warm, easy-to-understand summary (3-4 sentences) for a patient explaining
this result. Do not provide a medical diagnosis. Encourage consulting a real doctor.
Avoid alarming language even for concerning results — stay calm and informative."""

    response = client.chat.completions.create(
        model  = "openai/gpt-oss-120b",
        messages = [{"role": "user" ,"content": prompt}],
        temperature = 0.4,
        max_tokens = 200,



    )
    return response.choices[0].message.content