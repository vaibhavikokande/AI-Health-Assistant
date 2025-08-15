import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("phi1.5", trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained("phi1.5", trust_remote_code=True)

def generate_answer(question, model):
    prompt = "Answer the following question: " + question
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )[0]

    answer = tokenizer.decode(output, skip_special_tokens=True)
    
    # Remove the prompt from the decoded text if it appears again
    return answer.replace(prompt, "").strip()

def chatbot(question, history):
    answer = generate_answer(question, llm_model)
    words = answer.split()
    for i in range(len(words)):
        time.sleep(0.05)  # Typing delay per word
        yield " ".join(words[:i+1])  # Send partial response

custom_theme = gr.themes.Base(
    primary_hue="blue",
    font=[gr.themes.GoogleFont("Fira Sans")],
    radius_size=gr.themes.sizes.radius_xxl,
    text_size=gr.themes.sizes.text_lg,
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.ChatInterface(
        fn=chatbot,
        title="🤖 AI Health Assistant",
        description="Ask me any general health-related questions! I respond like a doctor bot 🩺",
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Type your health question here...", scale=7),
        theme=custom_theme
    )

if __name__ == "__main__":
    demo.launch()
