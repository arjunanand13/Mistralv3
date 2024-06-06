from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3",token=YOUR_TOKEN)

def format_prompt(message, history, system_prompt=None):
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    if system_prompt:
        prompt += f"[SYS] {system_prompt} [/SYS]"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(
    prompt, history, system_prompt=None, temperature=0.2, max_new_tokens=1024, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history, system_prompt)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=True)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output

demo = gr.ChatInterface(
    fn=generate,
    title="MISTRAL 7B V3 CHATBOT",
    
)

demo.queue().launch(show_api=False)
