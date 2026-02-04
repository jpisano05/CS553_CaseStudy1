import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local: bool,
):
    global pipe
    
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    SYSTEM_PROMPT = 'Based on a taste profile, you are to recommend a type of coffee. This should include the type of bean, the brew method, and the type of coffee (latte, americano, etc.)'
    USER_PROMPT = message
    
    chat = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT}
    ]

    if use_local == True:
        #run local model
        
        outputs = pipe(
            chat,
            do_sample=False,
            max_new_tokens=4096
        )
        
        print("Output gotten")
        
        response = outputs[0]['generated_text'][-1]['content'].strip()
        yield response
    else:
        # run api model (non-streaming, chat-style)

        client = InferenceClient(
            token=hf_token.token,
            model="openai/gpt-oss-20b",
        )

        completion = client.chat_completion(
            messages=chat,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )

        response = completion.choices[0].message.content.strip()
        yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=1.0,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
        gr.Checkbox(label="Use Local Model?", value = False),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
