import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import psutil

#Code for differentiation between running locally and API is based on Professor Paffenroth's Chatbot

def respond(
    message,
    history,
    max_tokens,
    hf_token: gr.OAuthToken,
    use_local: bool,
):
    
    SYSTEM_PROMPT = '''You are a coffee expert. Based on a user's taste profile, recommend them a type of coffee or espresso based drink.
                        1. The type of coffee bean (origin and variety)
                        2. The brew method
                        3. The type of drink
                        
                        Give a single paragraph and be short and specific.'''
    USER_PROMPT = message
    EXAMPLE_INPUT = '''Bright and citrusy'''
    EXAMPLE_OUTPUT = '''I recommend a medium-bodied Ethiopian Yirgacheffe brewed as a pour-over and served as a latte, highlighting bright citrus and floral notes.'''
    
    chat = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': EXAMPLE_INPUT},
        {'role': 'assistant', 'content': EXAMPLE_OUTPUT},
        {'role': 'user', 'content': USER_PROMPT}
    ]

    if use_local == True:
        #run local model
        
        cpu_start = psutil.cpu_percent(interval=None)
        
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
        
        outputs = pipe(
            chat,
            do_sample=False,
            max_new_tokens=max_tokens
        )
        
        #for cost analysis
        cpu_end = psutil.cpu_percent(interval=None)
        print("CPU used: ", cpu_end - cpu_start)
        
        response = outputs[0]['generated_text'][-1]['content'].strip()
        yield response
    else:
        #run api model

        client = InferenceClient(
            token=hf_token.token,
            model="openai/gpt-oss-20b",
        )

        completion = client.chat_completion(
            messages=chat,
            max_tokens=max_tokens,
            stream=False,
        )
        
        response = completion.choices[0].message.content.strip()
        
        
        yield response


chatbot = gr.ChatInterface(
    title="The Coffee Connoisseur",
    fn=respond,
    additional_inputs=[
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Checkbox(label="Use Local Model?", value = False),
    ],
)

with gr.Blocks(title="Coffee Connoisseur", css="""
               body, * {
                    color: #795548;
                    font-family: "Comic Sans MS" !important;
                }

                body {
                    background-color: #F8BBD0;
                }

                .gr-chatbot, .gr-chat-message {
                    background-color: #BCAAA4 !important;
                    color: #795548 !important;
                    font-family: "Comic Sans MS" !important;
                }

                .gr-button {
                    background-color: #BCAAA4 !important;
                    color: #795548 !important;
                    font-family: "Comic Sans MS" !important;
                }

                .gr-slider .gr-slider-track, .gr-slider .gr-slider-thumb {
                    background-color: #BCAAA4 !important;
                }

                .gr-checkbox .gr-checkbox-label {
                    color: #795548 !important;
                    font-family: "Comic Sans MS" !important;
                }
               """) as demo:
    gr.Markdown(
        """
        <div>
        <strong>Instructions:</strong><br>
        Enter a taste profile for a desired coffee drink and the Coffee Connoisseur will recommend you a drink.<br>
        For best results, keep inputs short like "Floral and Delicate" or "Chocolatey and nutty".
        </div>
        """,
    )    
    
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()