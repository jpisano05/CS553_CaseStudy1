import gradio as gr
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def respond(
    message,
    max_tokens,
    hf_login,
    use_local: bool,
    ):
    
    #get login
    hf_token = None
    if hf_login and "access_token" in hf_login:
        hf_token = hf_login["access_token"]
    
    #create pipe
    global pipe
    
    #setup the local model
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    #prompts
    SYSTEM_PROMPT = '''You are a coffee expert. Based on a user's taste profile, recommend them a type of coffee or espresso based drink.
                        1. The type of coffee bean (origin and variety)
                        2. The brew method
                        3. The type of drink
                        
                        Give a single paragraph and be short and specific.'''
    USER_PROMPT = message
    EXAMPLE_INPUT = '''Bright and citrusy'''
    EXAMPLE_OUTPUT = '''I recommend a medium-bodied Ethiopian Yirgacheffe brewed as a pour-over and served as a latte, highlighting bright citrus and floral notes.'''
    
    #setup chat
    chat = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': EXAMPLE_INPUT},
        {'role': 'assistant', 'content': EXAMPLE_OUTPUT},
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
        yield [USER_PROMPT, response]
    else:
        # run api model (non-streaming, chat-style)
        
        if hf_token is None or "access_token" not in hf_token:
            yield "Please log in with your Hugging Face account first."
            return

        client = InferenceClient(
            token=hf_token["access_token"],
            model="openai/gpt-oss-20b",
        )

        completion = client.chat_completion(
            messages=chat,
            max_tokens=max_tokens,
            stream=False,
        )

        response = completion.choices[0].message.content.strip()
        print(USER_PROMPT)
        print(response)
        yield [USER_PROMPT, response]


with gr.Blocks(title="Coffee Connoisseur") as demo:

    with gr.Sidebar():
        gr.Markdown("Settings:")
        max_tokens_slider = gr.Slider(
            minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"
        )
        use_local_checkbox = gr.Checkbox(label="Use Local Model?", value=False)
        hf_login = gr.LoginButton()
        
    gr.Markdown("The Coffee Connoisseur")
    gr.Markdown(
        "Enter a taste profile for a desired coffee drink and the Coffee Connoisseur will recommend you a drink."
        "For best results, keep inputs short like \"Floral and Delicate\" or \"Chocolatey and nutty\""
    )

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="Enter your taste profile",
                placeholder="e.g., Bright and citrusy, chocolatey, nutty...",
                lines=2
            )
        with gr.Column(scale=1):
            submit_button = gr.Button("Get Recommendation", variant="primary")

    chat_output = gr.Chatbot(label="~~Your Coffee Recommendation~~", height=400)

    submit_button.click(
        fn=respond,
        inputs=[user_input, 
                max_tokens_slider, 
                hf_login,
                use_local_checkbox],
    )


if __name__ == "__main__":
    demo.launch()
