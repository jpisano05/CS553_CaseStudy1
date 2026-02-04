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
    
    #prepare message
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local == True:
        #run local model
        
        MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        
        print("Pipeline created")
        
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        print("Prompt built")
        
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        print("Output gotten")
        
        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()
        
        pass
    else:
        #run api model
        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")
        
        for message in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = message.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content

            response += token
            yield response


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
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
