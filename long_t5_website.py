import os
import sys

import fire
import gradio as gr
import torch
import transformers
from transformers import pipeline

# Main function
def main():
    base_model = "kworts/BARTxiv" ## you can download and save pertrained is another options
    model = pipeline(
    "summarization",
    base_model,
    device=0 if torch.cuda.is_available() else -1,)


    def evaluate(Text_to_summarize):
        
        result = model(Text_to_summarize)
        
        return result[0]["summary_text"]

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=25,
                label="Text_to_summarize",
                placeholder="add text to summarize",
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=25,
                label="Output"
            )
        ],
        title="Text summarization",
        description="This is attach with personal website",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    fire.Fire(main)

