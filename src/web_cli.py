import argparse
import gradio as gr

from src.base import Generator, StreamProcessor


class GradioClient:
    def __init__(self, args):
        self.thinker = Generator(
            base_url=args.thinker_uri,
            model_name=args.thinker_name,
            tokenizer_path=args.thinker_tokenizer_path
        )
        
        summarizer = Generator(
            base_url=args.summarizer_uri,
            model_name=args.summarizer_name,
            tokenizer_path=args.summarizer_tokenizer_path
        )
        self.summary_processor = StreamProcessor(
            summarizer,
            summary_trigger_size=512
        )
    
    def stream_response(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        thinking_stream = self.thinker.stream_chat(
            messages,
            # frequency_penalty=1.0,
            # temperature=0.7,
            # top_p=0.8,
            max_tokens=1024 * 16
        )
        
        summary_buffer = []
        answer_buffer = []
        summary_stream = self.summary_processor.process_thinking_stream(
            thinking_stream,
            frequency_penalty=1.05,
            temperature=0.7,
            top_p=0.8,
            max_tokens=1024
        )
        for event_type, content in summary_stream:
            if event_type == "summary":
                summary_buffer.append(content)
                yield {
                    "summary": "".join(summary_buffer),
                    "answer": ""
                }
            elif event_type == "answer":
                answer_buffer.append(content)
                yield {
                    "summary": "".join(summary_buffer),
                    "answer": "".join(answer_buffer)
                }


def create_gradio_interface(args):
    client = GradioClient(args)
    
    with gr.Blocks(title="Streaming Think Summary") as demo:
        gr.Markdown("# 思考摘要")
        
        with gr.Column():
            input_box = gr.Textbox(
                label="输入问题",
                placeholder="请输入您的问题...",
                scale=1
            )
            
            submit_btn = gr.Button("提交", variant="primary", scale=1)
            
            summary_box = gr.Textbox(
                label="实时摘要",
                interactive=False,
                lines=8,
                scale=2
            )
            
            answer_box = gr.Textbox(
                label="最终回答",
                interactive=False,
                lines=5,
                scale=2
            )
        
        def process_query(prompt):
            for response in client.stream_response(prompt):
                yield response["summary"], response["answer"]
        
        submit_btn.click(
            fn=process_query,
            inputs=input_box,
            outputs=[summary_box, answer_box]
        )
    
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming think summary.')
    parser.add_argument('--thinker_name', type=str, required=True)
    parser.add_argument('--thinker_uri', type=str, required=True)
    parser.add_argument('--thinker_tokenizer_path', type=str, default=None)
    parser.add_argument('--summarizer_name', type=str, required=True)
    parser.add_argument('--summarizer_uri', type=str, required=True)
    parser.add_argument('--summarizer_tokenizer_path', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = create_gradio_interface(args)
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
