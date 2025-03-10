import argparse

from src.base import Generator, StreamProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming think summary.')
    parser.add_argument('--thinker_name', type=str, required=True)
    parser.add_argument('--thinker_uri', type=str, required=True)
    parser.add_argument('--thinker_tokenizer_path', type=str, default=None)
    parser.add_argument('--summarizer_name', type=str, required=True)
    parser.add_argument('--summarizer_uri', type=str, required=True)
    parser.add_argument('--summarizer_tokenizer_path', type=str, default=None)
    return parser.parse_args()


class BaseClient:
    def __init__(self, args, summary_trigger_size=100) -> None:
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
            summary_trigger_size=summary_trigger_size
        )
    
    def process(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        thinking_stream = self.thinker.stream_chat(messages)

        summary_header_printed = False
        answer_header_printed = False
        for event_type, content in self.summary_processor.process_thinking_stream(thinking_stream):
            if event_type == "summary":
                if not summary_header_printed:
                    yield "<summary>\n"
                    summary_header_printed = True
                yield content
            elif event_type == "answer":
                if not answer_header_printed:
                    yield "\n</summary>\n"
                    answer_header_printed = True
                yield content


def main():
    args = parse_args()
    client = BaseClient(args, summary_trigger_size=200)
    while True:
        query = input("\n\n请输入问题（Ctrl+C退出）: ")
        for conent in client.process(query):
            print(conent, end='')


if __name__ == "__main__":
    main()
