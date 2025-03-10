import argparse

from openai import OpenAI
from transformers import AutoTokenizer


SUMMARY_PROMPT = '''
你是一个专业的文本摘要生成器。请根据以下要求总结提供的思考内容：

仅提取关键步骤和结论，去除示例、详细解释及重复内容。无需二次总结思考内容。
使用简洁的中文分点陈述，保留原有逻辑结构。
不添加任何分析、建议或格式模板，仅输出纯文本总结。

思考内容：
{content}
'''.strip()


class Generator:
    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        model_name: str = None,
        tokenizer_path: str = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name or self._get_default_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=True
        ) if tokenizer_path else None

    def _get_default_model(self):
        assert self.client.models.list().data is not None
        return self.client.models.list().data[0].id

    def apply_chat_template(self, message, **kwargs):
        assert self.tokenizer is not None
        return self.tokenizer.apply_chat_template(message, **kwargs)

    def stream_chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=kwargs.get("max_token", 8192),
            temperature=kwargs.get("temperature", 0.7),
            frequency_penalty=kwargs.get("frequency_penalty", 1.05),
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_generate(self, prompt, **kwargs):
        max_token = kwargs.get("max_token", 8192)
        temperature = kwargs.get("temperature", 0.7)
        frequency_penalty = kwargs.get("frequency_penalty", 1.05)

        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            max_tokens=max_token,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
        )
        for chunk in response:
            if chunk.choices[0].text:
                yield chunk.choices[0].text


class StreamProcessor:
    def __init__(self, summarizer, summary_trigger_size=100):
        self.summarizer = summarizer
        self.summary_trigger_size = summary_trigger_size
        self.reset_status()
    
    def reset_status(self):
        self.reason_buffer = []
        self.summary_buffer = []
        self.answer_buffer = []
        self.is_reasoning_phase = True
        self.is_answer_phase = False
        self.trigger_summary = False

    def process_thinking_stream(self, thinking_stream):
        for chunk in thinking_stream:
            if self.is_reasoning_phase:
                self._process_reason_chunk(chunk)
            elif self.is_answer_phase:
                self._process_answer_chunk(chunk)
            else:
                chunk = chunk.lstrip('\n')
                if chunk:
                    self._process_answer_chunk(chunk)
                    self.is_answer_phase = True

            if self.trigger_summary and self.reason_buffer:
                yield from self._handle_summary_generation()
            
            yield from self._handle_answer_output()

        yield from self._flush_remaining_content()
        self.reset_status()

    def _process_reason_chunk(self, chunk):
        if "</think>" in chunk:
            parts = chunk.split("</think>")
            self.reason_buffer.append(parts[0])
            if len(parts) > 1:
                self._process_answer_chunk(parts[1].lstrip('\n'))
            self.is_reasoning_phase = False
            self.trigger_summary = True
        else:
            self.reason_buffer.append(chunk)
            if len(self.reason_buffer) % self.summary_trigger_size == 0:
                self.trigger_summary = True

    def _process_answer_chunk(self, chunk):
        self.answer_buffer.append(chunk)

    def _handle_summary_generation(self):
        current_reason = ''.join(self.reason_buffer)
        current_summary = ''.join(self.summary_buffer)
        summary_stream = self._generate_summary(current_reason, current_summary)
        for summary_chunk in summary_stream:
            self.summary_buffer.append(summary_chunk)
            yield ('summary', summary_chunk)
        self.trigger_summary = False

    def _generate_summary(self, reason_text, summary_text):
        prompt = SUMMARY_PROMPT.format(content=reason_text)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_text = self.summarizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + summary_text
        return self.summarizer.stream_generate(input_text)

    def _handle_answer_output(self):
        while self.answer_buffer:
            content = self.answer_buffer.pop(0)
            if content:
                yield ('answer', content)

    def _flush_remaining_content(self):
        while self.answer_buffer:
            content = self.answer_buffer.pop(0)
            if content:
                yield ('answer', content)

def parse_args():
    parser = argparse.ArgumentParser(description='Streaming think summary.')
    parser.add_argument('--thinker_name', type=str, required=True)
    parser.add_argument('--thinker_uri', type=str, required=True)
    parser.add_argument('--thinker_tokenizer_path', type=str, default=None)
    parser.add_argument('--summarizer_name', type=str, required=True)
    parser.add_argument('--summarizer_uri', type=str, required=True)
    parser.add_argument('--summarizer_tokenizer_path', type=str, default=None)
    return parser.parse_args()


def main(prompt):
    args = parse_args()
    
    thinker = Generator(
        base_url=args.thinker_uri,
        model_name=args.thinker_name,
        tokenizer_path=args.thinker_tokenizer_path
    )
    
    summarizer = Generator(
        base_url=args.summarizer_uri,
        model_name=args.summarizer_name,
        tokenizer_path=args.summarizer_tokenizer_path
    )

    messages = [{"role": "user", "content": prompt}]
    thinking_stream = thinker.stream_chat(messages)
    summary_processor = StreamProcessor(summarizer, summary_trigger_size=100)
    
    summary_header_printed = False
    answer_header_printed = False
    for event_type, content in summary_processor.process_thinking_stream(thinking_stream):
        if event_type == 'summary':
            if not summary_header_printed:
                print("\n【实时摘要】", flush=True)
                summary_header_printed = True
            print(content, end='', flush=True)
        elif event_type == 'answer':
            if not answer_header_printed:
                print("\n【最终答案】", flush=True)
                answer_header_printed = True
            print(content, end='', flush=True)


if __name__ == "__main__":
    main("1+1=?")
