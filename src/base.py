import argparse

from openai import OpenAI
from transformers import AutoTokenizer


# SUMMARY_PROMPT = '''
# 你是一个专业的文本摘要生成器。请根据以下要求总结提供的思考内容：

# 仅提取关键步骤和结论，去除示例、详细解释及重复内容。无需二次总结思考内容。
# 使用简洁的中文分点陈述，保留原有逻辑结构。
# 不添加任何分析、建议或格式模板，仅输出纯文本总结。

# 思考内容：
# {content}
# '''.strip()


SUMMARY_PART_PROMPT = '''
你负责提炼输入的思维链片段，生成处理流程的核心方法论节点，用于清晰展示当前思维链的执行进程。

提炼规则：
1. 每个节点用独立判断句表述正在执行的方法论步骤。
2. 按处理顺序排列，节点间空行分隔。
3. 特别注意：排除思维链中所有与输出格式、内容长度或结构相关的具体要求。

输出格式：
1. 单句独立成行。
2. 句间保留空行。
3. 与输入思维链保持相同语言体系。

输入的思维链片段（实时生成中）：
{content}
'''.strip()

SUMMARY_TOTAL_PROMPT = '''
你负责提炼输入的完整思维链，生成处理流程的核心方法论节点，用于清晰展示当前思维链的执行进程。

提炼规则：
1. 每个节点用独立判断句表述正在执行的方法论步骤。
2. 按处理顺序排列，节点间空行分隔。
3. 特别注意：排除思维链中所有与输出格式、内容长度或结构相关的具体要求。

输出格式：
1. 单句独立成行。
2. 句间保留空行。
3. 与输入思维链保持相同语言体系。

输入的完整思维链：
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
            **kwargs
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_generate(self, prompt, **kwargs):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=True,
            **kwargs
        )
        for chunk in response:
            if chunk.choices[0].text:
                yield chunk.choices[0].text
    
    def generate(self, prompt, **kwargs):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            stream=False,
            **kwargs
        )
        return response.choices[0].text


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

    def process_thinking_stream(self, thinking_stream, **kwargs):
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
                prompt_template = SUMMARY_PART_PROMPT if self.is_reasoning_phase else SUMMARY_TOTAL_PROMPT
                yield from self._handle_summary_generation(prompt_template, **kwargs)
            
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

    def _handle_summary_generation(self, prompt_template, **kwargs):
        current_reason = ''.join(self.reason_buffer)
        current_summary = ''.join(self.summary_buffer)

        prompt = prompt_template.format(content=current_reason)
        summary_stream = self._generate_summary(prompt, current_summary, **kwargs)
        for summary_chunk in summary_stream:
            self.summary_buffer.append(summary_chunk)
            yield ('summary', summary_chunk)
        self.trigger_summary = False

    def _generate_summary(self, prompt, summary_text, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        input_text = self.summarizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + summary_text
        return self.summarizer.stream_generate(input_text, **kwargs)
        # return [self.summarizer.generate(input_text, **kwargs)] # debug

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
    thinking_stream = thinker.stream_chat(
        messages, frequency_penalty=1.0, temperature=0.6, top_p=0.95
    )
    summary_processor = StreamProcessor(summarizer, summary_trigger_size=100)
    
    summary_header_printed = False
    answer_header_printed = False
    summary_stream = summary_processor.process_thinking_stream(
        thinking_stream, frequency_penalty=1.05, temperature=0.7, top_p=0.8
    )
    for event_type, content in summary_stream:
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
