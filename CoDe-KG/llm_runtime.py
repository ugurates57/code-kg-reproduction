import hashlib
import json
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError


class GenerationError(RuntimeError):
    """A document-local API failure that may be recorded and resumed."""


class TruncatedResponseError(GenerationError):
    pass


class PermanentAPIError(RuntimeError):
    pass


def atomic_json(path, value):
    """Replace a JSON file atomically after flushing its contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_seed(seed=4000):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Local seeds do not guarantee reproducible API responses.


class LLM:
    def __init__(self, model_name):
        load_dotenv(Path(__file__).resolve().parent.parent / '.env')
        self.original_model_name = model_name
        self.model_name = os.environ['LLM_MODEL']
        self.base_url = os.environ['LLM_API_BASE'].rstrip('/')
        self.cache_dir = Path(os.environ.get('LLM_CACHE_DIR', 'outputs/llm_cache'))
        self.use_cache = os.environ.get('LLM_CACHE_BYPASS', '0') != '1'
        self.client = OpenAI(api_key=os.environ['LLM_API_KEY'], base_url=self.base_url,
                             timeout=120.0, max_retries=0)
        self.sleep = time.sleep
        self.max_attempts = 5

    def generate_batch(self, prompts, max_new_tokens, batch_size, temperature,
                       top_k=40, top_p=0.9, do_sample=True,
                       repetition_penalty=1.0, return_full_text=False):
        if return_full_text:
            raise ValueError('Only return_full_text=False is supported')
        results = []
        for messages in prompts:
            if not isinstance(messages, list):
                raise TypeError('Expected chat messages')
            request = dict(model=self.model_name, messages=messages,
                           max_tokens=max_new_tokens,
                           temperature=temperature if do_sample else 0.0, top_p=top_p)
            identity = dict(schema=1, gateway=self.base_url, request=request)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            path = self.cache_dir / (key + '.json')
            if self.use_cache and path.exists():
                cached = json.loads(path.read_text())
                if cached['identity'] != identity or not cached['text'].strip() or cached['finish_reason'] != 'stop':
                    raise ValueError(f'Invalid cache entry: {path}')
                print(f'[CACHE] hit {key}', flush=True)
                results.append([{'generated_text': cached['text']}])
                continue
            for attempt in range(self.max_attempts):
                self.sleep(7)
                try:
                    response = self.client.chat.completions.create(**request)
                    break
                except (APIConnectionError, APIStatusError) as exc:
                    status = getattr(exc, 'status_code', None)
                    transient = isinstance(exc, APIConnectionError) or status in (408, 409, 429) or (status is not None and status >= 500)
                    if not transient:
                        raise PermanentAPIError(f'Permanent gateway error: HTTP {status}; check credentials, model and configuration') from None
                    if attempt + 1 == self.max_attempts:
                        raise GenerationError(f'{type(exc).__name__}: HTTP {status}; exhausted {self.max_attempts} attempts') from None
                    delay = 65 if status == 429 else min(2 ** (attempt + 1), 30)
                    headers = getattr(getattr(exc, 'response', None), 'headers', {})
                    retry_after = headers.get('retry-after')
                    if retry_after:
                        try:
                            delay = max(0, float(retry_after))
                        except ValueError:
                            try:
                                delay = max(0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (ValueError, TypeError, OverflowError):
                                pass
                    if not 0 <= delay <= 3600:
                        raise GenerationError('Retry-After exceeds the bounded one-hour retry window') from None
                    print(f'[API] {type(exc).__name__} HTTP {status}; retry {attempt + 1}/{self.max_attempts - 1} in {delay}s', flush=True)
                    self.sleep(delay)
            choice = response.choices[0]
            if choice.finish_reason == 'length':
                raise TruncatedResponseError(f'Output token limit reached ({max_new_tokens}); response discarded')
            text = choice.message.content
            if choice.finish_reason != 'stop' or not text or not text.strip():
                raise GenerationError(f'Invalid response: finish_reason={choice.finish_reason}, empty={not bool(text and text.strip())}')
            if self.use_cache:
                atomic_json(path, dict(identity=identity, text=text, finish_reason='stop'))
            results.append([{'generated_text': text}])
        return results
