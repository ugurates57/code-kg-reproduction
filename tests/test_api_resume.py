"""Offline robustness checks; no gateway or model access."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
from openai import RateLimitError, APITimeoutError, AuthenticationError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CoDe-KG'))
from llm_runtime import LLM, GenerationError, TruncatedResponseError, PermanentAPIError
from pipeline_resume import read_traces, reconstruct_csv, FIELDS


def response(reason='stop'):
    return NS(choices=[NS(finish_reason=reason, message=NS(content='answer'))])


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.llm = LLM.__new__(LLM)
        self.llm.model_name = 'fake'
        self.llm.base_url = 'https://fake.invalid'
        self.llm.cache_dir = Path(self.temp.name)
        self.llm.use_cache = True
        self.llm.sleep = Mock()
        self.llm.max_attempts = 3
        self.create = Mock(return_value=response())
        self.llm.client = NS(chat=NS(completions=NS(create=self.create)))

    def call(self, messages=None, **kwargs):
        return self.llm.generate_batch([messages or [{'role': 'user', 'content': 'x'}]],
                                      batch_size=1, max_new_tokens=kwargs.pop('max_new_tokens', 10),
                                      temperature=.7, **kwargs)

    def test_cache_and_identity(self):
        self.assertEqual(self.call(), [[{'generated_text': 'answer'}]])
        self.assertEqual(len(list(self.llm.cache_dir.glob('*.json'))), 1)
        self.call()
        self.assertEqual(self.create.call_count, 1)
        self.call([{'role': 'user', 'content': 'changed'}])
        self.call(max_new_tokens=11)
        self.llm.model_name = 'other'
        self.call()
        self.llm.base_url = 'https://other.invalid'
        self.call()
        self.assertEqual(self.create.call_count, 5)

    def test_transients_bounded(self):
        req = httpx.Request('POST', 'https://fake.invalid')
        for error in [RateLimitError('rate', response=httpx.Response(429, request=req, headers={'retry-after': '2'}), body=None), APITimeoutError(request=req)]:
            self.create.reset_mock()
            self.create.side_effect = error
            with self.assertRaises(GenerationError):
                self.call()
            self.assertEqual(self.create.call_count, 3)
        self.assertTrue(any(call.args == (2.0,) for call in self.llm.sleep.call_args_list))

    def test_permanent(self):
        req = httpx.Request('POST', 'https://fake.invalid')
        self.create.side_effect = AuthenticationError('auth', response=httpx.Response(401, request=req), body=None)
        with self.assertRaises(PermanentAPIError):
            self.call()
        self.assertEqual(self.create.call_count, 1)

    def test_partial_batch_resume(self):
        self.create.side_effect = [response(), response('length')]
        prompts = [[{'role': 'user', 'content': str(i)}] for i in range(2)]
        with self.assertRaises(TruncatedResponseError):
            self.llm.generate_batch(prompts, 10, 1, .7)
        self.assertEqual(len(list(self.llm.cache_dir.glob('*.json'))), 1)
        self.create.side_effect = None
        self.llm.generate_batch(prompts, 10, 1, .7)
        self.assertEqual(self.create.call_count, 3)

    def test_document_continue_and_reconstruction(self):
        import run_pipeline as pipeline
        from unittest.mock import patch
        import pandas as pd
        output = Path(self.temp.name) / 'triples.csv'
        csv = Path(self.temp.name) / 'input.csv'
        pd.DataFrame([{'doc_id': '1', 'abstract': 'one'}, {'doc_id': '2', 'abstract': 'two'}]).to_csv(csv, index=False)
        fake = Mock()
        fake.model_name = 'fake'
        fake.generate_batch.side_effect = [GenerationError('timeout'), [[{'generated_text': 'answer'}]]]
        bert = Mock()
        bert.exists.return_value = True
        bert.predict_batch.return_value = [{'sentence': 'two', 'predicted_label': 'Simple Sentence'}]
        with patch.object(sys, 'argv', ['run', '--input_csv', str(csv), '--id_col', 'doc_id', '--out_csv', str(output)]), patch.object(pipeline, 'LLM', return_value=fake), patch.object(pipeline, 'configuration', return_value={'test': 1}), patch.object(pipeline, 'set_seed'), patch.object(pipeline, 'BertSentenceClassifier', return_value=bert), patch.object(pipeline, 'split_sentences', return_value=['two']), patch.object(pipeline, 'extract_relationships', return_value=[{'sentence': 'two', 'triples': [{'Entity 1': 'a', 'Relationship': 'r', 'Entity 2': 'b'}]}]):
            pipeline.main()
        trace = output.with_suffix('.trace.jsonl')
        records = read_traces([trace, trace])
        self.assertEqual(list(records), ['2'])
        reconstruct_csv(records, output)
        first = output.read_bytes()
        reconstruct_csv(read_traces([trace, trace]), output)
        self.assertEqual(first, output.read_bytes())
        self.assertEqual(len(pd.read_csv(output)), 1)
        self.assertEqual(json.loads(output.with_suffix('.failures.jsonl').read_text())['doc_id'], '1')


if __name__ == '__main__':
    unittest.main()
