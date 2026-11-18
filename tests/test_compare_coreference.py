"""Offline checks for the coreference-only comparison."""
import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / 'CoDe-KG/scripts/compare_coreference.py'
spec = importlib.util.spec_from_file_location('compare_coreference', PATH)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def annotation(start=0, end=0, expression='ADC', target='adenocarcinoma'):
    return dict(StartToken=start, EndToken=end, Expression=expression, RefersTo=target)


class ComparisonTests(unittest.TestCase):
    def test_duplicates_invalid_and_mismatch_are_separate(self):
        a = annotation()
        diagnostic, tuples = comparison.diagnostics(
            [a, a, annotation(start=9, end=9), annotation(start=1, end=1)], '(ADC) other')
        self.assertEqual(diagnostic['duplicate_annotations'], 1)
        self.assertEqual(diagnostic['duplicate_tuples'], 1)
        self.assertEqual(len(diagnostic['invalid_annotations']), 1)
        self.assertEqual(len(diagnostic['source_expression_mismatches']), 1)
        self.assertEqual(len(tuples), 2)

    def test_normalization_preserves_case_and_internal_punctuation(self):
        self.assertEqual(comparison.normalize(' (ADC),  disability-adjusted '), 'ADC disability-adjusted')
        self.assertNotEqual(comparison.normalize('ADC'), comparison.normalize('adc'))

    def test_missing_baseline_is_not_an_empty_prediction(self):
        report = comparison.compare({'394': 'ADC'}, {'394': [annotation()]}, {}, {})
        self.assertTrue(report['missing_baseline_394'])
        baseline = report['models']['baseline']
        self.assertEqual(baseline['completed_documents'], 0)
        self.assertIsNone(baseline['documents']['394']['strict_diagnostic'])
        self.assertEqual(baseline['strict_diagnostic_completed_documents_only'], {})

    def test_invalid_gold_flagged_separately(self):
        record = dict(original_text='ADC', coreference=dict(parsed_references=[annotation()]))
        report = comparison.compare({'x': 'ADC'}, {'x': [annotation(end=99)]}, {'x': record}, {})
        self.assertEqual(len(report['gold']['x']['invalid_annotations']), 1)
        counts = report['models']['baseline']['documents']['x']['strict_diagnostic']
        self.assertEqual(counts['gold_unique'], 0)
        self.assertEqual(counts['false_positive'], 1)

    def test_original_replacement_receives_duplicates_and_invalid_spans(self):
        refs = [annotation(), annotation(target='second'), annotation(start=8, end=8)]
        numbered = comparison.format_pairs(comparison.tokens_with_indices('(ADC) text'))
        self.assertEqual(comparison.apply_coref_switch(numbered, refs), 'second text')
        self.assertEqual(len(refs), 3)


if __name__ == '__main__':
    unittest.main()
