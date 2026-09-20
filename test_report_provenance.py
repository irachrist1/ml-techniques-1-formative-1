"""Pin every numeric claim in the report to a committed artifact.

`output/report.md` is hand-edited prose, so it is not regenerated from
`create_report.py` any more. That creates a failure mode worth guarding: the
tables in the report are interpolated from results, while a sentence next to a
table could quietly keep a stale number that no longer matches.

These tests assert that each result value the generator interpolates still
appears both in the generated text and in the submitted report. If a rerun moves
any of them, this fails and names the value, instead of leaving the report
disagreeing with its own evidence.

Unlike the other test modules, this one reads real committed results rather than
artificial fixtures. It asserts consistency between artifacts; it does not assert
that any particular value is correct.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from create_report import build_report

ROOT = Path(__file__).resolve().parent

#: Values the generator computes from committed artifacts, which must also appear
#: in the submitted report. Labelled so a failure says which claim broke.
PINNED = [
    ('headline test RMSE, busiest area', '118.16'),
    ('persistence RMSE, busiest area', '134.88'),
    ('RidgeAR parameter count', '145'),
    ('daily accumulator size in MB', '17.28'),
    ('superseded single-run memory reduction', '86.4'),
    ('ADF p at the 30-lag ceiling', '1.35e-27'),
    ('ADF p at the 168-lag ceiling', '9.57e-04'),
    ('round 2 CNN validation RMSE', '160.28'),
    ('round 3 CNN validation RMSE', '160.09'),
    ('round 3 relative change', '0.12%'),
    ('worst learned fit vs persistence', '39.76'),
    ('persistence on that area', '39.62'),
    ('worst learned fit relative loss', '0.35%'),
    ('RidgeAR validation RMSE, busiest area', '152.39'),
    ('activity before the failure spike', '414.87'),
    ('activity at the failure spike', '717.77'),
    ('activity after the failure spike', '348.86'),
    ('unvalidated read timing', '1.8 s'),
]

#: Known, accepted formatting difference. The benchmark artifact records a median
#: chunked wall time of 3.7834 s. The submitted report truncates that to "3.7 s";
#: the generator now prints the artifact value to two decimals. Same measurement,
#: different rounding, recorded here so it stays a deliberate choice.
KNOWN_ROUNDING_DIFFERENCES = [('chunked wall seconds', '3.78 s', '3.7 s')]


class ReportProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generated = build_report()
        cls.submitted = (ROOT / 'output/report.md').read_text()

    def test_every_pinned_value_is_produced_by_the_generator(self):
        for label, value in PINNED:
            with self.subTest(label):
                self.assertIn(value, self.generated, f'{label}: generator no longer produces {value}')

    def test_every_pinned_value_still_appears_in_the_submitted_report(self):
        for label, value in PINNED:
            with self.subTest(label):
                self.assertIn(value, self.submitted, f'{label}: report no longer states {value}')

    def test_known_rounding_differences_are_still_exactly_those(self):
        for label, generated, submitted in KNOWN_ROUNDING_DIFFERENCES:
            with self.subTest(label):
                self.assertIn(generated, self.generated)
                self.assertIn(submitted, self.submitted)

    def test_the_generator_refuses_to_invent_a_video_link(self):
        self.assertIn('pending', build_report().lower())
        self.assertIn('https://example.org/v', build_report('https://example.org/v'))


if __name__ == '__main__':
    unittest.main()
