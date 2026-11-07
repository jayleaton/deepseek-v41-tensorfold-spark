"""Handwritten CPU checks. Authoring only on dev; parent runs on Sparks under 2GiB.

No task generator, historical inference, model, network, or real task generation.
Run there: PYTHONDONTWRITEBYTECODE=1 python3 disjoint_check.py
"""
import tempfile
import unittest

from disjointness import (CollisionIndex, DisjointnessError, UnresolvedOverlap,
                          canonical_inputs, integer_ast, normalize_prompt,
                          prior_signature, prompt_sha256, semantic_signature)
from exclusion_sources import load_exclusions


def candidate(family, inputs, prompt='A distinct handwritten user question.'):
    return {'id': 'handwritten-' + family, 'family': family, 'slot': 0,
            'attempt': 0, 'inputs': inputs, 'prompt': prompt, 'suite': 'compute_dl'}


class DisjointChecks(unittest.TestCase):
    def test_six_reskinned_instances(self):
        examples = [
            ('integer', {'a': 7, 'b': 9, 'c': 4}, 'Evaluate 9*7+4 exactly.'),
            ('modular', {'a': 7, 'e': 9, 'm': 13}, 'What is 7^9 mod 13? Give the exact integer.'),
            ('division', {'a': 23, 'b': 5}, 'What is 23 divided by 5? Give quotient and remainder.'),
            ('dp', {'rows': 2, 'cols': 2, 'matrix': [[1, 2], [3, 4]]},
             'Find the minimum sum from top-left to bottom-right, right and down only, both endpoints included. Grid: [[1,2],[3,4]].'),
            ('graph', {'n': 3, 'edges': [[0, 1, 3], [1, 2, 4]]},
             'A directed graph has vertices 0 through 2. Find minimum cost from vertex 0 to vertex 2. Edges: [[1,2,4],[0,1,3]].'),
            ('aggregation', {'count': 3, 'records': [['A', 2, 3], ['A', 2, 3], ['B', 1, -1]]},
             'For each category compute sum of quantity*unit_amount. Transactions: [["B",1,-1],["A",2,3],["A",2,3]].'),
        ]
        for family, inputs, old_prompt in examples:
            with self.subTest(family=family):
                index = CollisionIndex([{'id': 'prior-' + family, 'prompt': old_prompt}])
                hit = index.check(candidate(family, inputs))
                self.assertIsNotNone(hit)
                self.assertIn('operation-operands', [entry['reason'] for entry in hit['collisions']])

    def test_commutative_multiplication_and_old_structured_ops(self):
        self.assertEqual(integer_ast('7*9+4'), integer_ast('9*7+4'))
        for family, inputs, op, args in [
                ('integer', {'a': 7, 'b': 9, 'c': 0}, '*', ['9', '7']),
                ('modular', {'a': 7, 'e': 9, 'm': 13}, 'powmod', ['7', '9', '13']),
                ('division', {'a': 20, 'b': 5}, 'exact-div', ['20', '5'])]:
            index = CollisionIndex([{'id': 'old', 'prompt': 'Original structured arithmetic.',
                                     'spec': {'op': op, 'args': args}}])
            self.assertIsNotNone(index.check(candidate(family, inputs)))

    def test_duplicates_preserved_grid_order_and_graph_order(self):
        one = {'count': 1, 'records': [['A', 2, 3]]}
        two = {'count': 2, 'records': [['A', 2, 3], ['A', 2, 3]]}
        self.assertNotEqual(semantic_signature('aggregation', one), semantic_signature('aggregation', two))
        grid = {'rows': 2, 'cols': 2, 'matrix': [[1, 2], [3, 4]]}
        other = dict(grid, matrix=[[3, 4], [1, 2]])
        self.assertNotEqual(canonical_inputs('dp', grid), canonical_inputs('dp', other))
        graph = {'n': 3, 'edges': [[0, 1, 2], [1, 2, 3]]}
        self.assertEqual(canonical_inputs('graph', graph), canonical_inputs('graph', dict(graph, edges=list(reversed(graph['edges'])))))

    def test_exact_historical_decimals_and_rationals(self):
        def old(op, args):
            return {'id': 'decimal-old', 'prompt': 'Original exact arithmetic.',
                    'spec': {'op': op, 'args': args}}

        integral = candidate('integer', {'a': 1, 'b': 2, 'c': 0})
        for args in (['1.0', '2'], ['2.000', '1e0'], ['2/2', '2.0']):
            with self.subTest(args=args):
                self.assertIsNotNone(CollisionIndex([old('*', args)]).check(integral))
                self.assertEqual(prior_signature(old('*', args)), integer_ast('1*2'))
        decimal = old('*', ['1.50', '2.0'])
        rational = old('*', ['2', '3/2'])
        self.assertEqual(prior_signature(decimal), prior_signature(rational))
        self.assertIsNone(CollisionIndex([decimal]).check(integral))
        # Same numeric answer is not reused operation-plus-operands.
        self.assertIsNone(CollisionIndex([decimal]).check(
            candidate('integer', {'a': 3, 'b': 1, 'c': 0})))
        exact = prior_signature(old('*', ['-36933814.9', '-538036']))
        self.assertEqual(exact[0], 'integer-expression')
        self.assertIn(('rational', -369338149, 10), exact[1][1:])
        self.assertEqual(prior_signature(old('+', ['0.0', '1.0'])), integer_ast('0+1'))
        self.assertEqual(prior_signature(old('-', ['2.0', '1.0'])), integer_ast('2-1'))
        self.assertEqual(prior_signature(old('exact-div', ['20.0', '5.0'])), ('division', 20, 5))
        self.assertEqual(prior_signature(old('exact-div', ['1.5', '2'])),
                         ('division', ('rational', 3, 2), 2))
        for value in ('NaN', 'Infinity', '1/0', '__import__("os")', True, 1.5):
            with self.subTest(invalid=value), self.assertRaises(DisjointnessError):
                prior_signature(old('*', [value, '2']))
        with self.assertRaises(DisjointnessError):
            prior_signature(old('powmod', ['1.5', '2', '3']))

    def test_other_historical_structured_operations(self):
        for op, args in [('powmod', ['7', '9', '13']),
                         ('digit-sum', ['8', '121']),
                         ('power-difference', ['81', '59']),
                         ('factorial-ratio', ['37', '28']),
                         ('stock', ['100', '2', '3', '4'])]:
            with self.subTest(op=op):
                row = {'id': 'old-' + op, 'prompt': 'Original structured operation.',
                       'spec': {'op': op, 'args': args}}
                self.assertIsNotNone(prior_signature(row))

    def test_nfkc_casefold_and_whitespace_full_prompt(self):
        self.assertEqual(normalize_prompt(' ＳＴＲＡＳＳＥ\t7 * 9\n'), normalize_prompt('Straße 7 * 9'))
        old = {'id': 'old', 'prompt': ' ＳＴＲＡＳＳＥ\t7 * 9\n',
               'spec': {'op': '+', 'args': ['1', '2']}}
        hit = CollisionIndex([old]).check(candidate('integer', {'a': 3, 'b': 5, 'c': 1}, 'Straße 7 * 9'))
        self.assertIn('normalized-full-prompt', [x['reason'] for x in hit['collisions']])
        self.assertNotEqual(normalize_prompt('7*9'), normalize_prompt('7+9'))

    def test_missing_exclusion_is_not_empty(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(DisjointnessError):
                load_exclusions({'w9_root': root})

    def test_ambiguous_problem_blocks_and_resolution_is_bound(self):
        row = {'id': 'unknown', 'prompt': 'Use the enclosed grid to determine the optimal cost.'}
        with self.assertRaises(UnresolvedOverlap):
            CollisionIndex([row])
        resolution = {'prompt_sha256': '0' * 64, 'original_evidence': {'path': 'fixture-original', 'sha256': '0' * 64, 'location': 'handwritten fixture'},
                      'reason': 'Handwritten fixture only.', 'decision': 'nonoverlap'}
        with self.assertRaises(UnresolvedOverlap):
            CollisionIndex([row], {'unknown': resolution})
        resolution['prompt_sha256'] = prompt_sha256(row['prompt'])
        self.assertIsNone(CollisionIndex([row], {'unknown': resolution}).check(
            candidate('integer', {'a': 2, 'b': 3, 'c': 4})))

    def test_shared_program_concept_is_not_concrete_example(self):
        row = {'id': 'program', 'prompt': 'Write a Python function to calculate a product. Example: 7*9+4.'}
        self.assertIsNone(CollisionIndex([row]).check(candidate('integer', {'a': 7, 'b': 9, 'c': 4})))
        mbpp = dict(row, prompt='Write the Python function described below. Multiply two supplied integers. Example: 7*9+4.')
        self.assertIsNone(CollisionIndex([mbpp]).check(candidate('integer', {'a': 7, 'b': 9, 'c': 4})))
        mbpp['prompt'] = 'Write the Python function described below. Given A=7, evaluate an unspecified fixed expression.'
        with self.assertRaises(UnresolvedOverlap):
            CollisionIndex([mbpp])

    def test_safe_ast_and_bad_typed_input(self):
        for expression in ('__import__("os").system("false")', 'a*b+4', 'True*2', '1.0*2'):
            with self.assertRaises(DisjointnessError):
                integer_ast(expression)
        with self.assertRaises(DisjointnessError):
            canonical_inputs('integer', {'a': True, 'b': 2, 'c': 3})

    def test_within_new_set_and_common_constants(self):
        index = CollisionIndex([])
        first = candidate('integer', {'a': 7, 'b': 9, 'c': 4})
        index.add(first)
        self.assertIsNotNone(index.check(first))
        other = candidate('integer', {'a': 7, 'b': 10, 'c': 4}, 'Another instance with shared constants.')
        other['id'] = 'handwritten-other'
        self.assertIsNone(index.check(other))


if __name__ == '__main__':
    unittest.main()
