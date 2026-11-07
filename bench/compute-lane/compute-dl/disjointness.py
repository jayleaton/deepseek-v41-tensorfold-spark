"""Fail-closed instance disjointness; never evaluate historical code.

A generic programming contract and its illustrative examples are not a concrete
instance of the new computation. Only the explicit standalone-program/ function
contract forms below receive that classification. Other unknown problems require
an original-problem resolution, bound to the complete prompt digest. No heuristic
that deletes numbers, operators, examples, or constants establishes disjointness.
"""
import ast
import hashlib
import json
import re
import unicodedata
from fractions import Fraction

FAMILIES = ('integer', 'modular', 'division', 'dp', 'graph', 'aggregation')


class DisjointnessError(ValueError):
    """Preparation failure: do not increment a generation attempt."""


class UnresolvedOverlap(DisjointnessError):
    def __init__(self, potential):
        self.potential = potential
        super().__init__('BLOCK: unresolved original problems: ' + json.dumps(potential, sort_keys=True))


def normalize_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise DisjointnessError('missing original user prompt')
    return ' '.join(unicodedata.normalize('NFKC', prompt).casefold().split())


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()


def _key(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def _int(value):
    if type(value) is not int:
        raise DisjointnessError('typed operand must be an integer, not bool/float/string')
    return value


def _old_number(value):
    """Exact historical decimal/rational operand; never accept binary floats.

    Fraction parses decimal text directly, including scientific notation. It
    never passes through float or evaluates a source expression. Canonical nodes
    below remain JSON serializable; denominator-one values use integer literals.
    """
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, str) and len(value) <= 20000 and re.fullmatch(
            r'[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,4})?|\d+/\d+)',
            value, flags=re.ASCII):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise DisjointnessError('invalid exact historical operand') from exc
    raise DisjointnessError('historical operand must be exact integer/decimal/rational text')


def _old_int(value):
    number = _old_number(value)
    if number.denominator != 1:
        raise DisjointnessError('historical operation requires an integral operand')
    return number.numerator


def _number_node(number):
    if number.denominator == 1:
        return ('literal', number.numerator)
    return ('rational', number.numerator, number.denominator)


def _binary(op, left, right):
    """Shared canonical AST construction for integer AST and old exact numbers."""
    if op == '-' and right[0] in ('literal', 'rational'):
        op, right = '+', (right[0], -right[1], *right[2:])
    if op == '+':
        if left == ('literal', 0):
            return right
        if right == ('literal', 0):
            return left
    if op in ('+', '*'):
        left, right = sorted((left, right), key=_key)
    return (op, left, right)


def _expr(node):
    if isinstance(node, ast.Constant):
        return ('literal', _int(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _expr(node.operand)
        if inner[0] != 'literal':
            raise DisjointnessError('only signed literals permitted')
        return ('literal', -inner[1] if isinstance(node.op, ast.USub) else inner[1])
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
        op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Pow: '**'}[type(node.op)]
        return _binary(op, _expr(node.left), _expr(node.right))
    raise DisjointnessError('nonliteral expression; no calls/names/execution allowed')


def integer_ast(expression):
    if not isinstance(expression, str) or len(expression) > 20000:
        raise DisjointnessError('invalid expression length')
    try:
        tree = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 256:
            raise DisjointnessError('expression too complex')
        return ('integer-expression', _expr(tree.body))
    except (SyntaxError, RecursionError) as exc:
        raise DisjointnessError('invalid literal expression') from exc


def canonical_inputs(family, inputs):
    """Canonical typed input key; graph/records unordered, duplicates retained."""
    if not isinstance(inputs, dict):
        raise DisjointnessError('inputs must be an object')
    expected = {'integer': {'a', 'b', 'c'}, 'modular': {'a', 'e', 'm'},
                'division': {'a', 'b'}, 'dp': {'rows', 'cols', 'matrix'},
                'graph': {'n', 'edges'}, 'aggregation': {'count', 'records'}}
    if family not in expected or set(inputs) != expected[family]:
        raise DisjointnessError('unexpected typed input fields: ' + str(family))
    x = dict(inputs)
    if family in ('integer', 'modular', 'division'):
        for value in x.values():
            _int(value)
        if family == 'modular' and (x['m'] <= 0 or x['e'] < 0):
            raise DisjointnessError('invalid modular operands')
        if family == 'division' and x['b'] <= 0:
            raise DisjointnessError('division requires positive divisor')
    elif family == 'dp':
        rows, cols = _int(x['rows']), _int(x['cols'])
        if rows <= 0 or cols <= 0 or not isinstance(x['matrix'], list) or len(x['matrix']) != rows:
            raise DisjointnessError('grid shape')
        for row in x['matrix']:
            if not isinstance(row, list) or len(row) != cols:
                raise DisjointnessError('grid row shape')
            for value in row:
                _int(value)
    elif family == 'graph':
        n = _int(x['n'])
        if n < 2 or not isinstance(x['edges'], list):
            raise DisjointnessError('graph shape')
        seen = set()
        for edge in x['edges']:
            if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                raise DisjointnessError('edge shape')
            u, v, w = map(_int, edge)
            if not (0 <= u < n and 0 <= v < n) or u == v or w <= 0 or (u, v) in seen:
                raise DisjointnessError('graph invariant')
            seen.add((u, v))
        x['edges'] = sorted(map(list, x['edges']))
    else:
        if _int(x['count']) < 1 or not isinstance(x['records'], list) or len(x['records']) != x['count']:
            raise DisjointnessError('record count')
        for record in x['records']:
            if not isinstance(record, (list, tuple)) or len(record) != 3 or type(record[0]) is not str:
                raise DisjointnessError('transaction shape')
            _int(record[1]); _int(record[2])
        x['records'] = sorted(map(list, x['records']), key=_key)
    return _key([family, x])


def semantic_signature(family, inputs):
    canonical_inputs(family, inputs)
    x = inputs
    if family == 'integer':
        return integer_ast(f"({x['a']})*({x['b']})+({x['c']})")
    if family == 'modular':
        return ('modular', x['a'], x['e'], x['m'])
    if family == 'division':
        return ('division', x['a'], x['b'])
    if family == 'dp':
        return ('grid-min-visited-sum', 'right-down', 'both-endpoints', x['matrix'])
    if family == 'graph':
        return ('directed-shortest-distance', x['n'], 0, x['n'] - 1, sorted(map(list, x['edges'])))
    return ('category-sum-quantity-times-unit-amount', sorted(map(list, x['records']), key=_key))


def _embedded(prompt, label):
    """Parse a balanced literal after a data label, never a surrounding program."""
    match = re.search(label + r'\s*[:=]\s*', prompt, re.I)
    if not match:
        return None
    tail = prompt[match.end():]
    if not tail.startswith(('[', '{', '(')):
        return None
    # Locate a complete container, respecting quoted strings and escapes. The
    # following prose is never included in the literal or executed as Python.
    stack, quote, escaped = [], None, False
    for pos, char in enumerate(tail[:1000000]):
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char in '[{(':
            stack.append(char)
        elif char in ']})':
            if not stack or stack.pop() != {']': '[', '}': '{', ')': '('}[char]:
                return None
            if not stack:
                try:
                    return ast.literal_eval(tail[:pos + 1])
                except (ValueError, SyntaxError, RecursionError):
                    return None
    return None


def _prompt_signature(prompt):
    p = normalize_prompt(prompt)
    assignments = re.findall(r'\b([abcemn])\s*=\s*([+-]?\d+)\b', p)
    x = {k: int(v) for k, v in assignments}
    if len(assignments) != len(x):
        return None  # contradictory or repeated declarations need original review
    if {'a', 'b', 'c'} <= x.keys() and re.search(r'a\s*\*\s*b\s*\+\s*c', p):
        return semantic_signature('integer', {k: x[k] for k in ('a', 'b', 'c')})
    if {'a', 'e', 'm'} <= x.keys() and ('modulo' in p or ' mod ' in p) and ('exponent' in p or '^' in p or '**' in p):
        return semantic_signature('modular', {k: x[k] for k in ('a', 'e', 'm')})
    mod = re.search(r'([+-]?\d+)\s*(?:\^|\*\*)\s*(\d+)\s*(?:modulo|mod)\s*(\d+)', p)
    if mod:
        return ('modular', *map(int, mod.groups()))
    if {'a', 'b'} <= x.keys() and ('remainder' in p or '0<=r<b' in p.replace(' ', '')):
        return ('division', x['a'], x['b'])
    div = re.search(r'([+-]?\d+)\s*(?:divided by|divmod|//)\s*([+-]?\d+)', p)
    if div:
        return ('division', *map(int, div.groups()))
    matrix = _embedded(prompt, r'(?:grid|matrix)')
    if matrix is not None and all(word in p for word in ('right', 'down', 'minimum', 'sum', 'endpoint', 'top-left', 'bottom-right')):
        data = {'rows': len(matrix), 'cols': len(matrix[0]) if matrix else 0, 'matrix': matrix}
        return semantic_signature('dp', data)
    edges = _embedded(prompt, r'edges')
    vertices = re.search(r'vertices 0 through (\d+)', p)
    route = re.search(r'from vertex (\d+) to vertex (\d+)', p)
    if edges is not None and vertices and route and 'directed' in p and 'undirected' not in p and 'minimum' in p and 'cost' in p:
        n = int(vertices[1]) + 1
        canonical_inputs('graph', {'n': n, 'edges': edges})
        return ('directed-shortest-distance', n, int(route[1]), int(route[2]), sorted(map(list, edges)))
    records = _embedded(prompt, r'(?:transactions|records)')
    if records is not None and 'category' in p and 'sum' in p and re.search(r'quantity\s*\*\s*unit_amount', p):
        return semantic_signature('aggregation', {'count': len(records), 'records': records})
    expr = re.fullmatch(r'(?:what is|compute|evaluate)\s+([\d\s()+*\-]+)(?:\s+exactly)?[?.]?(?: give the exact integer\.)?', p)
    if expr:
        return integer_ast(expr[1].strip())
    return None


def prior_signature(record):
    spec = record.get('spec')
    if isinstance(spec, dict) and 'op' in spec and 'args' in spec:
        op, raw_args = spec['op'], spec['args']
        if not isinstance(raw_args, (list, tuple)):
            raise DisjointnessError('historical op args must be a sequence')
        if op in ('+', '*', '-') and len(raw_args) == 2:
            left, right = [_number_node(_old_number(v)) for v in raw_args]
            return ('integer-expression', _binary(op, left, right))
        if op in ('exact-div', 'divmod', 'division') and len(raw_args) == 2:
            nodes = [_number_node(_old_number(v)) for v in raw_args]
            # Preserve the existing integer signature; nonintegral operands are
            # exact rational nodes, distinct from any new integer-only instance.
            return ('division', *(node[1] if node[0] == 'literal' else node for node in nodes))
        args = [_old_int(v) for v in raw_args]
        if op == 'powmod' and len(args) == 3:
            return ('modular', *args)
        if op == 'power-difference' and len(args) == 2:
            return integer_ast(f'2**{args[0]}-3**{args[1]}')
        if op in ('digit-sum', 'factorial-ratio', 'stock'):
            return ('w9-' + op, args)
        raise DisjointnessError('unrecognized historical op/args')
    if record.get('family') in FAMILIES and 'inputs' in record:
        return semantic_signature(record['family'], record['inputs'])
    if _generic_program(record['prompt']):
        return None  # examples inside a general program contract are illustrative
    return _prompt_signature(record['prompt'])


def _generic_program(prompt):
    p = normalize_prompt(prompt)
    if p.startswith('solve the following programming problem. write a standalone cpython 3 program reading standard input'):
        return True  # W9's explicit arbitrary-standard-input contracts
    contract = re.split(r'\bexamples?\s*[:\n]', p, maxsplit=1)[0]
    if re.search(r'\b(?:for|where|given)\s+[a-z]+\s*=\s*[+-]?\d+', contract) or re.search(r'(?:grid|matrix|transactions|edges)\s*:\s*\[', contract):
        return False  # a fixed input is not merely a generic programming concept
    return (p.startswith('write a python function')
            or p.startswith('write the python function described below.')
            or ((p.startswith('def ') or p.startswith('from typing import ') or p.startswith('import '))
                and 'def ' in p and 'return the complete function in a single python code fence.' in p))


def source_review_inventory(records):
    """JSON-serializable original evidence for parent CPU review/export.

    Unknown concrete problems are conservatively unresolved regardless of size.
    This deliberately overblocks rather than claiming a heuristic manual review.
    Recognized generic contracts are listed too, with the classification reason;
    their illustrative examples are never interpreted as requested instances.
    """
    inventory = []
    for record in records:
        sig = prior_signature(record)
        generic = sig is None and _generic_program(record['prompt'])
        inventory.append({
            'id': record['id'], 'source': record.get('source'),
            'source_path': record.get('source_path'), 'source_sha256': record.get('source_sha256'),
            'prompt': record['prompt'], 'prompt_sha256': prompt_sha256(record['prompt']),
            'original_prompt': record.get('original_prompt', record['prompt']),
            'classification': 'operation-signature' if sig is not None else ('generic-program-contract' if generic else 'unresolved-original'),
            'signature': sig,
            'reason': ('unambiguous operation and operands' if sig is not None else
                       'explicit arbitrary-input programming contract; examples are illustrative' if generic else
                       'conservative BLOCK: read original problem and supply prompt-bound resolution'),
        })
    return inventory


class CollisionIndex:
    """check is read-only; call add after accepting a candidate to index new rows.

    resolutions is keyed by ID. Each entry requires prompt_sha256, nonempty
    original_evidence (path/hash/location reviewed), reason, and either signature
    or decision='nonoverlap'. This is evidence supplied by an actual reviewer,
    never an automatic claim that every original problem was manually reviewed.
    """
    def __init__(self, records, resolutions=None):
        self.ids, self.inputs, self.prompts, self.signatures = {}, {}, {}, {}
        self.potential = []
        resolutions = resolutions or {}
        for record in records:
            self._insert(record, False, resolutions.get(record.get('id')))
        if self.potential:
            raise UnresolvedOverlap(self.potential)

    def _insert(self, record, candidate, resolution=None):
        if not isinstance(record.get('id'), str) or not record['id']:
            raise DisjointnessError('missing task id')
        prompt = normalize_prompt(record.get('prompt'))
        sig = prior_signature(record)
        if sig is None and not _generic_program(record['prompt']):
            evidence = (resolution or {}).get('original_evidence', {})
            valid_evidence = (isinstance(evidence, dict) and evidence.get('path')
                              and re.fullmatch(r'[a-f0-9]{64}', evidence.get('sha256', ''))
                              and evidence.get('location'))
            if resolution and resolution.get('prompt_sha256') == prompt_sha256(record['prompt']) and valid_evidence and resolution.get('reason'):
                sig = resolution.get('signature')
                if sig is None and resolution.get('decision') != 'nonoverlap':
                    raise DisjointnessError('resolution lacks operation signature or nonoverlap decision')
            else:
                self.potential.extend(source_review_inventory([record]))
        detail = {'id': record['id'], 'source': record.get('source', 'new' if candidate else 'prior')}
        self.ids.setdefault(record['id'], []).append(detail)
        self.prompts.setdefault(prompt, []).append(detail)
        if 'original_prompt' in record:
            original = normalize_prompt(record['original_prompt'])
            if original != prompt:
                self.prompts.setdefault(original, []).append(detail)
        if 'inputs' in record and record.get('family') in FAMILIES:
            key = canonical_inputs(record['family'], record['inputs'])
            self.inputs.setdefault(key, []).append(detail)
        if sig is not None:
            self.signatures.setdefault(_key(sig), []).append(detail)

    def check(self, candidate):
        """Return all documented collision reasons, or None; errors never reject/retry."""
        if self.potential:
            raise UnresolvedOverlap(self.potential)
        for field in ('id', 'family', 'slot', 'attempt', 'inputs', 'prompt', 'suite'):
            if field not in candidate:
                raise DisjointnessError('missing public candidate field: ' + field)
        if candidate['suite'] != 'compute_dl' or type(candidate['id']) is not str or not candidate['id']:
            raise DisjointnessError('invalid candidate suite/id')
        if type(candidate['slot']) is not int or not 0 <= candidate['slot'] < 20 or type(candidate['attempt']) is not int or candidate['attempt'] < 0:
            raise DisjointnessError('invalid slot/attempt')
        family, inputs = candidate['family'], candidate['inputs']
        keys = (('id', self.ids, candidate['id']),
                ('typed-inputs', self.inputs, canonical_inputs(family, inputs)),
                ('normalized-full-prompt', self.prompts, normalize_prompt(candidate['prompt'])),
                ('operation-operands', self.signatures, _key(semantic_signature(family, inputs))))
        hits = [{'reason': reason, 'matches': mapping[key]} for reason, mapping, key in keys if key in mapping]
        return {'candidate_id': candidate['id'], 'attempt': candidate['attempt'], 'collisions': hits} if hits else None

    def add(self, candidate):
        collision = self.check(candidate)
        if collision:
            raise DisjointnessError('cannot add colliding candidate: ' + _key(collision))
        self._insert(candidate, True)
