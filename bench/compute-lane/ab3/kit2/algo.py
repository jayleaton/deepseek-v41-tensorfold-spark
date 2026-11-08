"""The harder algorithmic set: 40 stdin/stdout problems in the Codeforces <=1600 style, written for this harness (so the
model has not seen these exact statements). Each has a reference solution and a seeded input generator; tasks.py runs
the reference to fix the hidden tests. Test 0 is the small example shown in the prompt; the rest stay hidden."""
import bisect
import collections
import heapq
import math

P = 10**9 + 7


def ints(s):
    return list(map(int, s.split()))


def arr(rng, n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def line(xs):
    return ' '.join(map(str, xs))


PROBLEMS = []


def problem(pid, title, statement, sizes=(5, 30, 300, 3000, 20000)):
    def wrap(fn):
        PROBLEMS.append({'id': f'algo-{pid:02d}', 'title': title, 'statement': statement, 'solve': fn[0], 'gen': fn[1],
                         'sizes': sizes})
        return fn
    return wrap


# 01
def _s(inp):
    t = ints(inp)
    n, k, a = t[0], t[1], t[2:]
    best = cur = sum(a[:k])
    for i in range(k, n):
        cur += a[i] - a[i - k]
        best = max(best, cur)
    return str(best)


def _g(rng, n):
    k = rng.randint(1, n)
    return f'{n} {k}\n{line(arr(rng, n, -10**6, 10**6))}\n'


problem(1, 'Best window', 'Given n, k (1<=k<=n) and n integers, print the largest sum of k consecutive elements.\n'
        'Input: "n k" then the n integers. Output: one integer.')((_s, _g))


# 02
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    tails = []
    for x in a:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return str(len(tails))


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 1, max(2, n)))}\n'


problem(2, 'Strictly rising', 'Print the length of the longest strictly increasing subsequence of the n integers.\n'
        'Input: n then n integers. Output: one integer.')((_s, _g))


# 03
def _s(inp):
    t = ints(inp)
    n, m = t[0], t[1]
    coins = t[2:2 + n]
    ways = [1] + [0] * m
    for c in coins:
        for v in range(c, m + 1):
            ways[v] = (ways[v] + ways[v - c]) % P
    return str(ways[m])


def _g(rng, n):
    coins = sorted(set(arr(rng, min(n, 40), 1, 60)))
    return f'{len(coins)} {min(10 * n, 5000)}\n{line(coins)}\n'


problem(3, 'Coin ways', 'Distinct coin values c_1..c_n, unlimited supply. Print the number of multisets of coins summing to '
        'm, modulo 1000000007.\nInput: "n m" then the n values. Output: one integer.')((_s, _g))


# 04
def _s(inp):
    lines = inp.split('\n')
    r, c = map(int, lines[0].split())
    g = lines[1:1 + r]
    sr = sc = er = ec = 0
    for i in range(r):
        for j in range(c):
            if g[i][j] == 'S':
                sr, sc = i, j
            elif g[i][j] == 'E':
                er, ec = i, j
    d = [[-1] * c for _ in range(r)]
    d[sr][sc] = 0
    q = collections.deque([(sr, sc)])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < r and 0 <= b < c and g[a][b] != '#' and d[a][b] < 0:
                d[a][b] = d[i][j] + 1
                q.append((a, b))
    return str(d[er][ec])


def _g(rng, n):
    side = max(2, int(n ** 0.5))
    g = [['#' if rng.random() < 0.28 else '.' for _ in range(side)] for _ in range(side)]
    cells = [(i, j) for i in range(side) for j in range(side)]
    (a, b), (c, d) = rng.sample(cells, 2)
    g[a][b], g[c][d] = 'S', 'E'
    return f'{side} {side}\n' + '\n'.join(''.join(row) for row in g) + '\n'


problem(4, 'Maze steps', 'A grid of r rows and c columns has walls #, free cells ., one start S and one exit E. Moving '
        'up/down/left/right through non-wall cells, print the fewest moves from S to E, or -1 if E is unreachable.\n'
        'Input: "r c" then r lines of c characters. Output: one integer.')((_s, _g))


# 05
def _s(inp):
    t = ints(inp)
    n, m = t[0], t[1]
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    comps = n
    for i in range(m):
        a, b = find(t[2 + 2 * i]), find(t[3 + 2 * i])
        if a != b:
            parent[a] = b
            comps -= 1
    return str(comps)


def _g(rng, n):
    m = rng.randint(0, n)
    return f'{n} {m}\n' + ''.join(f'{rng.randint(1, n)} {rng.randint(1, n)}\n' for _ in range(m))


problem(5, 'Islands of friends', 'n people (1..n) and m friendships "a b" (undirected, may repeat or be a self-loop). Print '
        'the number of connected groups.\nInput: "n m" then m lines "a b". Output: one integer.')((_s, _g))


# 06
def _s(inp):
    t = ints(inp)
    n, m, s = t[0], t[1], t[2]
    adj = [[] for _ in range(n + 1)]
    for i in range(m):
        a, b, w = t[3 + 3 * i:6 + 3 * i]
        adj[a].append((b, w))
        adj[b].append((a, w))
    dist = [-1] * (n + 1)
    pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] >= 0:
            continue
        dist[u] = d
        for v, w in adj[u]:
            if dist[v] < 0:
                heapq.heappush(pq, (d + w, v))
    return line(dist[1:])


def _g(rng, n):
    n = max(2, n // 4)
    m = min(3 * n, 30000)
    return f'{n} {m} {rng.randint(1, n)}\n' + ''.join(f'{rng.randint(1, n)} {rng.randint(1, n)} {rng.randint(1, 1000)}\n' for _ in range(m))


problem(6, 'Shortest roads', 'An undirected weighted graph with n nodes, m edges "a b w" (w>=1) and a source s. Print n '
        'integers: the shortest distance from s to each node 1..n, or -1 if unreachable.\nInput: "n m s" then m lines. '
        'Output: n integers on one line.')((_s, _g))


# 07
def _s(inp):
    s = inp.split()[0]
    best = 0
    for c in range(len(s)):
        for lo, hi in ((c, c), (c, c + 1)):
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo -= 1
                hi += 1
            best = max(best, hi - lo - 1)
    return str(best)


def _g(rng, n):
    n = min(n, 2500)
    return ''.join(rng.choice('ab' if rng.random() < 0.5 else 'abc') for _ in range(n)) + '\n'


problem(7, 'Mirror piece', 'Print the length of the longest palindromic substring of the given lowercase string '
        '(length <= 2500).\nInput: the string. Output: one integer.', sizes=(5, 30, 300, 1500, 2500))((_s, _g))


# 08
def _s(inp):
    p, t = inp.split('\n')[:2]
    f = [0] * len(p)
    k = 0
    for i in range(1, len(p)):
        while k and p[i] != p[k]:
            k = f[k - 1]
        if p[i] == p[k]:
            k += 1
        f[i] = k
    cnt = k = 0
    for ch in t:
        while k and ch != p[k]:
            k = f[k - 1]
        if ch == p[k]:
            k += 1
        if k == len(p):
            cnt += 1
            k = f[k - 1]
    return str(cnt)


def _g(rng, n):
    pat = ''.join(rng.choice('ab') for _ in range(rng.randint(1, 4)))
    return pat + '\n' + ''.join(rng.choice('ab') for _ in range(n)) + '\n'


problem(8, 'Overlapping finds', 'Count the occurrences of pattern p in text t, overlaps included.\nInput: p on the first '
        'line, t on the second (lowercase). Output: one integer.')((_s, _g))


# 09
def _s(inp):
    n = int(inp)
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'[:min(2, n + 1)]
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
    return str(sum(sieve))


def _g(rng, n):
    return f'{rng.randint(n, 300 * n)}\n'


problem(9, 'Prime count', 'Print how many primes are <= n (1 <= n <= 6000000).\nInput: n. Output: one integer.')((_s, _g))


# 10
def _s(inp):
    t = ints(inp)
    q, out = t[0], []
    for i in range(q):
        a, b, m = t[1 + 3 * i:4 + 3 * i]
        out.append(pow(a, b, m))
    return '\n'.join(map(str, out))


def _g(rng, n):
    q = min(n, 2000)
    return f'{q}\n' + ''.join(f'{rng.randint(0, 10**18)} {rng.randint(0, 10**18)} {rng.randint(1, 10**9)}\n' for _ in range(q))


problem(10, 'Power remainders', 'q queries "a b m": print a^b mod m for each (0<=a,b<=10^18, 1<=m<=10^9; 0^0 = 1).\n'
        'Input: q then q lines. Output: q lines.')((_s, _g))


# 11
def _s(inp):
    t = ints(inp)
    n, iv = t[0], sorted(zip(t[1::2], t[2::2]))
    total, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        total += ce - cs
    return str(total)


def _g(rng, n):
    out = []
    for _ in range(n):
        s = rng.randint(0, 10 * n)
        out.append(f'{s} {s + rng.randint(1, 50)}')
    return f'{n}\n' + '\n'.join(out) + '\n'


problem(11, 'Painted length', 'n half-open segments [s, e) on a line. Print the total length covered by at least one.\n'
        'Input: n then n lines "s e" (s<e). Output: one integer.')((_s, _g))


# 12
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    res, st = [-1] * n, []
    for i, x in enumerate(a):
        while st and a[st[-1]] < x:
            res[st.pop()] = x
        st.append(i)
    return line(res)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 1, 100))}\n'


problem(12, 'Next taller', 'For each of n integers print the first strictly greater value to its right, or -1.\n'
        'Input: n then n integers. Output: n integers on one line.')((_s, _g))


# 13
def _s(inp):
    t = ints(inp)
    n, k, a = t[0], t[1], t[2:]
    seen = collections.Counter({0: 1})
    pre = res = 0
    for x in a:
        pre += x
        res += seen[pre - k]
        seen[pre] += 1
    return str(res)


def _g(rng, n):
    return f'{n} {rng.randint(-5, 5)}\n{line(arr(rng, n, -3, 3))}\n'


problem(13, 'Target sums', 'Count the non-empty contiguous subarrays of the n integers whose sum equals k.\nInput: "n k" then '
        'n integers. Output: one integer.')((_s, _g))


# 14
def _s(inp):
    t = ints(inp)
    n, c = t[0], t[1]
    items = list(zip(t[2::2], t[3::2]))
    best = [0] * (c + 1)
    for w, v in items:
        for cap in range(c, w - 1, -1):
            best[cap] = max(best[cap], best[cap - w] + v)
    return str(best[c])


def _g(rng, n):
    k = min(n, 100)
    return f'{k} {min(20 * k, 2000)}\n' + ''.join(f'{rng.randint(1, 100)} {rng.randint(1, 1000)}\n' for _ in range(k))


problem(14, 'Backpack', 'n items "w v" and capacity C. Each item used at most once. Print the largest total value with total '
        'weight <= C.\nInput: "n C" then n lines. Output: one integer.')((_s, _g))


# 15
def _s(inp):
    a, b = inp.split('\n')[:2]
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return str(prev[-1])


def _g(rng, n):
    n = min(n, 1500)
    return ''.join(rng.choice('abc') for _ in range(n)) + '\n' + ''.join(rng.choice('abc') for _ in range(rng.randint(1, n))) + '\n'


problem(15, 'Edit steps', 'Print the edit distance (insert, delete, replace one character) between two lowercase strings '
        '(each <= 1500).\nInput: two lines. Output: one integer.', sizes=(5, 30, 300, 800, 1500))((_s, _g))


# 16
def _s(inp):
    t = ints(inp)
    n, m = t[0], t[1]
    adj = [[] for _ in range(n + 1)]
    indeg = [0] * (n + 1)
    for i in range(m):
        a, b = t[2 + 2 * i], t[3 + 2 * i]
        adj[a].append(b)
        indeg[b] += 1
    pq = [i for i in range(1, n + 1) if indeg[i] == 0]
    heapq.heapify(pq)
    order = []
    while pq:
        u = heapq.heappop(pq)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(pq, v)
    return line(order) if len(order) == n else '-1'


def _g(rng, n):
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    m = min(2 * n, 40000)
    edges = []
    for _ in range(m):
        i, j = sorted(rng.sample(range(n), 2)) if n > 1 else (0, 0)
        if n > 1:
            edges.append((perm[i], perm[j]))
    if rng.random() < 0.25 and n > 2:
        edges.append((perm[-1], perm[0]))
    return f'{n} {len(edges)}\n' + ''.join(f'{a} {b}\n' for a, b in edges)


problem(16, 'Ordered chores', 'n tasks and m rules "a b" (a before b). Print the lexicographically smallest order of all '
        'tasks that obeys every rule, or -1 if none exists.\nInput: "n m" then m lines. Output: one line.')((_s, _g))


# 17
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    best, cur = a[0], 0
    for x in a:
        cur = max(x, cur + x)
        best = max(best, cur)
    return str(best)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, -1000, 1000))}\n'


problem(17, 'Best stretch', 'Print the largest sum of a non-empty contiguous subarray.\nInput: n then n integers. Output: '
        'one integer.')((_s, _g))


# 18
def _s(inp):
    t = ints(inp)
    n, q = t[0], t[1]
    a = t[2:2 + n]
    pre = [0]
    for x in a:
        pre.append(pre[-1] + x)
    out = []
    for i in range(q):
        l, r = t[2 + n + 2 * i], t[3 + n + 2 * i]
        out.append(pre[r] - pre[l - 1])
    return '\n'.join(map(str, out))


def _g(rng, n):
    q = min(n, 5000)
    qs = []
    for _ in range(q):
        l = rng.randint(1, n)
        qs.append(f'{l} {rng.randint(l, n)}')
    return f'{n} {q}\n{line(arr(rng, n, -10**9, 10**9))}\n' + '\n'.join(qs) + '\n'


problem(18, 'Range totals', 'n integers and q queries "l r" (1-based, inclusive). Print each range sum.\nInput: "n q", the '
        'integers, then q lines. Output: q lines.')((_s, _g))


# 19
def _s(inp):
    t = ints(inp)
    n, k, a = t[0], t[1], sorted(t[2:])
    lo, hi = 0, a[-1] - a[0]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cnt, last = 1, a[0]
        for x in a[1:]:
            if x - last >= mid:
                cnt += 1
                last = x
        if cnt >= k:
            lo = mid
        else:
            hi = mid - 1
    return str(lo)


def _g(rng, n):
    n = max(2, n)
    return f'{n} {rng.randint(2, n)}\n{line(arr(rng, n, 0, 10**9))}\n'


problem(19, 'Spread the posts', 'n positions on a line (may repeat). Choose k of them (2<=k<=n) to maximise the smallest gap '
        'between chosen positions. Print that gap.\nInput: "n k" then n integers. Output: one integer.')((_s, _g))


# 20
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    a.sort()
    return str(sum(abs(x - a[n // 2]) for x in a))


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, -10**9, 10**9))}\n'


problem(20, 'Meeting point', 'Print the minimum over integer x of sum |a_i - x|.\nInput: n then n integers. Output: one '
        'integer.')((_s, _g))


# 21
def _s(inp):
    t = ints(inp)
    out = []
    fact = [1] * (200001)
    for i in range(1, 200001):
        fact[i] = fact[i - 1] * i % P
    for i in range(t[0]):
        n, r = t[1 + 2 * i], t[2 + 2 * i]
        out.append(0 if r < 0 or r > n else fact[n] * pow(fact[r] * fact[n - r] % P, P - 2, P) % P)
    return '\n'.join(map(str, out))


def _g(rng, n):
    q = min(n, 3000)
    qs = []
    for _ in range(q):
        a = rng.randint(0, 200000)
        qs.append(f'{a} {rng.randint(0, a + 2)}')
    return f'{q}\n' + '\n'.join(qs) + '\n'


problem(21, 'Choose mod', 'q queries "n r" (0<=n<=200000, 0<=r<=n+2): print C(n, r) mod 1000000007 (0 when r>n).\nInput: q '
        'then q lines. Output: q lines.')((_s, _g))


# 22
def _s(inp):
    s = inp.split()[0]
    pos, best, lo = {}, 0, 0
    for i, ch in enumerate(s):
        if pos.get(ch, -1) >= lo:
            lo = pos[ch] + 1
        pos[ch] = i
        best = max(best, i - lo + 1)
    return str(best)


def _g(rng, n):
    return ''.join(rng.choice('abcdefghij') for _ in range(n)) + '\n'


problem(22, 'No repeats', 'Print the length of the longest substring with all distinct characters.\nInput: a lowercase '
        'string. Output: one integer.')((_s, _g))


# 23
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    st, best = [], 0
    for i, h in enumerate(a + [0]):
        start = i
        while st and st[-1][1] >= h:
            j, hh = st.pop()
            best = max(best, hh * (i - j))
            start = j
        st.append((start, h))
    return str(best)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 0, 10**4))}\n'


problem(23, 'Skyline poster', 'n bars of width 1 and heights h_i side by side. Print the largest axis-aligned rectangle area '
        'inside the bars.\nInput: n then n heights. Output: one integer.')((_s, _g))


# 24
def _s(inp):
    t = ints(inp)
    n, k, a = t[0], t[1], t[2:]
    dq, out = collections.deque(), []
    for i, x in enumerate(a):
        while dq and a[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(a[dq[0]])
    return line(out)


def _g(rng, n):
    return f'{n} {rng.randint(1, n)}\n{line(arr(rng, n, -100, 100))}\n'


problem(24, 'Window peaks', 'Print the maximum of every window of k consecutive elements, left to right.\nInput: "n k" then '
        'n integers. Output: n-k+1 integers on one line.')((_s, _g))


# 25
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    res, s = 0, set(a)
    for x in s:
        if x - 1 not in s:
            y = x
            while y + 1 in s:
                y += 1
            res = max(res, y - x + 1)
    return str(res)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 0, 2 * n))}\n'


problem(25, 'Run of values', 'Print the length of the longest run of consecutive integer values present among the n numbers '
        '(order irrelevant, duplicates count once).\nInput: n then n integers. Output: one integer.')((_s, _g))


# 26
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    jumps, end, far = 0, 0, 0
    for i in range(n - 1):
        if i > far:
            return '-1'
        far = max(far, i + a[i])
        if i == end:
            if far <= i:
                return '-1'
            jumps += 1
            end = far
    return str(jumps)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 0 if rng.random() < 0.3 else 1, 5))}\n'


problem(26, 'Fewest hops', 'From index 1 of n cells, cell i lets you jump forward up to a_i cells. Print the fewest jumps to '
        'reach cell n, or -1 if impossible (0 when n=1).\nInput: n then n integers. Output: one integer.')((_s, _g))


# 27
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    inv = 0

    def sort(xs):
        nonlocal inv
        if len(xs) < 2:
            return xs
        m = len(xs) // 2
        l, r = sort(xs[:m]), sort(xs[m:])
        out, i, j = [], 0, 0
        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                out.append(l[i])
                i += 1
            else:
                out.append(r[j])
                inv += len(l) - i
                j += 1
        return out + l[i:] + r[j:]
    sort(a)
    return str(inv)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 1, n))}\n'


problem(27, 'Out of order pairs', 'Print the number of pairs i<j with a_i > a_j.\nInput: n then n integers. Output: one '
        'integer.')((_s, _g))


# 28
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    g = 0
    for x in a:
        g = math.gcd(g, x)
    lcm = 1
    for x in a:
        lcm = lcm * x // math.gcd(lcm, x)
    return f'{g} {lcm % P}'


def _g(rng, n):
    n = min(n, 2000)
    return f'{n}\n{line(arr(rng, n, 1, 10**4))}\n'


problem(28, 'Shared measure', 'Print the gcd of the n positive integers and their lcm modulo 1000000007 (the lcm reduced only '
        'after it is computed exactly).\nInput: n then n integers. Output: two integers.', sizes=(5, 30, 300, 1000, 2000))((_s, _g))


# 29
def _s(inp):
    lines = inp.split('\n')
    r, c = map(int, lines[0].split())
    g = [list(map(int, lines[1 + i].split())) for i in range(r)]
    dp = [[0] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            up = dp[i - 1][j] if i else -10**18
            left = dp[i][j - 1] if j else -10**18
            dp[i][j] = g[i][j] + (0 if i == 0 and j == 0 else max(up, left))
    return str(dp[-1][-1])


def _g(rng, n):
    side = max(1, int(n ** 0.5))
    return f'{side} {side}\n' + '\n'.join(line(arr(rng, side, -50, 100)) for _ in range(side)) + '\n'


problem(29, 'Harvest path', 'An r x c grid of integers. Walk from the top-left to the bottom-right moving only right or down. '
        'Print the largest possible sum of visited cells.\nInput: "r c" then r rows. Output: one integer.')((_s, _g))


# 30
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    cnt = collections.Counter(a)
    top = max(cnt.values())
    return str(min(x for x, c in cnt.items() if c == top))


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, -20, 20))}\n'


problem(30, 'Most common', 'Print the value occurring most often; on a tie, the smallest such value.\nInput: n then n '
        'integers. Output: one integer.')((_s, _g))


# 31
def _s(inp):
    t = ints(inp)
    n, k, a = t[0], t[1], sorted(t[2:])
    i, res = 0, 0
    for j in range(n):
        while a[j] - a[i] > k:
            i += 1
        res += j - i
    return str(res)


def _g(rng, n):
    return f'{n} {rng.randint(0, 50)}\n{line(arr(rng, n, 0, 10 * n))}\n'


problem(31, 'Close pairs', 'Print the number of pairs i<j with |a_i - a_j| <= k.\nInput: "n k" then n integers. Output: one '
        'integer.')((_s, _g))


# 32
def _s(inp):
    s = inp.split()[0]
    st, best = [-1], 0
    for i, ch in enumerate(s):
        if ch == '(':
            st.append(i)
        else:
            st.pop()
            if not st:
                st.append(i)
            else:
                best = max(best, i - st[-1])
    return str(best)


def _g(rng, n):
    return ''.join(rng.choice('()') for _ in range(n)) + '\n'


problem(32, 'Balanced run', 'Print the length of the longest contiguous balanced bracket substring of a string of ( and ).\n'
        'Input: the string. Output: one integer.')((_s, _g))


# 33
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    take = skip = 0
    for x in a:
        take, skip = skip + x, max(take, skip)
    return str(max(take, skip))


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 0, 1000))}\n'


problem(33, 'No neighbours', 'Pick a subset of the n non-negative integers with no two adjacent indices. Print the largest '
        'possible sum.\nInput: n then n integers. Output: one integer.')((_s, _g))


# 34
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    best = 0
    for bit in range(30, -1, -1):
        best <<= 1
        pre = {x >> bit for x in a}
        cand = best | 1
        if any((cand ^ p) in pre for p in pre):
            best = cand
    return str(best)


def _g(rng, n):
    n = max(2, n)
    return f'{n}\n{line(arr(rng, n, 0, 2**30 - 1))}\n'


problem(34, 'Biggest xor', 'Print the maximum of a_i XOR a_j over all pairs i<j (n>=2, 0<=a_i<2^30).\nInput: n then n '
        'integers. Output: one integer.')((_s, _g))


# 35
def _s(inp):
    t = ints(inp)
    n, m = t[0], t[1]
    edges = sorted((t[4 + 3 * i], t[2 + 3 * i], t[3 + 3 * i]) for i in range(m))
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    total, used = 0, 0
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            total += w
            used += 1
    return str(total) if used == n - 1 else '-1'


def _g(rng, n):
    n = max(2, n // 3)
    m = min(3 * n, 30000)
    return f'{n} {m}\n' + ''.join(f'{rng.randint(1, n)} {rng.randint(1, n)} {rng.randint(1, 10**6)}\n' for _ in range(m))


problem(35, 'Cheapest network', 'n towns and m possible cables "a b w". Print the least total cost connecting all towns, or '
        '-1 if impossible.\nInput: "n m" then m lines. Output: one integer.')((_s, _g))


# 36
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    a = sorted(a)
    res = 0
    for i in range(n):
        res += a[i] * (2 * i - n + 1)
    return str(res)


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, -10**6, 10**6))}\n'


problem(36, 'All pair gaps', 'Print the sum of |a_i - a_j| over all pairs i<j.\nInput: n then n integers. Output: one '
        'integer.')((_s, _g))


# 37
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    lo = hi = 1
    best = -10**30
    for x in a:
        cand = (x, lo * x, hi * x)
        lo, hi = min(cand), max(cand)
        best = max(best, hi)
    return str(best)


def _g(rng, n):
    n = min(n, 300)
    return f'{n}\n{line(arr(rng, n, -9, 9))}\n'


problem(37, 'Best product', 'Print the largest product of a non-empty contiguous subarray (exact, it may be large).\nInput: n '
        'then n integers. Output: one integer.', sizes=(5, 20, 60, 150, 300))((_s, _g))


# 38
def _s(inp):
    t = ints(inp)
    n, limit, a = t[0], t[1], sorted(t[2:])
    lo, hi, boats = 0, n - 1, 0
    while lo <= hi:
        if a[lo] + a[hi] <= limit:
            lo += 1
        hi -= 1
        boats += 1
    return str(boats)


def _g(rng, n):
    limit = rng.randint(100, 300)
    return f'{n} {limit}\n{line(arr(rng, n, 1, limit))}\n'


problem(38, 'Two to a boat', 'n people with weights w_i <= L ride boats holding at most two people and total weight <= L. '
        'Print the fewest boats.\nInput: "n L" then n weights. Output: one integer.')((_s, _g))


# 39
def _s(inp):
    t = ints(inp)
    n, a = t[0], t[1:]
    cnt = collections.Counter(x % 3 for x in a)
    return str(cnt[0] * (cnt[0] - 1) // 2 + cnt[1] * cnt[2])


def _g(rng, n):
    return f'{n}\n{line(arr(rng, n, 1, 10**9))}\n'


problem(39, 'Thirds', 'Print the number of pairs i<j whose sum a_i + a_j is divisible by 3.\nInput: n then n positive '
        'integers. Output: one integer.')((_s, _g))


# 40
def _s(inp):
    t = ints(inp)
    n = t[0]
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if i >= 2:
            dp[i] += dp[i - 2]
        if i >= 3:
            dp[i] += dp[i - 3]
        dp[i] %= P
    return str(dp[n])


def _g(rng, n):
    return f'{rng.randint(1, 50 * n)}\n'


problem(40, 'Stair hops', 'Climb n stairs taking 1, 2 or 3 steps at a time. Print the number of distinct step sequences '
        'modulo 1000000007.\nInput: n (1<=n<=10^6). Output: one integer.')((_s, _g))
