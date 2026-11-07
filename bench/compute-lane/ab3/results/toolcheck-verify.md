# AB3 tool-result check

checker f70eb85c3d8da824, re-run jail: {'passed': True, 'detail': "ok ['blocked', 'read-only', '65534', '/', '1', 'True', '4300', '1024']  | loop timeout", 'image': 'tensorfold-dsv41-imr:r-3bf584af1c54', 'runner_sha256': '80177f5c8a086de532b1372e15cf9b49eeacaa490da2ed4312fc68d2b885a7d0', 'run_jobs_sha256': 'c53b7dcba74a35c99282735f5e65622a7316194195eb72899c4f6cd0046f90f8'}

| arm/run | receipts | calls | ok | wrong | error-ok | inconclusive | undelivered | executors | fallbacks | overall |
|---|---|---|---|---|---|---|---|---|---|---|
| ab3-b1-2-lane/d-s101 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b1-2-lane/d-s102 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b1-2-lane/d-s103 | 20 | 17 | 17 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 17 | 0 | PASS |
| ab3-b2-1-lane/d-s101 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b2-1-lane/d-s102 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b2-1-lane/d-s103 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b3-4-lane/d-s101 | 20 | 14 | 14 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 14 | 0 | PASS |
| ab3-b3-4-lane/d-s102 | 20 | 17 | 17 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 17 | 0 | PASS |
| ab3-b3-4-lane/d-s103 | 20 | 19 | 19 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 19 | 0 | PASS |
| ab3-b1-3-full/rn-s101 | 20 | 18 | 18 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 8 | 8 | PASS |
| ab3-b1-3-full/rn-s102 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 6 | 6 | PASS |
| ab3-b1-3-full/rn-s103 | 20 | 17 | 17 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 7 | 7 | PASS |
| ab3-b2-2-full/rn-s101 | 20 | 17 | 17 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 7 | 7 | PASS |
| ab3-b2-2-full/rn-s102 | 20 | 19 | 19 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 9 | 9 | PASS |
| ab3-b2-2-full/rn-s103 | 20 | 19 | 19 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 9 | 9 | PASS |
| ab3-b3-1-full/rn-s101 | 20 | 17 | 17 | 0 | 0 | 0 | 0 | in-model runner: 9, code tool: 8 | 8 | PASS |
| ab3-b3-1-full/rn-s102 | 20 | 18 | 18 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 8 | 8 | PASS |
| ab3-b3-1-full/rn-s103 | 20 | 18 | 18 | 0 | 0 | 0 | 0 | in-model runner: 10, code tool: 8 | 8 | PASS |
| ab3-b1-4-only/ro-s101 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b1-4-only/ro-s102 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b1-4-only/ro-s103 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b2-3-only/ro-s101 | 20 | 15 | 15 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 15 | 0 | PASS |
| ab3-b2-3-only/ro-s102 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b2-3-only/ro-s103 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b3-2-only/ro-s101 | 20 | 14 | 14 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 14 | 0 | PASS |
| ab3-b3-2-only/ro-s102 | 20 | 16 | 16 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 16 | 0 | PASS |
| ab3-b3-2-only/ro-s103 | 20 | 18 | 18 | 0 | 0 | 0 | 0 | sandbox (lane, no S3): 18 | 0 | PASS |

