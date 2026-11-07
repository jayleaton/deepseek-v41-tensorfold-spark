# Measured configurations

| file | what |
| --- | --- |
| `arm-L-effective.env` | the effective container environment of arm L (lane on, runner off), rank 0, read from the boot's snapshot; P and D change only `TF_DSV41_COMPUTE_LANE` (to 0) and the request's tools |
| `requests-lane-studies.json` | the request body of decl-control, compute-DL and format-retest (seed and tools vary by reply and arm; the prompt goes in `messages`) |
| `tool-full-ab3.env` | AB3's tool-full arm: 18 engine lines over arm L's environment |
| `requests-ab3-tool-full.json` | AB3's tool-full request body without `messages`; the measurement client added `"deadline_s": 300` to each request and ran four streams at a time |

The recipe profile (`config/prod-zig-flash17.env.example`) defaults to tool-full plus
`TF_DSV41_COMPUTE_LANE_DEFAULT=1`; arm L is its commented opt-in block. Both blocks share a base profile
whose three differences from the measured arm-L environment are listed in
[docs/compute-lane.md](../../../docs/compute-lane.md#the-flash17-profile-against-the-measured-arms).
