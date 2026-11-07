#!/usr/bin/env python3
"""Read-only seal gate; dispatch is unreachable until an explicit reviewed binding.

No queue or lock operations live here. The future controller owns head-time
admission, rank lifecycle, restore and the existing read-only root-rank1-host lease.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from stage_turns import A1_SHA, FROZEN_SHA, ORDERS, validate_waits

HERE = Path(__file__).resolve().parent
# A1 is authorized, but its CPU gates and future controller remain unqualified.
# Binding requires a reviewed source amendment; runtime env/CLI cannot replace it.
AMENDED_SPEC_SHA256 = A1_SHA
BINDING_SHA256 = None
GATES = (
    'mode_and_refusal_semantics', 'forced_call_dispatch_negative',
    'cpu_full_initial_prompt_arrays', 'identical_dl_wire_bytes_and_served_initial_counts',
    'a1_conditional_input_accounting_and_claim_withholding', 'native_returned_id_accounting',
    'explicit_552_request_layout_and_wire_order', 'strict_and_sensitivity_graders',
    'blind_120_math_mapping', 'isolated_jail_controls', 'three_contrast_analysis',
    'diagnostic_reference_and_safety_controls', 'three_arm_controller_fake_recovery',
    'rank_adapter_native_env_snapshots', 'both_rank_preflights',
    'quiet_memory_queue_admission', 'read_only_root_rank1_host_lease',
    'restore_and_exact_owned_drain', 'reserve_unmeasured_only_no_replacement',
)


def require(condition):
    if not condition:
        raise ValueError('seal or prerequisite absent/invalid')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute())
    require('..' not in Path(relative).parts)
    path = root / relative
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)))
    require(path.resolve().is_relative_to(root))
    return path


def validate_boots(index):
    validate_waits(index['waits'])
    expected = [(turn, arm, f'{turn}-{arm}') for turn, arms in ORDERS.items() for arm in arms]
    require([(b['turn'], b['arm'], b['boot']) for b in index['boots']] == expected)
    require(index['orders'] == {t: list(arms) for t, arms in ORDERS.items()})
    require(index['reserve'] == {'turn': 'reserve', 'new_boots': False, 'unmeasured_fixed_units_only': True})


def validate_binding(turn):
    require(turn in (*ORDERS, 'reserve'))
    # First check, before even reading any supplied binding/controller artifacts.
    require(isinstance(AMENDED_SPEC_SHA256, str) and re.fullmatch('[0-9a-f]{64}', AMENDED_SPEC_SHA256))
    require(AMENDED_SPEC_SHA256 != FROZEN_SHA)
    require(isinstance(BINDING_SHA256, str) and re.fullmatch('[0-9a-f]{64}', BINDING_SHA256))
    binding_path = artifact(HERE, 'staged_binding.json')
    require(sha(binding_path) == BINDING_SHA256)
    binding = json.loads(binding_path.read_text())
    require(binding['schema'] == 'decl-control-launch-binding-v1')
    require(binding['status'] == 'CPU_QUALIFIED')
    require(binding['original_spec_sha256'] == FROZEN_SHA)
    require(binding['amended_spec_sha256'] == AMENDED_SPEC_SHA256)
    require(binding['controller_protocol'] == 'sealed-three-arm-campaign-v1')
    require(binding['max_turns'] == 4 and binding['scheduled_turns'] == list(ORDERS))
    validate_waits(binding['waits'])
    inventory = binding['files']
    require(isinstance(inventory, dict) and inventory)
    for relative, expected in inventory.items():
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected))
        require(sha(artifact(HERE, relative)) == expected)
    # Everything executed or consumed must be in the binding's sealed inventory.
    for key in ('spec', 'original_spec', 'controller', 'campaign', 'stage_index'):
        require(binding[key] in inventory)
    require(inventory[binding['spec']] == AMENDED_SPEC_SHA256)
    require(inventory[binding['original_spec']] == FROZEN_SHA)
    controller = artifact(HERE, binding['controller'])
    require(controller.name == 'staged_three_arm_controller.py')
    require(binding['controller_sha256'] == inventory[binding['controller']])
    require(set(binding['gates']) == set(GATES))
    for gate in GATES:
        receipt_ref = binding['gates'][gate]
        require(receipt_ref in inventory)
        receipt = json.loads(artifact(HERE, receipt_ref).read_text())
        require(receipt['gate'] == gate and receipt['status'] == 'PASS')
        require(receipt['amended_spec_sha256'] == AMENDED_SPEC_SHA256)
        require(receipt['controller_sha256'] == binding['controller_sha256'])
        require(receipt['campaign_sha256'] == inventory[binding['campaign']])
    index_path = artifact(HERE, binding['stage_index'])
    index = json.loads(index_path.read_text())
    validate_boots(index)
    for relative, expected in index['files'].items():
        path = artifact(index_path.parent, relative)
        require(sha(path) == expected)
        require(inventory[str(path.relative_to(HERE))] == expected)
    # This receipt-based gate cannot establish CPU truth: parent must bind only
    # after actual Spark verification, review and the full dependency freeze.
    return binding, controller, artifact(HERE, binding['campaign'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('turn', choices=(*ORDERS, 'reserve'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        binding, controller, campaign = validate_binding(args.turn)
    except (ValueError, TypeError, KeyError, OSError):
        print('BLOCKED: amended spec, sealed three-arm controller and full CPU prerequisites are not bound.', file=sys.stderr)
        return 78
    if args.check_only:
        print('SEALED: dispatch not requested')
        return 0
    # No shell expansion, inherited experiment flags, queue append or lock access.
    # Admission and maximum-four-turn/reserve accounting remain controller duties.
    env = {k: v for k, v in os.environ.items() if k in
           ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAKEFLAGS='-j1')
    command = ['systemd-run', '--user', '--service-type=exec', '--wait', '--pipe',
               '-p', 'MemoryMax=2G', '-p', 'MemorySwapMax=0',
               '-p', 'TimeoutStopSec=7200', 'nice', '-n', '19', 'ionice', '-c3',
               'timeout', str(binding['waits']['outer_timeout_s']),
               sys.executable, '-B', str(controller), '--campaign', str(campaign), '--turn', args.turn]
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    raise SystemExit(main())
