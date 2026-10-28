"""The sole shell-quoting adapter: validate raw values before encoding shell syntax."""
import os
import shlex

from redact import check_argv, known_secrets
from safe_output import _cli_secrets, _known


def join_checked(argv, *, env=None):
    """Return POSIX shell argv only after the shared credential rule accepts every raw value."""
    values = [os.fsdecode(value) for value in argv]
    # Configured/unexported credentials and explicit runner environments both
    # count. Credential-shaped flags/headers are checked by check_argv itself.
    check_argv(values, _known() + known_secrets(env) + _cli_secrets([]))
    return shlex.join(values)


def quote_checked(value, *, env=None):
    return join_checked([value], env=env)
