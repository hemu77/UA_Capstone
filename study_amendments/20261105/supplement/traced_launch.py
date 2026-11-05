"""Run the unchanged reviewed entrypoint once, retaining only redacted diagnostics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parent
EXPECTED = 'd7dac09925abdf85df49cf1f400b43384ba6aea1bcc8fde6c5500267117fa008'
CODES = {'invalid_api_key', 'model_not_found', 'permission_denied', 'insufficient_quota',
         'credit_balance_exhausted', 'billing_hard_limit_reached', 'rate_limit_exceeded'}
TYPES = {'ValueError', 'RuntimeError', 'AuthenticationError', 'PermissionDeniedError',
         'NotFoundError', 'RateLimitError', 'APIConnectionError', 'APITimeoutError', 'BudgetExceeded'}

def run(module):
    diagnostic = {}
    filename = str(BUNDLE / 'runner.py')
    def trace(frame, event, arg):
        if event == 'exception' and frame.f_code.co_filename == filename:
            _, error, _ = arg
            status, error_code = getattr(error, 'status_code', None), getattr(error, 'code', None)
            diagnostic.update(source='runner.py', function=frame.f_code.co_name, line=frame.f_lineno,
                              error_type=type(error).__name__ if type(error).__name__ in TYPES else 'OtherError',
                              http_status=status if type(status) is int else None,
                              error_code=error_code if type(error_code) is str and error_code in CODES else None)
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        code = module.main()
    finally:
        sys.settrace(previous)
    return code, diagnostic

if __name__ == '__main__':
    path = BUNDLE / 'runner.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED:
        sys.exit('Reviewed entrypoint changed; refusing launch')
    spec = importlib.util.spec_from_file_location('supplement', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(path), 'launch']
    code, diagnostic = run(module)
    with (BUNDLE / 'safe-launch-status.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(dict(exit_code=code, diagnostic=diagnostic)) + '\n')
    print(json.dumps(dict(exit_code=code, diagnostic=diagnostic)), flush=True)
    sys.exit(code)
