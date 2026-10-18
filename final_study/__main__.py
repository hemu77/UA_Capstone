"""One entry point; only `launch` may make model calls."""
import argparse
import json

from . import protocol, runner

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('action', choices=['freeze', 'preflight', 'status', 'verify', 'launch'])
cli.add_argument('--stage', choices=['hindi_check', 'main'])
cli.add_argument('--approve-usd', type=float)
cli.add_argument('--approval-reference')
args = cli.parse_args()
try:
    if args.action == 'freeze':
        result = protocol.freeze()
    elif args.action == 'preflight':
        result = runner.preflight()
    elif args.action == 'status':
        result = runner.status()
    elif args.action == 'verify':
        if not args.stage:
            cli.error('--stage required')
        result = runner.verify_stage(protocol.load_contract(), args.stage)
        result = {k: v for k, v in result.items() if k != 'results'}
    else:
        if not args.stage or args.approve_usd is None or not args.approval_reference:
            cli.error('launch requires --stage, --approve-usd and --approval-reference')
        result = runner.launch(args.stage, args.approve_usd, args.approval_reference)
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as error:
    cli.exit(1, type(error).__name__ + ': ' +
        ('Live execution stopped; inspect local evidence. No automatic paid resend.\n'
         if args.action == 'launch' else str(error) + '\n'))
