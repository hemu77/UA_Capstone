"""One entry point; only `launch` may make model calls."""
import argparse
import json

from . import protocol, runner

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('action', choices=['freeze', 'preflight', 'status', 'verify', 'analyze', 'launch', 'recover'])
cli.add_argument('--stage', choices=['hindi_check', 'main'])
cli.add_argument('--approve-usd', type=float)
cli.add_argument('--approval-reference')
cli.add_argument('--auto-recover', action='store_true',
    help='Retire lost-transport attempts and resume (one replacement per request, capped).')
cli.add_argument('--walltime-seconds', type=int,
    help='Pause between requests before scheduler expiry; scheduled pause exits with code 75.')
args = cli.parse_args()
if args.walltime_seconds is not None and args.action != 'launch':
    cli.error('--walltime-seconds is only supported by launch')
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
    elif args.action == 'analyze':
        # Offline: replays saved receipts and reports a complete or PARTIAL study.
        result = runner.analyze_stage(args.stage or 'main')
    elif args.action == 'recover':
        if not args.stage or not args.approval_reference:
            cli.error('recover requires --stage and the original --approval-reference')
        result = runner.recover(args.stage, args.approval_reference)
    else:
        if not args.stage or args.approve_usd is None or not args.approval_reference:
            cli.error('launch requires --stage, --approve-usd and --approval-reference')
        result = runner.run(args.stage, args.approve_usd, args.approval_reference, args.auto_recover,
                            walltime_seconds=args.walltime_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('status') == 'PAUSED_AT_CHECKPOINT':
        raise SystemExit(75)
except Exception as error:
    if isinstance(error, runner.RecoveryRefused):
        cli.exit(2, 'RecoveryRefused: ' + str(error) + '\n')
    cli.exit(1, type(error).__name__ + ': ' +
        ('Live execution stopped; inspect local evidence. If it was a lost provider '
         'request, run `recover` (or relaunch with --auto-recover).\n'
         if args.action == 'launch' else str(error) + '\n'))
