#!/usr/bin/env python3
"""No-network regression tests for scheduled closeout comment deduplication."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/launch/nudge_242_closeout.sh'
MARKER = '<!-- vanedb-closeout-242:v1 -->'

GH = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
root = Path(os.environ['NUDGE_TEST_DIR'])
args = sys.argv[1:]
cfg = json.loads((root / 'config.json').read_text())
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps(args) + '\n')
comments_path = root / 'comments.json'
comments = json.loads(comments_path.read_text())
if args[:2] == ['pr', 'view']:
    if cfg.get('pr_error'): sys.exit(1)
    print(json.dumps({'state': cfg.get('demo_state', 'OPEN'),
                     'mergeable': cfg.get('mergeable', 'UNKNOWN'), 'headRefOid': 'a' * 40}))
elif args[:2] == ['issue', 'view']:
    if cfg.get('issue_error'): sys.exit(1)
    print(cfg.get('issues', {}).get(args[2], 'OPEN'))
elif args[:2] == ['issue', 'reopen']:
    cfg.setdefault('issues', {})[args[2]] = 'OPEN'
    (root / 'config.json').write_text(json.dumps(cfg))
elif args[:2] == ['issue', 'comment']:
    body = Path(args[args.index('--body-file') + 1]).read_text()
    comments.append({'id': max([c['id'] for c in comments] + [0]) + 1, 'body': body})
    comments_path.write_text(json.dumps(comments))
elif args[:1] == ['api']:
    if any('/releases/tags/' in a for a in args):
        if cfg.get('release_error'):
            print('{"status":"503"}')
            sys.exit(1)
        if cfg.get('release'):
            print('{"tag_name":"0.2.0"}')
        else:
            print('{"status":"404"}')
            sys.exit(1)
    elif '--method' in args:
        assert args[args.index('--method') + 1] == 'PATCH', args
        if cfg.get('patch_error'): sys.exit(1)
        endpoint = next(a for a in args if a.startswith('repos/'))
        cid = int(endpoint.rsplit('/', 1)[1])
        field = args[args.index('--field') + 1]
        assert field.startswith('body=@'), args
        comment = next(c for c in comments if c['id'] == cid)
        comment['body'] = Path(field[len('body=@'):]).read_text()
        comments_path.write_text(json.dumps(comments))
    elif any('/issues/242/comments?' in a for a in args):
        if cfg.get('comments_error'):
            print(json.dumps([comments[:100]]))  # A later page may have failed.
            sys.exit(1)
        if 'comments_response' in cfg:
            print(cfg['comments_response'])
        else:
            assert '--paginate' in args and '--slurp' in args, args
            # Model gh's separate page arrays, including a marker beyond page 1.
            print(json.dumps([comments[i:i+100] for i in range(0, len(comments), 100)] or [[]]))
    elif any('/contents/' in a for a in args):
        sys.exit(0 if cfg.get('vault', True) else 1)
    elif '/installation/repositories' in args:
        if '--jq' in args:
            print('vanedb/obsidian-vane-search' if cfg.get('app') else 'vanedb/vanedb')
        else:
            print('{}')
    else:
        sys.exit('Unexpected gh api: ' + repr(args))
else:
    sys.exit('Unexpected gh: ' + repr(args))
'''


class NudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='closeout-nudge-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        script = self.root / 'docs/launch/nudge_242_closeout.sh'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, script)
        self.script = script
        (self.root / 'bench').mkdir()
        self.comparison('*Pending.*\n*Pending.*\n')
        (self.root / 'comments.json').write_text('[]')
        self.config = {}
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        (self.bin / 'gh').write_text(GH)
        (self.bin / 'gh').chmod(0o755)
        (self.bin / 'git').write_text('#!/bin/sh\nprintf "%s\\n" "${NUDGE_TEST_TIP:-aaaaaaa}"\n')
        (self.bin / 'git').chmod(0o755)

    def comparison(self, text):
        (self.root / 'bench/COMPARISON.md').write_text(text)

    def run_nudge(self, *, apply=True, env=None, success=True):
        (self.root / 'config.json').write_text(json.dumps(self.config))
        variables = {**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                     'NUDGE_TEST_DIR': str(self.root), 'DEMO_REPO_TOKEN': '', **(env or {})}
        result = subprocess.run(['bash', str(self.script), *(['--apply'] if apply else [])],
                                env=variables, capture_output=True, text=True)
        self.config = json.loads((self.root / 'config.json').read_text())
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]

    def comments(self):
        return json.loads((self.root / 'comments.json').read_text())

    def writes(self):
        return [c for c in self.calls() if c[:2] in [['issue', 'comment'], ['issue', 'reopen']]
                or '--method' in c]

    def test_first_post_then_unchanged_repeat_is_silent(self):
        self.run_nudge()
        self.assertTrue(self.comments()[0]['body'].startswith(MARKER))
        self.run_nudge()
        self.assertEqual(len(self.comments()), 1)
        self.assertEqual(len(self.writes()), 1)

    def test_incidental_tip_token_app_and_mergeability_do_not_repost_or_edit(self):
        self.run_nudge()
        self.config.update(app=True, mergeable='MERGEABLE')
        self.run_nudge(env={'NUDGE_TEST_TIP': 'bbbbbbb', 'DEMO_REPO_TOKEN': 'fixture'})
        self.assertEqual(len(self.writes()), 1)

    def test_each_meaningful_state_change_updates_same_comment(self):
        self.run_nudge()
        self.comparison('*Pending.*\n')
        self.run_nudge()
        for update in [{'demo_state': 'MERGED'}, {'vault': False}, {'release': True}]:
            self.config.update(update)
            self.run_nudge()
        self.assertEqual(len(self.comments()), 1)
        self.assertEqual(sum('--method' in c for c in self.calls()), 4)
        self.assertIn('release is live', self.comments()[0]['body'])

    def test_marker_on_later_page_is_found(self):
        self.run_nudge()
        status = self.comments()[0]
        status['id'] = 250
        others = [{'id': i, 'body': 'unrelated'} for i in range(1, 151)]
        (self.root / 'comments.json').write_text(json.dumps(others + [status]))
        self.run_nudge()
        self.assertEqual(len(self.comments()), 151)
        self.assertEqual(len(self.writes()), 1)

    def test_read_failures_never_blindly_post(self):
        for cfg in [{'comments_error': True}, {'comments_response': 'garbled'},
                    {'comments_response': '{}'}, {'comments_response': '[]'},
                    {'comments_response': '[[{}]]'}, {'issue_error': True}]:
            with self.subTest(cfg=cfg):
                self.config = cfg
                self.run_nudge(success=False)
                self.assertEqual(self.writes(), [])

    def test_duplicate_markers_fail_without_mutating(self):
        self.run_nudge()
        comment = self.comments()[0]
        (self.root / 'comments.json').write_text(json.dumps([comment, {**comment, 'id': 2}]))
        self.run_nudge(success=False)
        self.assertEqual(len(self.writes()), 1)

    def test_patch_failure_does_not_fall_back_to_duplicate_post(self):
        self.run_nudge()
        self.config.update(patch_error=True, demo_state='MERGED')
        self.run_nudge(success=False)
        self.assertEqual(sum(c[:2] == ['issue', 'comment'] for c in self.calls()), 1)

    def test_closed_trackers_reopen_even_when_comment_unchanged(self):
        self.run_nudge()
        self.config['issues'] = {'242': 'CLOSED', '198': 'CLOSED'}
        self.run_nudge()
        self.run_nudge()
        self.assertEqual([c[2] for c in self.writes() if c[:2] == ['issue', 'reopen']], ['242', '198'])
        self.assertEqual(sum(c[:2] == ['issue', 'comment'] for c in self.calls()), 1)

    def test_only_both_cleared_gates_skip_all_mutations(self):
        self.comparison('complete\n')
        self.config['release'] = True
        self.config['issues'] = {'242': 'CLOSED', '198': 'CLOSED'}
        self.run_nudge()
        self.assertEqual(self.writes(), [])

    def test_missing_release_still_reopens_with_zero_pending(self):
        self.comparison('complete\n')
        self.config['issues'] = {'242': 'CLOSED', '198': 'CLOSED'}
        self.run_nudge()
        self.assertEqual(len(self.writes()), 3)

    def test_unknown_release_with_zero_pending_does_not_reopen_or_post(self):
        self.comparison('complete\n')
        self.config.update(release_error=True, issues={'242': 'CLOSED', '198': 'CLOSED'})
        self.run_nudge(success=False)
        self.assertEqual(self.writes(), [])

    def test_unknown_release_preserves_known_pending_reopening_without_notification(self):
        self.run_nudge()
        previous = self.comments()
        self.config.update(release_error=True, issues={'242': 'CLOSED', '198': 'CLOSED'})
        self.run_nudge(success=False)
        self.assertEqual([c[2] for c in self.writes() if c[:2] == ['issue', 'reopen']], ['242', '198'])
        self.assertEqual(self.comments(), previous)
        self.assertEqual(sum(c[:2] == ['issue', 'comment'] for c in self.calls()), 1)

    def test_missing_comparison_cannot_clear_gate(self):
        (self.root / 'bench/COMPARISON.md').unlink()
        self.config['release'] = True
        self.config['issues'] = {'242': 'CLOSED', '198': 'CLOSED'}
        self.run_nudge()
        self.assertEqual(len(self.writes()), 3)
        self.assertIn('unknown×', self.comments()[0]['body'])

    def test_dry_run_prints_body_without_mutation_or_comment_history_read(self):
        self.config['issues'] = {'242': 'CLOSED', '198': 'CLOSED'}
        result = self.run_nudge(apply=False)
        self.assertIn('Closeout nudge', result.stdout)
        self.assertEqual(self.writes(), [])
        self.assertFalse(any('/issues/242/comments?' in ' '.join(c) for c in self.calls()))


if __name__ == '__main__':
    unittest.main()
