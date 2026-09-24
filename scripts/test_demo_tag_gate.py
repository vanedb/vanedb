#!/usr/bin/env python3
"""No-network regression checks for the demo merge and final publication gates."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/launch/maintainer_tag_demo_0.2.0.sh'
HEAD = 'a' * 40
GREEN = {'name': 'test', '__typename': 'CheckRun', 'status': 'COMPLETED', 'conclusion': 'SUCCESS'}
PR = {'state': 'OPEN', 'headRefOid': HEAD, 'mergeable': 'MERGEABLE',
      'mergeStateStatus': 'CLEAN', 'statusCheckRollup': [GREEN],
      'url': 'https://github.com/vanedb/obsidian-vane-search/pull/20'}


class DemoTagGateTests(unittest.TestCase):
    def run_gate(self, pr=None, *, vault=True, args=(), final=None):
        with tempfile.TemporaryDirectory(prefix='demo-tag-gates-') as tmp:
            root = Path(tmp)
            (root / 'pr.json').write_text(json.dumps(PR if pr is None else pr))
            config = {'runs': [{'name': 'test', 'status': 'completed', 'conclusion': 'success'}],
                      'statuses': [], 'build_fail': '', 'record': 'valid', 'bundle': 'accepted-bundle'}
            config.update(final or {})
            (root / 'final.json').write_text(json.dumps(config))
            (root / 'accepted-sha').write_text(hashlib.sha256(b'accepted-bundle').hexdigest())
            gh = root / 'gh'
            gh.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['DEMO_TEST_DIR'])
args=sys.argv[1:]
with (root/'calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
if args[:2] == ['pr','view']:
    print((root/'pr.json').read_text())
elif args[:2] == ['pr','merge']:
    # Stop before any side effect. This is the only accepted positive endpoint.
    raise SystemExit(73)
elif args[:1] == ['api'] and '/check-runs?' in args[-1]:
    print(json.dumps([{'check_runs': json.loads((root/'final.json').read_text())['runs']}]))
elif args[:1] == ['api'] and '/status?' in args[-1]:
    print(json.dumps([{'statuses': json.loads((root/'final.json').read_text())['statuses']}]))
elif args[:1] == ['api'] and args[1].startswith('repos/vanedb/obsidian-vane-search/contents/'):
    raise SystemExit(0 if os.environ['DEMO_TEST_VAULT']=='1' else 1)
else:
    raise SystemExit('Unexpected gh invocation: '+repr(args))
''')
            gh.chmod(0o755)
            git = root / 'git'
            git.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['DEMO_TEST_DIR']);args=sys.argv[1:]
with (root/'git-calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
if args[0]=='clone':
    dest=Path(args[-1]);dest.mkdir();(dest/'manifest.json').write_text('{"version":"0.2.0"}')
    rec=dest/'docs/releases/0.2.0-desktop-acceptance.md';rec.parent.mkdir(parents=True)
    mode=json.loads((root/'final.json').read_text())['record']
    if mode!='missing':
        sha=(root/'accepted-sha').read_text() if mode=='valid' else 'malformed'
        rec.write_text('| Installed `main.js` SHA-256 | `'+sha+'` |\\n')
elif args[0]=='rev-parse' and args[1].startswith('refs/tags/'):
    raise SystemExit(1)
elif args[0] in ['config','fetch','checkout','ls-remote','tag','push']:
    pass
else:
    raise SystemExit('unexpected git invocation: '+repr(args))
""")
            git.chmod(0o755)
            npm = root / 'npm'
            npm.write_text("""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
root=Path(os.environ['DEMO_TEST_DIR']);args=sys.argv[1:]
with (root/'build-calls.jsonl').open('a') as f: f.write(json.dumps(args)+'\\n')
cfg=json.loads((root/'final.json').read_text())
if ' '.join(args)==cfg['build_fail']:raise SystemExit(74)
if args==['run','build']:Path('main.js').write_text(cfg['bundle'])
""")
            npm.chmod(0o755)
            node = root / 'node'
            node.write_text('#!/bin/sh\nexit 0\n')
            node.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                   'DEMO_REPO_TOKEN': 'test-fixture-not-a-real-token',
                   'DEMO_TEST_DIR': str(root), 'DEMO_TEST_VAULT': '1' if vault else '0'}
            env.pop('DEMO_URL', None)
            result = subprocess.run(['bash', str(SCRIPT), '--confirm-vault-walkthrough',
                                     '--merge-if-open', *args], env=env, capture_output=True, text=True)
            calls_path = root / 'calls.jsonl'
            calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
            for name in ['git-calls.jsonl', 'build-calls.jsonl']:
                path = root / name
                if path.exists():
                    calls += [[name, *json.loads(line)] for line in path.read_text().splitlines()]
            return result, calls

    def assert_refused(self, pr=None, **kwargs):
        result, calls = self.run_gate(pr, **kwargs)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any(call[:2] == ['pr', 'merge'] for call in calls))

    def test_only_clean_merge_state_is_accepted(self):
        for state in ['UNKNOWN', 'UNSTABLE', 'HAS_HOOKS', 'DIRTY', 'DRAFT', 'BLOCKED', 'BEHIND', '']:
            with self.subTest(state=state):
                self.assert_refused({**PR, 'mergeStateStatus': state})

    def test_missing_or_invalid_head_is_refused(self):
        for head in ['', 'abcdefg', 'a' * 39, 'g' * 40]:
            with self.subTest(head=head):
                self.assert_refused({**PR, 'headRefOid': head})

    def test_check_rollup_must_exist(self):
        for checks in [[], None]:
            with self.subTest(checks=checks):
                self.assert_refused({**PR, 'statusCheckRollup': checks})

    def test_failed_pending_or_neutral_checks_are_refused_even_when_clean(self):
        for check in [
            {**GREEN, 'status': 'IN_PROGRESS', 'conclusion': ''},
            *[{**GREEN, 'conclusion': value} for value in
              ['FAILURE', 'CANCELLED', 'TIMED_OUT', 'NEUTRAL', 'ACTION_REQUIRED', 'STALE']],
            {'__typename': 'StatusContext', 'state': 'PENDING'},
            {'__typename': 'StatusContext', 'state': 'FAILURE'},
        ]:
            with self.subTest(check=check):
                self.assert_refused({**PR, 'statusCheckRollup': [GREEN, check]})

    def test_missing_vault_record_refuses_merge(self):
        self.assert_refused(vault=False)

    def test_preselected_commit_cannot_merge_an_open_pr(self):
        self.assert_refused(args=('--commit=' + 'b' * 40,))

    def test_merged_pr_cannot_tag_a_different_commit(self):
        merged = {**PR, 'state': 'MERGED', 'mergeCommit': {'oid': 'b' * 40}}
        for commit in ['c' * 40, 'c' * 7, 'b' * 7, 'invalid']:
            with self.subTest(commit=commit):
                self.assert_refused(merged, args=('--commit=' + commit,))

    def test_merged_pr_needs_valid_merge_identity(self):
        for oid in ['', 'b' * 7, 'x' * 40]:
            with self.subTest(oid=oid):
                self.assert_refused({**PR, 'state': 'MERGED', 'mergeCommit': {'oid': oid}})

    def test_final_merged_ci_must_be_green_before_build_or_tag(self):
        merged = {**PR, 'state': 'MERGED', 'mergeCommit': {'oid': 'b' * 40}}
        for final in [
            {'runs': []},
            {'runs': [{'name': 'unrelated', 'status': 'completed', 'conclusion': 'success'}]},
            {'runs': [{'status': 'in_progress', 'conclusion': None}]},
            {'runs': [{'status': 'completed', 'conclusion': 'failure'}]},
            {'statuses': [{'state': 'pending'}]},
        ]:
            with self.subTest(final=final):
                result, calls = self.run_gate(merged, final=final)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('final merged-commit CI', result.stderr)
                self.assertFalse(any(call[0] == 'build-calls.jsonl' or call[:2] == ['git-calls.jsonl', 'tag'] for call in calls))

    def test_failed_final_build_never_tags(self):
        merged = {**PR, 'state': 'MERGED', 'mergeCommit': {'oid': 'b' * 40}}
        for command in ['ci', 'run check:release', 'run typecheck', 'test', 'run build']:
            with self.subTest(command=command):
                result, calls = self.run_gate(merged, final={'build_fail': command})
                self.assertEqual(result.returncode, 74)
                self.assertFalse(any(call[:2] == ['git-calls.jsonl', 'tag'] for call in calls))

    def test_bad_or_missing_acceptance_and_changed_bundle_never_tag(self):
        merged = {**PR, 'state': 'MERGED', 'mergeCommit': {'oid': 'b' * 40}}
        for final in [{'record': 'missing'}, {'record': 'malformed'}, {'bundle': 'changed-bundle'}]:
            with self.subTest(final=final):
                result, calls = self.run_gate(merged, final=final)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('refused:', result.stderr)
                self.assertFalse(any(call[:2] == ['git-calls.jsonl', 'tag'] for call in calls))

    def test_green_final_merge_build_and_bundle_can_tag(self):
        merged = {**PR, 'state': 'MERGED', 'mergeCommit': {'oid': 'b' * 40}}
        result, calls = self.run_gate(merged, args=('--commit=' + 'b' * 40,))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call[1:] for call in calls if call[0]=='build-calls.jsonl'],
                         [['ci'], ['run','check:release'], ['run','typecheck'], ['test'], ['run','build']])
        self.assertTrue(any(call[:3] == ['git-calls.jsonl', 'tag', '-a'] for call in calls))
        self.assertTrue(any(call[:2] == ['git-calls.jsonl', 'push'] for call in calls))
        self.assertTrue(any(call[0]=='api' and ('/commits/' + 'b'*40 + '/check-runs?') in call[-1] for call in calls))

    def test_explicit_local_bare_remote_plumbing_remains_supported(self):
        git = shutil.which('git')
        self.assertIsNotNone(git)
        with tempfile.TemporaryDirectory(prefix='demo-tag-bare-') as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            env = {**os.environ, 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_SYSTEM': os.devnull}
            env.pop('DEMO_REPO_TOKEN', None)
            def run(*args):
                return subprocess.run([git, *args], env=env, capture_output=True, text=True, check=True)
            run('init', '--initial-branch=main', str(source))
            run('-C', str(source), 'config', 'user.name', 'Test Fixture')
            run('-C', str(source), 'config', 'user.email', 'fixture@example.invalid')
            (source / 'manifest.json').write_text('{"version":"0.2.0"}')
            run('-C', str(source), 'add', 'manifest.json')
            run('-C', str(source), 'commit', '-m', 'test fixture')
            commit = run('-C', str(source), 'rev-parse', 'HEAD').stdout.strip()
            bare = root / 'remote.git'
            run('clone', '--bare', str(source), str(bare))
            env['DEMO_URL'] = bare.as_uri()
            result = subprocess.run(['bash', str(SCRIPT), '--confirm-vault-walkthrough', '--commit=' + commit],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            tagged = run('--git-dir=' + str(bare), 'rev-list', '-n', '1', 'refs/tags/0.2.0').stdout.strip()
            self.assertEqual(tagged, commit)
            self.assertEqual(run('--git-dir=' + str(bare), 'cat-file', '-t', 'refs/tags/0.2.0').stdout.strip(), 'tag')

    def test_green_merge_is_pinned_to_the_checked_head(self):
        for extra in [{**GREEN, 'conclusion': 'SKIPPED'},
                      {'__typename': 'StatusContext', 'state': 'SUCCESS'}]:
            with self.subTest(extra=extra):
                result, calls = self.run_gate({**PR, 'statusCheckRollup': [GREEN, extra]})
                self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
                merge = [call for call in calls if call[:2] == ['pr', 'merge']]
                self.assertEqual(merge, [['pr', 'merge', '20', '-R', 'vanedb/obsidian-vane-search',
                                          '--merge', '--match-head-commit', HEAD]])
                self.assertTrue(any(call[0] == 'api' and call[1].endswith('?ref=' + HEAD) for call in calls))


if __name__ == '__main__':
    unittest.main()
