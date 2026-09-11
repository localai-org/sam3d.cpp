# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import publish_gguf as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for source in set(publication.DOCUMENTS.values()):
            target = self.root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / source).read_bytes())
        self.spec = json.loads((ROOT / 'distribution/body/artifacts.json').read_text())
        self.paths = {}
        for index, name in enumerate(publication.COMPONENTS):
            data = b'GGUF' + struct.pack('<IQQ', 3, index + 1, 1) + b'synthetic only' + bytes([index])
            file = self.root / name
            file.write_bytes(data)
            self.paths[name] = file
            self.spec['files'][name].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (self.root / 'distribution/body/artifacts.json').write_text(json.dumps(self.spec))
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def prepare(self, repo=publication.REPOSITORY):
        return publication.prepare(self.root, self.paths, repo, self.stack)

    def args(self):
        return ['--backbone', str(self.paths[publication.COMPONENTS[0]]),
                '--branch', str(self.paths[publication.COMPONENTS[1]]),
                '--mhr', str(self.paths[publication.COMPONENTS[2]])]

    def test_dry_run_no_hf_import_network_or_absolute_paths(self):
        output = io.StringIO()
        with patch.object(publication, 'ROOT', self.root), patch.dict(sys.modules, {'huggingface_hub': None}), contextlib.redirect_stdout(output):
            self.assertEqual(publication.main(self.args()), 0)
        self.assertIn('Dry run', output.getvalue())
        self.assertNotIn(str(self.root), output.getvalue())

    def test_file_allowlist_checksums_and_license_boundary(self):
        manifest, models, payloads = self.prepare()
        expected = set(publication.COMPONENTS) | set(publication.DOCUMENTS) | {'MANIFEST.json', 'SHA256SUMS'}
        self.assertEqual(set(models) | set(payloads), expected)
        self.assertEqual(payloads['LICENSE'], (ROOT / 'LICENSES/SAM.txt').read_bytes())
        self.assertNotEqual(payloads['LICENSE'], (ROOT / 'LICENSE').read_bytes())
        self.assertIn('sam3d.cpp contributors', payloads['NOTICE'].decode())
        records = json.loads(payloads['MANIFEST.json'])['files']
        self.assertEqual(records, manifest['files'])
        for line in payloads['SHA256SUMS'].decode().splitlines():
            expected_hash, name = line.split('  ')
            data = self.paths[name].read_bytes() if name in models else payloads[name]
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected_hash)
        self.assertEqual(len(payloads['SHA256SUMS'].decode().splitlines()), len(expected) - 1)

    def test_repo_override_updates_card_and_manifest(self):
        manifest, _, payloads = self.prepare('test-org/Body-GGUF')
        self.assertEqual(manifest['repository'], 'test-org/Body-GGUF')
        card = payloads['README.md'].decode()
        self.assertNotIn(publication.REPOSITORY, card)
        self.assertIn('hf download test-org/Body-GGUF', card)
        publication.validate_card(card)

    def test_reject_bad_repository(self):
        for value in ['../bad', '/tmp/repo', 'https://hf.co/a/b', 'a/b/c', 'facebook/model',
                      'a/b..c', 'a/repo.git', 'a/repo--bad', 'a/x\nSECRET', 'a/@token']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                publication.validate_repo(value)

    def test_reject_hash_size_and_format_mismatch(self):
        path = self.paths[publication.COMPONENTS[0]]
        original = path.read_bytes()
        for data in [original[:-1], b'bad!' + original[4:], original[:-1] + b'z']:
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                self.prepare()
        path.write_bytes(original)

    def test_reject_missing_symlink_fifo_and_extra_component(self):
        path = self.paths[publication.COMPONENTS[0]]
        path.unlink()
        with self.assertRaises(OSError):
            self.prepare()
        path.symlink_to(self.paths[publication.COMPONENTS[1]])
        with self.assertRaises(OSError):
            self.prepare()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaisesRegex(ValueError, 'regular files'):
            self.prepare()
        self.paths['private.key'] = path
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.prepare()

    def test_reject_wrong_card_license_base_or_missing_notice(self):
        card = self.root / 'distribution/body/README.md'
        original = card.read_text()
        for text in [original.replace('license: other', 'license: apache-2.0'),
                     original.replace('base_model: facebook/', 'base_model: wrong/'),
                     original + '{{unresolved}}']:
            card.write_text(text)
            with self.assertRaises(ValueError):
                self.prepare()
        card.write_text(original)
        (self.root / 'LICENSES/DINOv3.md').unlink()
        with self.assertRaises(OSError):
            self.prepare()

    def test_duplicate_specification_key(self):
        file = self.root / 'distribution/body/artifacts.json'
        file.write_text('{"schema_version":1,"schema_version":2}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.prepare()

    def test_cli_publication_flags_required_before_validation(self):
        for extra in [['--upload'], ['--create-repo']]:
            with patch.object(publication, 'prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    publication.main(self.args() + extra)
                prepare.assert_not_called()

    def test_dirty_uncommitted_or_changed_source_rejects_before_hf(self):
        manifest, models, payloads = self.prepare()
        with patch.dict(sys.modules, {'huggingface_hub': None}):
            with self.assertRaisesRegex(ValueError, 'clean source'):
                publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads)
            manifest['conversion_source'] = {'commit': 'a' * 40, 'dirty': False}
            with patch.object(publication, 'source_state', return_value={'commit': 'b' * 40, 'dirty': False}):
                with self.assertRaisesRegex(ValueError, 'source checkout changed'):
                    publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads)

    def fake_hf(self):
        module = types.ModuleType('huggingface_hub')
        errors = types.ModuleType('huggingface_hub.errors')
        errors.RepositoryNotFoundError = type('RepositoryNotFoundError', (Exception,), {})
        module.HfApi = MagicMock()
        module.HfApi.return_value.model_info.return_value.sha = 'remote-parent'
        module.ModelCard = MagicMock()
        module.CommitOperationAdd = lambda **kwargs: kwargs
        return module, errors

    def test_upload_one_commit_exact_allowlist_and_no_creation(self):
        state = {'commit': 'a' * 40, 'dirty': False}
        with patch.object(publication, 'source_state', return_value=state):
            manifest, models, payloads = self.prepare()
            hf, errors = self.fake_hf()
            with patch.dict(sys.modules, {'huggingface_hub': hf, 'huggingface_hub.errors': errors}):
                publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads)
            api = hf.HfApi.return_value
            api.create_repo.assert_not_called()
            api.create_commit.assert_called_once()
            call = api.create_commit.call_args.kwargs
            self.assertEqual(call['parent_commit'], 'remote-parent')
            self.assertEqual({op['path_in_repo'] for op in call['operations']}, set(models) | set(payloads))
            hf.ModelCard.return_value.validate.assert_called_once()

    def test_creation_requires_explicit_flag_and_is_public(self):
        state = {'commit': 'a' * 40, 'dirty': False}
        with patch.object(publication, 'source_state', return_value=state):
            manifest, models, payloads = self.prepare()
            hf, errors = self.fake_hf()
            api = hf.HfApi.return_value
            api.model_info.side_effect = errors.RepositoryNotFoundError()
            with patch.dict(sys.modules, {'huggingface_hub': hf, 'huggingface_hub.errors': errors}):
                with self.assertRaisesRegex(ValueError, 'creation requires'):
                    publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads)
                api.create_repo.assert_not_called()
                api.model_info.side_effect = [errors.RepositoryNotFoundError(), types.SimpleNamespace(sha='new-parent')]
                publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads, True)
            api.create_repo.assert_called_once_with(publication.REPOSITORY, repo_type='model', private=False, exist_ok=False)

    def test_modified_open_model_is_rejected_before_upload(self):
        state = {'commit': 'a' * 40, 'dirty': False}
        with patch.object(publication, 'source_state', return_value=state):
            manifest, models, payloads = self.prepare()
            self.paths[publication.COMPONENTS[0]].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'model file changed'):
                publication.upload(self.root, publication.REPOSITORY, manifest, models, payloads)

    def test_network_error_payload_is_not_logged(self):
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(publication, 'ROOT', self.root), \
             patch.object(publication, 'source_state', return_value={'commit': 'a' * 40, 'dirty': False}), \
             patch.object(publication, 'upload', side_effect=OSError('sensitive response body')), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(publication.main(self.args() + ['--upload', '--accept-model-licenses']), 1)
        self.assertNotIn('sensitive response body', output.getvalue() + errors.getvalue())
        self.assertIn('OSError', errors.getvalue())

    def test_license_installation_and_gitignore(self):
        self.assertEqual((ROOT / 'LICENSE').read_bytes(), (ROOT / 'LICENSES/MHR-Apache-2.0.txt').read_bytes())
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('install(FILES LICENSE NOTICE DESTINATION', cmake)
        self.assertIn('install(FILES docs/LICENSING.md DESTINATION', cmake)
        self.assertIn('install(DIRECTORY LICENSES/ DESTINATION', cmake)
        self.assertTrue((ROOT / 'LICENSES/Trellis2cpp-MIT.txt').is_file())
        ignored = ['.env', 'dir/.env.secret', 'foo.gguf', 'foo.ckpt', 'foo.key', 'photo.jpg',
                   'generated/photo.png', 'models/file', 'build/app', 'CMakeUserPresets.json',
                   'reference/python/.venv/bin/python']
        kept = ['demo/web/localai.png', 'tests/fixtures/point-condition-0.weights', '.env.example', 'distribution/body/README.md']
        result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '--stdin'],
                                input='\n'.join(ignored + kept) + '\n', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(set(result.stdout.splitlines()), set(ignored))


if __name__ == '__main__':
    unittest.main()
