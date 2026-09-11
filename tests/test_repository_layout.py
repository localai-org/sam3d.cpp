# SPDX-License-Identifier: Apache-2.0
"""Keep C++/reference boundaries and documentation navigation from drifting."""
from pathlib import Path
import re
import subprocess
import tomllib
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_cpp_root_and_isolated_python_project(self):
        obsolete = ('pyproject.toml', 'uv.lock', 'STATUS.md', 'TODO.md',
                    'DESIGN.md', 'LICENSING.md', 'THIRD_PARTY_NOTICES.md')
        for name in obsolete:
            self.assertFalse((ROOT / name).exists(), name)
        for name in ('CMakeLists.txt', 'LICENSE', 'NOTICE', 'README.md',
                     'docs/ROADMAP.md', 'docs/DESIGN.md', 'docs/LICENSING.md',
                     'docs/DEVELOPMENT.md', 'reference/HISTORY.md'):
            self.assertTrue((ROOT / name).is_file(), name)
        folder = ROOT / 'reference/python'
        project = tomllib.loads((folder / 'pyproject.toml').read_text())
        lock = tomllib.loads((folder / 'uv.lock').read_text())
        self.assertFalse(project['tool']['uv']['package'])
        self.assertEqual(project['project']['name'], 'sam3d-reference-tools')
        own = next(p for p in lock['package'] if p['name'] == project['project']['name'])
        self.assertEqual(own['source'], {'virtual': '.'})
        self.assertFalse(any(p['name'] == 'torch' for p in lock['package']))

    def test_one_current_roadmap_and_archived_history(self):
        design = (ROOT / 'docs/DESIGN.md').read_text()
        self.assertNotIn('Immediate next tasks', design)
        self.assertNotIn('- [ ]', design)
        history = (ROOT / 'reference/HISTORY.md').read_text()
        self.assertIn('This is not the current task list', history)
        self.assertIn('docs/ROADMAP.md', history)

    def test_document_links_and_python_commands(self):
        paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files',
                                         '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0')
        problems = []
        for relative in sorted(set(paths)):
            path = ROOT / relative
            if not path.is_file() or not (path.suffix == '.md' or relative == 'NOTICE'):
                continue
            if relative.startswith('ggml/'):
                continue
            text = path.read_text()
            if relative != 'distribution/body/README.md':  # Rendered into a different HF file tree.
                for target in re.findall(r'\]\(([^\s)]+)\)', text):
                    value = urlsplit(target)
                    if value.scheme or value.netloc or not value.path:
                        continue
                    if not (path.parent / unquote(value.path)).exists():
                        problems.append(f'{relative}: missing link {target}')
            if re.search(r'\buv (?:run|sync) --frozen\b', text):
                problems.append(f'{relative}: uv command needs --project reference/python')
        self.assertEqual(problems, [])


if __name__ == '__main__':
    unittest.main()
