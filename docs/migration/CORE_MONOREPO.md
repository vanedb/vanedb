# Core monorepo migration record

This record makes the history rewrite auditable and keeps the original commit
IDs discoverable after `vanedb-cpp` and `vanedb-bench` move into
`vanedb/vanedb`.

## Frozen inputs

The rehearsal and migration branch start from these merged `main` commits:

| Repository | Source commit | Imported path | Rewritten head |
|---|---|---|---|
| `vanedb/vanedb` | `9ad023f91efb19ca0c884fb48059a36c852decb5` | repository root | unchanged |
| `vanedb/vanedb-cpp` | `534c93b7c5e8b1ff67fc094a74fd24e9ff5f36fe` | `cpp/` | `13ca0689a8e3e6285815620ec2ad42dc0befdf21` |
| `vanedb/vanedb-bench` | `836f43743cf79b59fffda268ad11f35a216ba413` | `bench/` | `07d75e44d99f38bd0b1da0c6c226635f298a729c` |

All three worktrees were clean and the organization had no open pull requests
when the inputs were frozen. Only each source repository's merged `main`
history is imported. The archived source repositories retain any unmerged
historical branches.

## Rewrite

The imported clones were rewritten with `git-filter-repo` 2.47.0:

```bash
git filter-repo --to-subdirectory-filter cpp \
  --tag-rename '':'vanedb-cpp-' --force
git filter-repo --to-subdirectory-filter bench --force
```

The rewritten histories were merged with `--allow-unrelated-histories`.
Commit messages, authorship, author dates, and parent relationships inside
each imported history are preserved; commit IDs necessarily change because
every path changes.

- C++: 51 commits on imported `main`, plus one tag-only mapped commit (52
  mapping entries total).
- Benchmark: 14 commits on imported `main`.
- Host plus both imports: 148 commits after the two import merges and the
  benchmark-localization commit.
- Historical C++ tag `archive/docs-superpowers` becomes
  `vanedb-cpp-archive/docs-superpowers`.

Full old-to-new mappings:

- [C++ commit map](vanedb-cpp-commit-map.txt)
- [benchmark commit map](vanedb-bench-commit-map.txt)

## Operational changes

- `bench/` consumes `../vanedb-capi` and `../cpp`; its Git dependency and
  C++ submodule are removed.
- Both benchmark engines report one monorepo revision.
- Root CI detects affected components, calls Rust/C++/integration workflows,
  and always finishes with `Required CI Gate`.
- Nested C++ workflows, CODEOWNERS, and Dependabot configuration move to the
  repository root.
- Canonical Python releases use `vanedb-vX.Y.Z`; supplementary C++ releases
  use `vanedb-cpp-vX.Y.Z`.
- `obsidian-vane-search` remains a separate downstream application.

## Merge and archival gates

The old repositories must remain writable until all of the following are true:

1. the migration PR passes Rust, C++, Python-wheel, cross-engine, and history
   acceptance checks;
2. the migration PR is merged;
3. branch protection on `vanedb/vanedb` requires `Required CI Gate`;
4. open C++ and benchmark issues are transferred or replaced by canonical
   monorepo issues, with paired Rust/C++ bugs represented once and labeled for
   both components;
5. source repository READMEs point to their new monorepo directories.

Only then should `vanedb-cpp` and `vanedb-bench` be archived. Archival is
the rollback boundary: before it, the migration can be abandoned without
changing either source repository.
