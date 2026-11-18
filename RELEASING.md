<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Releasing radixshmem

A release is a `v<version>` tag. Pushing it runs `.github/workflows/wheels.yml`, which builds and tests the wheels,
publishes them to TestPyPI, and, for a final release and after an approval, to PyPI and a GitHub Release.

## One-time setup

1. On [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org), separate accounts, add a trusted publisher
   for the project `radixshmem` (a pending publisher until the first upload): owner `ai-dynamo`, repository
   `radixshmem`, workflow `wheels.yml`, environment `pypi` on PyPI and `testpypi` on TestPyPI. No API token is
   stored anywhere; the workflow authenticates through GitHub's OIDC token.
2. In the repository settings, create the environments `testpypi` and `pypi`. Give `pypi` required reviewers (the
   release managers) and limit both to tags matching `v*`.
3. Add a tag ruleset so that only release managers can create tags matching `v*`.

A pending publisher does not reserve the name on PyPI: the first upload does.

## Versions

- The version is defined once, in `python/radixshmem/_version.py`. The tag is `v` plus that version; the
  workflow refuses a tag that does not match.
- Versions follow PEP 440. Release candidates are `X.Y.ZrcN`; they are published to TestPyPI only.
- Nodes and clients of a cluster refuse each other when the cluster protocol, the cluster manifest or
  `SHM_VERSION` differ. A release that changes one of them raises the minor version, and its notes say so:
  every node and client of a cluster has to be upgraded together.
- A version number cannot be reused on PyPI or TestPyPI, even after its files are deleted.

## Steps

1. Open a pull request that sets `__version__` to the release candidate, for example `0.2.0rc1`, and merge it once
   CI passes.
2. Tag the merged commit and push the tag:

   ```bash
   git tag v0.2.0rc1 <commit>
   git push <remote> v0.2.0rc1
   ```

3. The workflow checks the tag against `_version.py`, runs govulncheck on the Go etcd client, builds the wheels
   for CPython 3.10 to 3.14 in the builder image, tests them on clean ubuntu 22.04, ubuntu 24.04 and Rocky 9
   hosts, and uploads them to TestPyPI.
4. Install the candidate on a test cluster and run the multi-node checks (the two-node Python tests and
   `python/tests/agentic_stress.py`):

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       radixshmem==0.2.0rc1
   ```

5. For the release, set `__version__` to `0.2.0` in a pull request, merge it, and tag `v0.2.0` the same way. After
   the TestPyPI upload, the `publish-pypi` job waits for a reviewer of the `pypi` environment. Approving it
   publishes to PyPI and creates the GitHub Release with the wheels and generated notes; edit the notes to call out
   protocol or shared-memory changes.

## When something fails

- **govulncheck reports a vulnerability**: update the Go modules in `src/registry/etcd_go` (`go get` the fixed
  versions, then `go mod tidy`), update the Go rows of `ATTRIBUTIONS.md`, and release with a new version number.
- **A dependency of the builder image changes** (`packaging/manylinux/Dockerfile`): update `ATTRIBUTIONS.md` in the
  same pull request.
- **A published release is broken**: yank it on PyPI, which keeps its files but stops `pip install radixshmem` from
  choosing it, then fix it and publish a new version. Never delete it to reuse the number.
