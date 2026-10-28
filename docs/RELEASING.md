# Publishing the Python and Go alpha

The Python version is `0.2.0a1`; the corresponding Go version is
`v0.2.0-alpha.1`. Both are experimental. Do not describe them as stable or
production-ready. Keep the one-station evaluation, neural convergence warnings,
and all four ensemble losses visible in the README and release notes.

## Validate distributions

Start from a clean checkout with passing CI. Build into an empty output directory:

```bash
python -m pip install build twine
python -m build --outdir /tmp/opentrace-dist
python -m twine check --strict /tmp/opentrace-dist/*
python scripts/verify_distribution.py --dist-dir /tmp/opentrace-dist --version 0.2.0a1
```

The verifier creates separate virtual environments for the wheel and sdist,
resolves all declared dependencies, runs `pip check`, and exercises public APIs
outside the checkout with isolated Python. It does not inherit system packages
or rely on editable installs. CI runs this same check.

## Python publication

Configure a PyPI Trusted Publisher for `opentrace-ml` with:

| Setting | Value |
|---|---|
| Owner | `vrajpatell` |
| Repository | `opentrace-ml` |
| Workflow filename | `python-release.yml` |
| Environment | `pypi` |

For a new project, configure a pending publisher in the maintainer's PyPI
account. The first successful upload creates the project. This grants publishing
authority to that exact workflow and environment; configure it only in the
maintainer-controlled account. No long-lived API token belongs in this repository.

Create `v0.2.0a1` on the reviewed commit and publish a GitHub **prerelease** for
that tag. The workflow builds/checks the artifacts, publishes through OIDC, then
installs the exact version from PyPI in a fresh environment. For an upload that
failed before publishing any files, rerun the failed jobs; manual workflow runs
must select the same release tag. Never overwrite a published version or move its
tag. If an upload succeeded but verification failed, rerun only verification.

External verification after upload:

```bash
python scripts/verify_distribution.py --requirement opentrace-ml==0.2.0a1 \
  --version 0.2.0a1 --index-url https://pypi.org/simple
```

## Go publication

After Go CI passes, tag the same reviewed commit with `go/v0.2.0-alpha.1`.
The directory prefix is mandatory because `go/go.mod` defines a nested module.
Publish a GitHub prerelease for that tag if release notes are desired; Go package
distribution itself uses the Git tag and module proxy, without a separate upload.

Verify outside this repository with a fresh module/cache:

```bash
mkdir /tmp/opentrace-go-consumer
cd /tmp/opentrace-go-consumer
go mod init example.com/opentrace-consumer
GOMODCACHE=/tmp/opentrace-go-cache GOPROXY=https://proxy.golang.org \
  go get github.com/vrajpatell/opentrace-ml/go@v0.2.0-alpha.1
```

Copy the Go README quick-start into `main.go`, then run `go run .` using the
same `GOMODCACHE` and proxy settings. Do not use a local `replace` directive.
Confirm the resolved version with `go list -m all`. Published Go tags are
immutable; corrections require a new version.

A GitHub release or green build is not proof of a PyPI upload or a successful
proxy fetch. Record each distribution's status separately.
