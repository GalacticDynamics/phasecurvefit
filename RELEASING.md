# Release Process for phasecurvefit

This package uses **uv** for dependency management and **hatch-vcs** for
automatic version detection from git tags.

## Versioning with hatch-vcs

The version is automatically detected from git tags.

### Tag Format

Tags should follow the format: `vX.Y.Z` (e.g., `v0.1.0`)

The version must follow PEP 440 format: `X.Y.Z` with optional suffixes like
`a1`, `b2`, `rc1`, `.post1`, `.dev0`, etc.

## How Versioning Works

When you tag a commit:

- The package version matches the tag (e.g., tag `v0.1.0` → version `0.1.0`)
- After the tag, development versions are created automatically (e.g.,
  `0.1.1.dev5+gabc1234`)

## Release Workflows

### Release via Git Tags (Recommended)

1. **Create and push a tag:**

   ```bash
   git tag v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```

2. **Build and publish:**

   ```bash
   # Build the package (version will be detected from tags)
   uv build

   # Publish to PyPI
   uv publish
   ```

3. **Create a GitHub Release:**
   - Go to https://github.com/GalacticDynamics/phasecurvefit/releases/new
   - Choose or create the tag `v0.1.0`
   - Fill in release notes
   - Publish the release

### Manual Build and Publish

```bash
# From repository root
uv build
uv publish
```

**Note**: Publishing requires PyPI credentials configured.

## Pre-release Checklist

Before creating a release:

1. **Run all tests:**

   ```bash
   uv run nox -s test
   ```

2. **Check code quality:**

   ```bash
   uv run nox -s lint
   ```

3. **Build documentation:**

   ```bash
   uv run nox -s docs
   ```

4. **Update CHANGELOG** (if you maintain one)

## Version Bump Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **Patch** (0.0.x): Bug fixes, documentation updates
- **Minor** (0.x.0): New features, backwards-compatible changes
- **Major** (x.0.0): Breaking changes

## Publishing to PyPI

Ensure you have PyPI credentials configured:

```bash
# Set PyPI token (recommended)
export UV_PUBLISH_TOKEN=pypi-...

# Or configure in ~/.pypirc
```

Then publish:

```bash
uv publish
```

## Troubleshooting

- **Version not detected**: Ensure you've pushed the tag and it follows the
  `vX.Y.Z` format
- **Build fails**: Check that all tests pass and dependencies are correctly
  specified
- **Publish fails**: Verify PyPI credentials and that the version doesn't
  already exist on PyPI
