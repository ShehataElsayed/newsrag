# Release checklist

- [ ] Confirm package name, repository visibility, maintainer identity and license with the owner.
- [ ] Review code and dependencies for secrets and unintended private content.
- [ ] Run Ruff, mypy, tests and build on all supported Python versions.
- [ ] Verify source distribution and wheel install in clean environments.
- [ ] Test real multilingual corpora, long files, citation failure cases and source-date edge cases.
- [ ] Inspect dependency advisories, package metadata and README rendering.
- [ ] Publish a signed/tagged release and verify the public repository and package index pages.

The locally tested prototype is not a substitute for this release review. Do not claim automated fact checking or universal provider compatibility without tests for those claims.
