# Publishing with PyPI Trusted Publishing

This repository includes `.github/workflows/release.yml`. A version tag beginning with `v` triggers checks, builds a wheel and source distribution, then asks PyPI to publish through GitHub OIDC. No API token is stored in the repository.

The owner must complete these first:

1. Confirm the public repository under the correct GitHub account, its exact `owner/newsrag` path, and the desired license. The account/owner cannot be guessed from a similar name.
2. In GitHub, create an environment named `pypi`. Protect it with required reviewers if appropriate, and restrict deployment to version tags if possible. Do not approve a release until the code and artifacts have been reviewed.
3. Sign into the intended PyPI account. In the account sidebar, open **Publishing** and add a **pending GitHub publisher** for project `newsrag`: owner is the exact GitHub account or organization name, repository `newsrag`, workflow filename `release.yml`, environment `pypi`. A pending publisher does not reserve the name.
4. Review the release, ensure the version in `setup.py` and `src/newsrag/__init__.py` matches the intended tag, then create and push an annotated tag such as `v0.2.0` on the reviewed commit. The release workflow will ask for the `pypi` environment approval and publish only after approval.
5. Verify the workflow succeeded and check the actual project on PyPI. An accepted pending publisher or a green build alone is not proof of publication.

Official PyPI instructions: https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/ and https://docs.pypi.org/trusted-publishers/using-a-publisher/ .

If a token was created for a previous manual-upload plan, it is not used by this workflow. The owner can revoke it in PyPI account settings after this route works.
