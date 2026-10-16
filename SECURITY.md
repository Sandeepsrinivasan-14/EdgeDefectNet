# Security policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub:
**[Report a vulnerability](https://github.com/Sandeepsrinivasan-14/EdgeDefectNet/security/advisories/new)**
(Security tab, "Report a vulnerability").

Include the affected version or commit, what an attacker can do, and a minimal reproduction. This is a
single-maintainer project: expect an acknowledgement within about a week and a fix or a written assessment
afterwards. Reporters are credited in the release notes unless they prefer otherwise.

## Scope and threat model

In scope:

- The inference code (`edgedefectnet/`), the CLI and the Gradio demo (`app.py`) handling **untrusted images**.
  Images are size- and pixel-limited and decoded by Pillow; a crafted file that crashes, hangs or exhausts memory
  in this code is a valid report.
- The web app in `web/` and the build/deploy workflows in `.github/workflows/`.
- The Docker image (runs as an unprivileged user; the demo has no authentication and should not be exposed to the
  internet without a reverse proxy that provides it).

Out of scope, but worth knowing:

- **Model files are code-like.** Only load ONNX files you trust. Verify downloads with the published SHA-256
  (`edgedefectnet-predict --verify`, or `sha256sum -c models/edgedefectnet.onnx.sha256`).
- Checkpoints for training are loaded with `torch.load(..., weights_only=True)` except the project's own resume
  file (`last.pt`), which contains optimizer state and must only be resumed from a run you made yourself.
- Vulnerabilities in third-party dependencies (Pillow, ONNX Runtime, Gradio, PyTorch) should be reported
  upstream; Dependabot keeps this repository's dependency ranges current.
- The browser app loads three.js and ONNX Runtime Web from public CDNs (pinned versions, with a fallback CDN). It
  does not use Subresource Integrity; self-host them if that matters for your deployment.
