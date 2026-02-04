# Security

## Reporting
Please report security issues by opening a GitHub issue with the "security" label. Do not include sensitive data.

## Scope
- MRE never executes arbitrary user code
- Model paths are restricted to a safe directory or allowlist
- Uploads are size-limited and validated

## Recommendations
- Run MRE locally or in trusted environments
- Avoid loading untrusted model checkpoints
