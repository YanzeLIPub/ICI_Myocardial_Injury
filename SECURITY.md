# Security and privacy

Report a suspected exposure privately to **yangkunyu@hust.edu.cn**. Describe the affected file or commit and the nature of the concern without attaching patient data, credentials, or trained artifacts to a public issue.

The public release is built from an explicit file allowlist. It contains no clinical dataset or downloadable trained model. The release check rejects unlisted tracked files, binary artifacts, credentials in recognized formats, and local absolute paths. The checks complement manual review; they do not establish that a clinical dataset has been anonymized.

## Contributor rules

- Submit source changes only. Do not add clinical rows, source exports, notebooks with outputs, screenshots, logs, model files, prediction tables, or empirical reference arrays.
- Use small artificial inputs for tests. Do not generate test examples by querying the study model.
- Do not put access credentials in code, documentation, workflow variables, or attachments.
- Keep any authorized clinical work outside the public repository and its Git history.
- Run the release check and tests before proposing a change. Changes to the publication file allowlist require explicit review.

The CI workflow uses read-only repository permissions and receives no clinical data, trained artifacts, or project secrets. The repository is not a deployment service.
