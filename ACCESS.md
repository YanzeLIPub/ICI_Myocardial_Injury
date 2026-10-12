# Controlled access to data and the trained model

## Public materials

The public release contains analysis source code, software checks using artificial inputs, and documentation. It excludes clinical records, patient-level derived tables, trained parameters, fitted transformations, empirical reference distributions, and predictions from the study model. No inference endpoint is provided.

## Request process

Send an initial request to Yanze Li at **yanzeli95@gmail.com**, including:

1. The investigator's name, institution, and institutional contact information.
2. A concise research question and the materials requested.
3. The proposed analysis, intended users, and computing environment.
4. The relevant ethics and institutional approvals, or a description of the approval process underway.
5. Proposed safeguards, retention period, and plans for reporting aggregate findings.

Do not include patient records in the initial email or in a public GitHub issue. If access is approved, the research team and relevant institutions will specify the authorized materials, secure transfer or analysis environment, named recipients, and retention and deletion requirements in a written agreement.

## Restrictions on controlled materials

Permission to access a model does not confer permission to redistribute it. Controlled access must address restrictions on redistribution of trained parameters, empirical reference arrays, patient-level outputs, and query collections that could support model extraction or distillation. Re-identification attempts and linkage to identify individuals are prohibited. Requests for a different use or onward transfer require separate authorization.

Institutional approval is required before sharing clinical data or patient-derived artifacts. Availability depends on the requested use and applicable approvals; the public repository does not guarantee access. Public source code does not include the fitted values needed to reconstruct the study model.
