# ICI Myocardial Injury

Research source code for dynamic assessment before the first recorded troponin I elevation during immune checkpoint inhibitor therapy. The clinical question is whether earlier reassessment can identify patients requiring further evaluation within a limited repeat-testing budget.

This repository contains reusable analysis functions and tests constructed from artificial inputs. It does not distribute the study dataset, the fitted study model, or a service that returns predictions from that model.

## Code scope

- Construction of assessment times using information available before the scoring stop time.
- Exponential survival modeling and percentile-based combination with the latest troponin I measurement.
- Patient-level alert evaluation and comparison within an initial repeat-testing budget.
- Counting proposed repeat tests while distinguishing verified specimen times from report-time proxies.

The functions require inputs supplied by an authorized investigator. Executing the tests checks implementation behavior; it does not reproduce the study findings or establish performance in a new clinical population. The published model requires its separately controlled fitted parameters and empirical reference distributions.

## Installation and checks

Use Python 3.11 or later in a dedicated environment:

```sh
python -m pip install .
python -m unittest discover -s tests
python tools/check_public_release.py
```

Input definitions and function documentation are provided in the source modules. Test inputs are small, hand-constructed examples, independent of patients and of outputs from the trained study model.

## Data and trained model access

Clinical records and patient-level derived data are not publicly distributed. The trained model, fitted preprocessing values, empirical reference arrays, and patient-level scores are also excluded from this repository.

Investigators may request controlled access from Kunyu Yang at **yangkunyu@hust.edu.cn**. Requests are reviewed for their scientific purpose, institutional authorization, data protection arrangements, and applicable ethics and use agreements. Access is not automatic. The research ethics approvals do not themselves authorize public redistribution of clinical records.

See [ACCESS.md](ACCESS.md) for the request process and [SECURITY.md](SECURITY.md) for reporting security concerns. Do not submit patient information or model files in public issues, pull requests, or attachments.

## Use and interpretation

This is research software. A risk score does not establish the cause of myocardial injury or determine treatment. The code alone cannot provide the predictions of the fitted study model. Any new application requires appropriate clinical oversight and evaluation in its intended setting.

The repository provides public access to source code. No open-source license is granted with this release; requests for reuse and redistribution should be directed to the corresponding author. Access to trained artifacts is governed separately.
