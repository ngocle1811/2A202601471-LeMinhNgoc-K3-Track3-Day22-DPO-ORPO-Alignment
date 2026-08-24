# Data Card: Sample Preference Alignment Pairs

- **Dataset name**: Preference Alignment Lab sample dataset
- **Source**: educational examples included in this repository; original external provenance is not documented.
- **License/permission**: not specified in the starter repository; verify before redistribution or production use.
- **Size**: 24 preference pairs.
- **Schema**:
  - `prompt` (`str`): non-empty instruction or question.
  - `chosen` (`str`): preferred response.
  - `rejected` (`str`): meaningfully different, lower-quality response.
  - `metadata` (`dict`): currently contains `domain` and `rubric`.
- **Labeling rubric**: technical accuracy for introductory machine-learning education questions.
- **Known biases**: single education/ML domain; English only; preferred responses are generally longer and more detailed, which can create a length shortcut; no inter-annotator agreement is available.
- **Safety/PII checks**: loader can reject basic email and phone patterns with `reject_pii=True`. The sample loads with that guard enabled, but regex checks are not a complete privacy review.
- **Train/validation/test split method**: prompt-grouped deterministic split using seed 42. Current local evaluation uses 19 train and 5 validation examples; no independent test partition is defined.
- **Intended use**: teaching data validation, DPO/ORPO objectives, deterministic evaluation, and testing practices.
- **Out-of-scope use**: production model training, safety certification, or claims of broad model quality.
