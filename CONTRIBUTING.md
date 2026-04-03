# Contributing

Thanks for your interest in improving HI-MoE.

## Ways to contribute

You can help by:

- reporting bugs
- improving documentation
- adding tests
- improving training scripts
- contributing visualization and analysis tools
- cleaning up configs and reproducibility utilities

## Before you start

Please open an issue for:

- major feature additions
- large refactors
- changes to experiment protocols
- benchmark result updates

This helps keep research code and paper claims aligned.

## Development guidelines

### General

- Keep changes focused and small.
- Prefer readable code over clever code.
- Do not change multiple experiment assumptions in one pull request.
- Document any change that affects reported results.

### Code style

- Follow PEP 8 where practical.
- Use clear names for configs, modules, and experiment scripts.
- Add comments only where they genuinely clarify non-obvious logic.

### Configs

When adding a new config:

- start from an existing template
- use descriptive names
- document the change relative to the baseline
- keep generated configs separate from handwritten templates

### Experiments

For changes affecting model behavior or results, include:

- exact config used
- dataset split
- hardware used
- random seed if applicable
- key metric changes

## Pull requests

A good pull request should include:

- a short summary of the change
- motivation
- files changed
- any effect on training or evaluation
- screenshots or plots for visualization changes, if relevant

## Reporting issues

Please include:

- environment details
- install method
- full error message
- command used
- minimal reproduction steps

## Research integrity

If you add or update benchmark numbers:

- clearly separate pilot results from final results
- avoid mixing datasets or training recipes without stating it
- do not overwrite reported numbers without explanation

## Contact

For paper-related questions, use the repository issue tracker unless private contact is necessary.
