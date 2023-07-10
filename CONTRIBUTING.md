# Contributing to ChiRho

## Development

Please follow our established coding style including variable names, module imports, and function definitions. The ChiRho codebase follows the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/) (which you can check with `make lint`) and follows [`isort`](https://github.com/timothycrosley/isort) import order (which you can enforce with `make format`). 

## Dev Setup
To install dev dependencies for ChiRho, run the following command.
```sh
pip install -e .[test]
```

## Testing

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally
```sh
make lint              # linting
make format            # runs black and isort
make tests             # linting and unit tests
```

## Submitting

For relevant design questions to consider, see past [design documents](https://github.com/pyro-ppl/pyro/wiki/Design-Docs).

For larger changes, please open an issue for discussion before submitting a pull request.

In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies

For speculative changes meant for early-stage review, include `[WIP]` in the PR's title. (One of the maintainers will add the `WIP` tag.)

## Code of conduct
This project follows [GitHub community guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines) and [Pyro code of conduct](https://github.com/pyro-ppl/pyro/blob/dev/CODE_OF_CONDUCT.md).