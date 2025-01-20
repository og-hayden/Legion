# Contributing to Legion

Welcome! We're excited you're interested in contributing to Legion. This document outlines the process for contributing and helps you get started.

## Quick Links
- [Discord Server](https://discord.gg/9QnHYaAD)
- [GitHub Project Board](https://github.com/orgs/LLMP-io/projects/1)
- [Issues](https://github.com/LLMP-io/Legion/issues)

## Getting Started

1. **Fork the Repository**
   - Click the "Fork" button in the top right of the GitHub repository
   - Clone your fork locally: `git clone https://github.com/YOUR-USERNAME/Legion.git`
   - Add upstream remote: `git remote add upstream https://github.com/LLMP-io/Legion.git`

2. **Set Up Development Environment**
   Three options here.
   - **Option 1**: Using `Makefile`
      Optioned and automated
      - Requires you have `make` and `poetry` installed
      - `ENV` management options are currently `venv` or `conda`
      - `POETRY` can be set to `true` or `false` to use it, or `pip`

      2a. [Install Make](https://www.gnu.org/software/make/manual/make.html)

      2b. Install Poetry
      ```bash
      curl -sSL https://install.python-poetry.org | python3 -
      ```

      2c. Set up the environment

      ```bash
      make setup ENV=venv POETRY=false
      # or
      make setup ENV=conda POETRY=true
      # or
      make  # Just use the defaults
      ```
      This will:
      - Create and activate a virtual environment (with specified `ENV`)
      - Install all dependencies (with `pip` or `poetry`)
      - Set up pre-commit hooks

      You can also:
      ```bash
      make lint POETRY=<true|false>
      or
      make test POETRY=<true|false>
      ```

   - **Option 2**: Using `setup.py`
      Default and standard `venv`/`pip` setup
      ```bash
      # Run the setup script
      python3 scripts/setup_env.py

      # Activate the virtual environment
      # On macOS/Linux:
      source venv/bin/activate
      # On Windows:
      # venv\Scripts\activate
      ```
      This will:
      - Create and activate a virtual environment
      - Install all dependencies
      - Set up pre-commit hooks

   - **Option 3**: 🐳 Whale you can just use `Docker`
      If you have Docker installed and the docker engine running, from the
      project root you can just run:
      ```bash
      docker compose up --build
      ```
      This will spin up a cointainer, build the project to it, and run the tests.
      Want to keep it running and get a shell on it to run other commands?
      ```bash
      docker compose up -d  # detached
      docker compose exec legion bash
      ```

3. **Configure Git Email**
   To maintain privacy while contributing, you can use a GitHub-provided no-reply email address:
   ```bash
   # Replace 'username' with your GitHub username
   # Get your GitHub user ID from: https://api.github.com/users/username
   git config user.email "ID+username@users.noreply.github.com"
   ```

4. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```


## Development Guidelines

### Code Style
We follow PEP 8 guidelines for Python code style. Key points:
- Use 4 spaces for indentation
- Maximum line length of 79 characters for code, 72 for docstrings
- Use meaningful variable names
- Add docstrings to all functions, classes, and modules
- Use comprehensive type hints

### Type Checking
We use mypy for static type checking. Key requirements:
- All functions must have complete type hints
- All class attributes must be typed
- Use `Optional[T]` instead of `T | None`
- Use `Sequence[T]` for read-only lists
- Use `Mapping[K, V]` for read-only dicts
- Avoid `Any` unless absolutely necessary
- Run type checks before submitting PRs

### Before Submitting
1. Set up pre-commit hooks:
   ```bash
   # Install pre-commit
   pip install pre-commit

   # Install the git hooks
   pre-commit install
   ```

2. The pre-commit hooks will automatically run:
   - Non-integration tests
   - Style checking with ruff
   - Security checks with bandit and safety

   You can also run these checks manually:

3. Run tests:
```bash
# Run all tests except integration tests
pytest -v -m "not integration"

# Run only integration tests
pytest -v -m integration

# Run all tests
pytest -v
```

4. Run type checking (optional, but recommended):
```bash
# Run mypy type checker
python scripts/typecheck.py

# Check specific modules
python scripts/typecheck.py legion/agents legion/blocks
```

5. Run code style checks:
```bash
# Run ruff linter
python scripts/lint.py
```

6. Run security checks:
```bash
# Run security scans
python scripts/security.py

# Check specific modules
python scripts/security.py legion/agents legion/blocks
```

7. Update documentation if needed
8. Add tests for new features

## Pull Request Process

1. **Create Pull Request**
   - Push your changes to your fork
   - Create a PR against the `main` branch
   - Fill out the PR template
   - Link any related issues

2. **PR Guidelines**
   - Keep changes focused and atomic
   - Provide clear description of changes
   - Include any necessary documentation updates
   - Add screenshots for UI changes
   - Reference any related issues

3. **Review Process**
   - Maintainers will review your PR
   - Address any requested changes
   - Once approved, maintainers will merge your PR

### **Collaboration Patterns**

1. **Direct to Upstream** (Recommended)
   - Each contributor maintains their fork
   - Create PRs directly to upstream
   - Sync fork:
     ```bash
     git fetch upstream
     git rebase upstream/main
     ```

2. **Fork Collaboration**
   - Add collaborators to your fork's settings
   - Both can push branches
   - Create single PR to upstream
   - Maintain clear ownership of the PR

3. **Cross-fork PRs**
   - For non-collaborator contributions
   - Create PR between forks
   - Owner submits final PR upstream

Note: Contributions are tracked through PRs, not individual commits due to squash merging. All contributors are credited in the PR history.


## Issue Guidelines

### Creating Issues
- Check existing issues to avoid duplicates
- Use issue templates when available
- Provide clear reproduction steps for bugs
- For feature requests, explain the use case

### Good First Issues
Look for issues labeled `good-first-issue` if you're new to the project. These are typically:
- Documentation improvements
- Small bug fixes
- Test additions
- Simple feature implementations

## Getting Help

- [Join our Discord server](https://discord.gg/B63rdfQ6) for real-time discussion
- Use the #help channel for technical questions
- Tag maintainers in GitHub issues if you need clarification

## Discord Channels

- #announcements: Project updates and news
- #general: Introduce yourself and ask questions
- #help: Get help with technical issues
- #development: Discuss ongoing development
- #pull-requests: PR discussions and reviews
- #give-your-input: Share and discuss feature ideas
- #general-ai-discussion: Talk all things AI 🤖

## Recognition

We value all contributions, big and small! Contributors will be:
- Added to our Contributors list
- Recognized in release notes for significant contributions
- Badges applied to your profile within the Legion Discord Server
- Given credit in documentation when appropriate

Thank you for contributing to Legion! 🚀
