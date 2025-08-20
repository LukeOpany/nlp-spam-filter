# Contributing to NLP Spam Filter

Thank you for your interest in contributing to the NLP Spam Filter project! We welcome contributions from the community and appreciate your efforts to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style Standards](#code-style-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nlp-spam-filter.git
   cd nlp-spam-filter
   ```
3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/LukeOpany/nlp-spam-filter.git
   ```

## Development Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Jupyter Notebook
- Git

### Environment Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies:
   ```bash
   pip install jupyter notebook ipykernel
   ```

4. Download required NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Contributing Guidelines

### What We're Looking For

- Bug fixes
- Performance improvements
- Documentation improvements
- New features that enhance spam detection
- Test coverage improvements
- Code quality enhancements

### What We're Not Looking For

- Breaking changes without discussion
- Features that significantly increase complexity
- Changes that reduce model performance
- Modifications that break existing functionality

## Code Style Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions small and focused
- Use type hints where appropriate
- Maximum line length: 88 characters (Black formatter standard)

### Jupyter Notebook Guidelines

- Use clear, descriptive cell headers
- Add markdown cells to explain complex analysis steps
- Include inline comments for complex code
- Clear output before committing notebooks
- Use consistent variable naming throughout

### Documentation Standards

- Add docstrings to all functions using Google style:
  ```python
  def text_process(mess):
      """
      Process text message for spam classification.
      
      Args:
          mess (str): Input text message to process
          
      Returns:
          list: List of processed tokens
          
      Example:
          >>> text_process("Hello World!")
          ['hello', 'world']
      """
  ```

- Update README.md for significant changes
- Add inline comments for complex algorithms
- Keep documentation up to date with code changes

## Testing

### Running Tests

Currently, the project uses manual testing through the Jupyter notebook. For new contributions:

1. Test your changes thoroughly in the notebook
2. Verify model performance doesn't degrade
3. Test with various input types and edge cases
4. Document any new dependencies or setup requirements

### Test Requirements

- Ensure model accuracy remains above 95%
- Test with different message types and lengths
- Verify text preprocessing works correctly
- Check that visualizations display properly

## Submitting Changes

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
type(scope): Brief description

Detailed explanation if needed

- Bullet points for multiple changes
- Reference issues with #issue-number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): Add support for custom stopwords list

docs(readme): Update installation instructions

fix(preprocessing): Handle empty messages correctly
```

### Branch Naming

Use descriptive branch names:
- `feature/description-of-feature`
- `bugfix/description-of-bug`
- `docs/description-of-docs-change`

## Issue Guidelines

### Before Creating an Issue

1. Search existing issues to avoid duplicates
2. Check if the issue is already fixed in the latest version
3. Gather relevant information about your environment

### Issue Templates

**Bug Report:**
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant code snippets or error messages

**Feature Request:**
- Clear description of the proposed feature
- Use case and benefits
- Potential implementation approach
- Compatibility considerations

**Documentation Issue:**
- Section that needs improvement
- Specific improvements needed
- Target audience (beginners, advanced users, etc.)

## Pull Request Process

### Before Submitting

1. Ensure your branch is up to date with main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Test your changes thoroughly
3. Update documentation if needed
4. Add appropriate commit messages

### Pull Request Requirements

- **Clear title and description**: Explain what changes you made and why
- **Reference related issues**: Use "Fixes #123" or "Closes #123"
- **Small, focused changes**: Large PRs are harder to review
- **Working code**: Ensure your changes don't break existing functionality
- **Documentation updates**: Update docs for user-facing changes

### Review Process

1. Automated checks (if applicable)
2. Code review by maintainers
3. Testing and validation
4. Approval and merge

### After Your PR is Merged

1. Delete your feature branch
2. Update your local main branch:
   ```bash
   git checkout main
   git pull upstream main
   ```

## Development Workflow

### Typical Workflow

1. Create a new branch for your feature/fix
2. Make your changes
3. Test thoroughly
4. Commit with clear messages
5. Push to your fork
6. Create a pull request
7. Address review feedback
8. Merge when approved

### Working with Jupyter Notebooks

- Clear all outputs before committing
- Use version control friendly practices
- Test notebooks thoroughly before submitting
- Consider creating Python scripts for complex functions

## Getting Help

### Resources

- Project documentation in `/docs` folder
- README.md for project overview
- Existing issues and PRs for context
- Python and scikit-learn documentation

### Communication

- GitHub Issues for bugs and feature requests
- Pull Request comments for code-specific discussions
- Project discussions for general questions

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation where appropriate

Thank you for contributing to NLP Spam Filter!