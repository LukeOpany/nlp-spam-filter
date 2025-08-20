# Contributing to NLP Spam Filter

We love your input! We want to make contributing to the NLP Spam Filter project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. **Fork the repo** and create your branch from `master`
2. **Make your changes** following our code style guidelines
3. **Test your changes** if you've added code that should be tested
4. **Update documentation** if you've changed APIs or functionality
5. **Ensure your code follows** the existing style conventions
6. **Issue that pull request!**

## Code Style Standards

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Use type hints where applicable

### Jupyter Notebook Guidelines
- **Clear markdown cells**: Add explanatory markdown cells before code sections
- **Meaningful comments**: Comment complex algorithms and data transformations
- **Output cleanup**: Clear unnecessary output before committing
- **Cell organization**: Group related operations in logical sections
- **Variable naming**: Use descriptive names for datasets and models

### Documentation Style
- Use clear, concise language
- Include code examples where helpful
- Keep README and docs up to date
- Use proper markdown formatting

## Development Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git for version control

### Local Development Environment

1. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/nlp-spam-filter.git
   cd nlp-spam-filter
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

5. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## Testing Guidelines

### Manual Testing
Since this is primarily a Jupyter notebook project, testing involves:

1. **Data Loading**: Verify the dataset loads correctly
2. **Preprocessing**: Check text preprocessing functions work with sample data
3. **Model Training**: Ensure the pipeline trains without errors
4. **Predictions**: Test predictions on sample messages
5. **Performance**: Verify reported accuracy metrics

### Test Cases to Verify
- [ ] Dataset loads with correct dimensions (5,572 messages)
- [ ] Text preprocessing removes punctuation and stopwords correctly
- [ ] Pipeline trains successfully on training data
- [ ] Model achieves expected accuracy (>90%)
- [ ] Predictions work on new sample messages
- [ ] All imports work correctly

### Adding New Features
When adding new features, please ensure:
- [ ] Code is well-documented with docstrings and comments
- [ ] New dependencies are added to `requirements.txt`
- [ ] Documentation is updated to reflect new functionality
- [ ] Examples are provided for new features

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/LukeOpany/nlp-spam-filter/issues).

### Great Bug Reports Include:
- **Summary**: Quick summary of the bug
- **Environment**: OS, Python version, dependency versions
- **Steps to reproduce**: Specific steps to reproduce the behavior
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error traceback if applicable
- **Screenshots**: If applicable, add screenshots

### Bug Report Template
```markdown
**Environment:**
- OS: [e.g., Windows 10, macOS 12.0, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Jupyter version: [e.g., 6.4.0]

**Describe the bug:**
A clear description of what the bug is.

**To Reproduce:**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior:**
A clear description of what you expected to happen.

**Error message:**
```
[Full error traceback here]
```

**Additional context:**
Add any other context about the problem here.
```

## Feature Requests

We also use GitHub issues to track feature requests. When suggesting new features:

1. **Check existing issues** to avoid duplicates
2. **Provide clear use case** - explain why this feature would be useful
3. **Consider implementation** - think about how it might work
4. **Be specific** - detailed requirements help us understand your needs

## Questions and Discussions

For questions about the project:
- Check the [README](README.md) first
- Look through existing [issues](https://github.com/LukeOpany/nlp-spam-filter/issues)
- Open a new issue with the "question" label

## Code of Conduct

### Our Standards
- **Be respectful**: Treat all contributors with respect
- **Be collaborative**: Help others learn and grow
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone was a beginner once

### Unacceptable Behavior
- Harassment or discrimination of any kind
- Offensive comments or personal attacks
- Trolling or deliberately disruptive behavior
- Publishing private information without permission

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

## Recognition

Contributors will be recognized in the project documentation. We appreciate all forms of contribution, from code to documentation to bug reports!

---

Thank you for contributing to the NLP Spam Filter project! 🎉