# HW03 - wildcatshawkeyes

## Description

This package contains implementations for CPE 487/587 Machine Learning Tools Homework 03.

## Installation

### From PyPI
```bash
pip install hw03wildcatshawkeyes
```

### From GitHub Release
```bash
pip install https://github.com/wildcatshawkeyesgh/hw03wildcatshawkeyes/releases/download/v0.1.0/hw03wildcatshawkeyes-0.1.0-*.whl
```

### From Source
```bash
git clone https://github.com/wildcatshawkeyesgh/hw03wildcatshawkeyes
cd hw03wildcatshawkeyes
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
pip install dist/*.whl
```

## Usage

```python
from hw03wildcatshawkeyes import example_function

result = example_function([1.0, 2.0, 3.0])
print(result)
```

## Package Structure

```
hw03wildcatshawkeyes/
├── src/
│   └── hw03wildcatshawkeyes/
│       ├── __init__.py
│       └── “deepl/
│           ├── __init__.py
│           ├── multiclass.py.py
│           ├── two_layer_binary_classification.py
└── scripts/
    └── acc_classifier.py”.py
```

## Modules

### “deepl
- `multiclass.py`: Add description here
- `two_layer_binary_classification`: Add description here


## Scripts

- `acc_classifier.py”.py`: Add description here

## Dependencies

- Python >= 3.12
- PyTorch
- NumPy
- Matplotlib

## Author

- **Keyword**: wildcatshawkeyes
- **Email**: wildcatshawkeyes@gmail.com
- **GitHub**: wildcatshawkeyesgh
- **Course**: CPE 487/587 Machine Learning Tools
- **Institution**: University of Alabama in Huntsville

## License

MIT License
