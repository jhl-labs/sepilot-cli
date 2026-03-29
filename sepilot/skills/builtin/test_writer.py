"""Test Writer Skill"""

from ..base import PromptSkill


class TestWriterSkill(PromptSkill):
    name = "test-writer"
    description = "Generate comprehensive tests with best practices"
    triggers = [
        "write test", "add test", "create test", "unit test",
        "test case", "test code", "pytest", "unittest",
        "need test", "missing test", "test coverage",
    ]
    category = "testing"
    prompt = """\
## Test Writing Guidelines

When writing tests, follow these best practices:

### 1. Test Structure (AAA Pattern)
```
def test_something():
    # Arrange - Set up test data and conditions
    input_data = prepare_test_data()

    # Act - Execute the code being tested
    result = function_under_test(input_data)

    # Assert - Verify the results
    assert result == expected_value
```

### 2. Naming Conventions
- Use descriptive names: `test_<function>_<scenario>_<expected_result>`
- Examples:
  - `test_login_with_valid_credentials_returns_token`
  - `test_divide_by_zero_raises_exception`
  - `test_empty_list_returns_none`

### 3. Test Coverage Checklist
- [ ] Happy path (normal operation)
- [ ] Edge cases (empty, null, boundary values)
- [ ] Error cases (invalid input, exceptions)
- [ ] Boundary conditions (min/max values)
- [ ] Type variations (if applicable)

### 4. Best Practices
- One assertion concept per test (can have multiple asserts for same concept)
- Tests should be independent and isolated
- Use fixtures for common setup
- Mock external dependencies (API, DB, filesystem)
- Test behavior, not implementation

### 5. Framework-Specific Tips

**pytest:**
```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"

@pytest.mark.parametrize("input,expected", [
    (1, 2), (2, 4), (3, 6)
])
def test_double(input, expected):
    assert input * 2 == expected
```

**unittest:**
```python
import unittest

class TestMyFunction(unittest.TestCase):
    def setUp(self):
        self.data = prepare_data()

    def test_normal_case(self):
        self.assertEqual(my_function(1), 2)

    def test_raises_error(self):
        with self.assertRaises(ValueError):
            my_function(-1)
```

### 6. What to Test
- Public API/interfaces
- Critical business logic
- Complex calculations
- Error handling paths
- Integration points

Generate comprehensive, maintainable tests that catch real bugs."""
