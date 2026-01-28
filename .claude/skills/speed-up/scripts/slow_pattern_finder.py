#!/usr/bin/env python3
"""
Slow Pattern Finder

Scan Python files for known slow code patterns.

Usage:
    python slow_pattern_finder.py path/to/project/
    python slow_pattern_finder.py path/to/file.py
    python slow_pattern_finder.py . --pattern loops
"""

import argparse
import ast
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SlowPattern:
    """A detected slow pattern."""
    file: str
    line_start: int
    line_end: int
    pattern_type: str
    description: str
    code_snippet: str
    suggestion: str
    speedup: str


class PatternDetector(ast.NodeVisitor):
    """AST visitor to detect slow patterns."""

    def __init__(self, source_lines: list[str], filename: str):
        self.source_lines = source_lines
        self.filename = filename
        self.patterns: list[SlowPattern] = []

    def get_source(self, node) -> str:
        """Extract source code for a node."""
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno)
        return '\n'.join(self.source_lines[start:end])

    def visit_For(self, node):
        """Detect slow for loop patterns."""
        # Check for append in loop
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if child.func.attr == 'append':
                        self.patterns.append(SlowPattern(
                            file=self.filename,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            pattern_type="append_in_loop",
                            description="List.append() inside loop - consider list comprehension",
                            code_snippet=self.get_source(node),
                            suggestion="Use list comprehension: [expr for x in iterable]",
                            speedup="1.5-2x"
                        ))
                        break

        # Check for string concatenation in loop
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign):
                if isinstance(child.op, ast.Add):
                    if isinstance(child.target, ast.Name):
                        # Could be string concat
                        self.patterns.append(SlowPattern(
                            file=self.filename,
                            line_start=node.lineno,
                            line_end=getattr(node, 'end_lineno', node.lineno),
                            pattern_type="string_concat_loop",
                            description="Possible string concatenation in loop",
                            code_snippet=self.get_source(node),
                            suggestion="Use ''.join(list) instead of += in loop",
                            speedup="10-100x for many iterations"
                        ))
                        break

        self.generic_visit(node)

    def visit_Compare(self, node):
        """Detect slow membership tests."""
        # Check for 'x in list_var' patterns
        if any(isinstance(op, ast.In) for op in node.ops):
            for comparator in node.comparators:
                if isinstance(comparator, ast.List):
                    self.patterns.append(SlowPattern(
                        file=self.filename,
                        line_start=node.lineno,
                        line_end=node.lineno,
                        pattern_type="in_list_literal",
                        description="Membership test on list literal",
                        code_snippet=self.get_source(node),
                        suggestion="Use set literal: x in {a, b, c} instead of x in [a, b, c]",
                        speedup="O(n) -> O(1) for membership test"
                    ))

        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect slow function call patterns."""
        if isinstance(node.func, ast.Attribute):
            # Check for .keys() in membership test (parent is Compare with In)
            if node.func.attr == 'keys':
                self.patterns.append(SlowPattern(
                    file=self.filename,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    pattern_type="dict_keys_membership",
                    description="Using .keys() for membership test",
                    code_snippet=self.get_source(node),
                    suggestion="Use 'key in dict' instead of 'key in dict.keys()'",
                    speedup="Minor but cleaner"
                ))

            # Check for repeated len() calls that could be cached
            if node.func.attr == 'tolist':
                self.patterns.append(SlowPattern(
                    file=self.filename,
                    line_start=node.lineno,
                    line_end=node.lineno,
                    pattern_type="unnecessary_tolist",
                    description="Potentially unnecessary .tolist() conversion",
                    code_snippet=self.get_source(node),
                    suggestion="Check if tolist() is needed - numpy operations often work directly",
                    speedup="Varies - avoid if not needed"
                ))

        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Check for nested comprehensions that could be flattened."""
        # Count nesting depth
        depth = 0
        for generator in node.generators:
            depth += 1
            if hasattr(generator, 'ifs'):
                depth += len(generator.ifs)

        if depth > 2:
            self.patterns.append(SlowPattern(
                file=self.filename,
                line_start=node.lineno,
                line_end=getattr(node, 'end_lineno', node.lineno),
                pattern_type="deep_comprehension",
                description="Deeply nested list comprehension",
                code_snippet=self.get_source(node),
                suggestion="Consider breaking into multiple steps or using itertools",
                speedup="Readability and potential memory improvement"
            ))

        self.generic_visit(node)


def find_regex_patterns(content: str, filename: str) -> list[SlowPattern]:
    """Find patterns using regex (for things AST can't catch)."""
    patterns = []
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        # Check for range(len(...))
        if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', line):
            patterns.append(SlowPattern(
                file=filename,
                line_start=i,
                line_end=i,
                pattern_type="range_len",
                description="Using range(len(x)) instead of enumerate",
                code_snippet=line.strip(),
                suggestion="Use 'for i, item in enumerate(x)' instead",
                speedup="Minor but more Pythonic"
            ))

        # Check for manual index incrementing
        if re.search(r'\w+\s*\+=\s*1\s*$', line) and i > 1:
            prev_line = lines[i-2] if i > 1 else ""
            if 'for' in prev_line or 'while' in prev_line:
                patterns.append(SlowPattern(
                    file=filename,
                    line_start=i,
                    line_end=i,
                    pattern_type="manual_counter",
                    description="Manual counter increment in loop",
                    code_snippet=line.strip(),
                    suggestion="Use enumerate() or range() instead of manual counter",
                    speedup="Minor but cleaner"
                ))

        # Check for numpy array creation in apparent loop context
        if re.search(r'np\.(array|zeros|ones|empty)\s*\(', line):
            # Check if we're inside a loop (rough heuristic)
            for j in range(max(0, i-5), i):
                if re.search(r'^\s*(for|while)\s+', lines[j]):
                    patterns.append(SlowPattern(
                        file=filename,
                        line_start=i,
                        line_end=i,
                        pattern_type="array_creation_in_loop",
                        description="NumPy array creation possibly inside loop",
                        code_snippet=line.strip(),
                        suggestion="Pre-allocate array outside loop if possible",
                        speedup="2-10x depending on array size"
                    ))
                    break

    return patterns


def analyze_file(filepath: Path) -> list[SlowPattern]:
    """Analyze a single Python file for slow patterns."""
    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')

        # AST-based detection
        try:
            tree = ast.parse(content)
            detector = PatternDetector(lines, str(filepath))
            detector.visit(tree)
            patterns = detector.patterns
        except SyntaxError:
            patterns = []

        # Regex-based detection
        patterns.extend(find_regex_patterns(content, str(filepath)))

        return patterns

    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return []


def analyze_directory(dirpath: Path, exclude_patterns: list[str] = None) -> list[SlowPattern]:
    """Analyze all Python files in a directory."""
    exclude_patterns = exclude_patterns or ['venv', '__pycache__', '.git', 'node_modules']
    all_patterns = []

    for filepath in dirpath.rglob('*.py'):
        # Skip excluded directories
        if any(excl in filepath.parts for excl in exclude_patterns):
            continue

        all_patterns.extend(analyze_file(filepath))

    return all_patterns


def format_output(patterns: list[SlowPattern], verbose: bool = False) -> str:
    """Format patterns for display."""
    if not patterns:
        return "No slow patterns detected."

    # Group by type
    by_type: dict[str, list[SlowPattern]] = {}
    for p in patterns:
        by_type.setdefault(p.pattern_type, []).append(p)

    output = []
    output.append(f"Found {len(patterns)} potential slow patterns:\n")

    for pattern_type, items in sorted(by_type.items()):
        output.append(f"\n## {pattern_type.replace('_', ' ').title()} ({len(items)} found)")
        output.append("-" * 60)

        for p in items:
            output.append(f"\n**{p.file}:{p.line_start}**")
            output.append(f"  {p.description}")
            if verbose:
                output.append(f"  Code: {p.code_snippet[:80]}...")
            output.append(f"  Suggestion: {p.suggestion}")
            output.append(f"  Expected speedup: {p.speedup}")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description="Find slow code patterns")
    parser.add_argument("path", help="File or directory to analyze")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show code snippets")
    parser.add_argument("--pattern", help="Filter by pattern type")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        patterns = analyze_file(path)
    elif path.is_dir():
        patterns = analyze_directory(path)
    else:
        print(f"Error: {path} not found")
        return 1

    if args.pattern:
        patterns = [p for p in patterns if args.pattern.lower() in p.pattern_type.lower()]

    if args.json:
        import json
        data = [vars(p) for p in patterns]
        print(json.dumps(data, indent=2))
    else:
        print(format_output(patterns, args.verbose))

    return 0


if __name__ == "__main__":
    exit(main())
