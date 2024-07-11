from __future__ import annotations

import ast
import json
import re

import h5py
import scipy
import numpy as np
from sympy import Symbol
from pathlib import Path
from textwrap import dedent
from typing import Any

OrderedContent = list[tuple[str, str]]

H5PY_FILE = "evaluation/test_data.h5"


def extract_ipynb_content_ordered(file_path):
    """First we assume the input is an ipynb file. We want to extract the content of the ipynb file in the order of the cells."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        notebook = json.load(file)
    ordered_content = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ordered_content.append(("code", "".join(cell["source"])))
        elif cell["cell_type"] == "markdown":
            ordered_content.append(("markdown", "".join(cell["source"])))
        else:
            raise ValueError(f"Unknown cell type: {cell['cell_type']}")
    return ordered_content


def _extract_function_headers_and_docs(
        node: ast.FunctionDef,
        *,
        include_default_args: bool = True,
        method_indent: bool = False,
) -> str | None:
    """Extract function headers and their docstrings from the given Python code.

    Args:
        node (ast.FunctionDef): The function node from the AST.
        include_default_args (bool): Flag to include default arguments in the function header.
        method_indent (bool): We're extracting a method, so we need to indent the method header.
    """
    indent = "    " if method_indent else ""
    func_header = f"{indent}def {node.name}("
    if include_default_args:
        args = [arg.arg for arg in node.args.args]
        defaults = [ast.unparse(d) for d in node.args.defaults]
        defaults = [None] * (len(args) - len(defaults)) + defaults
        params = [
            f"{arg}={default}" if default is not None else arg
            for arg, default in zip(args, defaults)
        ]
        func_header += ", ".join(params) + "):"
    else:
        func_header += ", ".join(arg.arg for arg in node.args.args) + "):"
    docstring = ast.get_docstring(node)
    if docstring:
        docstring_formatted = (
                f"\n{indent}    '''"
                + docstring.replace("\n", f"\n{indent}    ")
                + f"\n{indent}    '''"
        )
    else:
        docstring_formatted = f"\n{indent}    '''\n{indent}    '''"
    func_combined = f"{func_header}{docstring_formatted}"
    return func_combined


def extract_function_headers_and_docs(
        code: str, *, include_default_args: bool = True
) -> str | None:
    """Extract function headers and their docstrings from the given Python code."""
    # Dedent the code to avoid indentation errors
    code = dedent(code)
    # Parse the function into an AST
    parsed_code = ast.parse(code)

    for node in ast.walk(parsed_code):
        if isinstance(node, ast.FunctionDef):
            return _extract_function_headers_and_docs(
                node, include_default_args=include_default_args
            )
        elif isinstance(node, ast.ClassDef):
            return extract_class_templates_and_docs(
                code, include_default_args=include_default_args
            )


# we need to extract all numbered md cells, e.g. "1.1", "1.2", "1.3" etc.
def extract_problem_steps_cells(
        ordered_content: OrderedContent, *, include_background: bool
) -> list[str]:
    """
    Extract the solution cells from the ordered_content.
    include_background: flag to include the background section in the problem steps.
    """
    problem_steps = []
    for content_type, content in ordered_content:
        if content_type == "markdown":
            # if there is a background section, we need to include it in the problem steps
            if (
                    "### background" in content.lower()
                    or "#### background" in content.lower()
            ):
                if include_background:
                    # Match from the "## 1.1" heading through the background section
                    regex = r"(## 1\.\d+[\s\S]+?(?=(## \d+|$)))"
                else:
                    # Match from the "## 1.1" heading up to just before "### Background" or "#### Background"
                    regex = r"(## 1\.\d+[\s\S]+?)(?=(###|####) Background)"
            else:
                # Match from the "## 1.1" heading to the end of the cell
                regex = r"(## 1\.\d+[\s\S]+)"

            match = re.search(regex, content, flags=re.DOTALL)
            if match:
                res = match.group(0).strip()
                problem_steps.append(res)

    return problem_steps


def extract_return_line_of_function(code: str) -> str | None:
    """Extract function return from the given Python code."""
    for line in code.splitlines()[::-1]:
        if "return" in line:
            return line


def extract_problem(
        ordered_content: OrderedContent, include_background: bool = False
) -> tuple[str, str, str]:
    """Assume the first three cells are the "problem_description", "input&output" and "dependencies" cells"""
    problem_description = ordered_content[0][1]
    if not include_background:
        if "### background" in problem_description.lower():
            regex = r"(## 1\.+[\s\S]+?)(?=(###|####) Background)"
            match = re.search(regex, problem_description, flags=re.DOTALL)
            if match:
                problem_description = match.group(0).strip()
    input_output = ordered_content[1][1]
    dependencies = ordered_content[2][1]
    return problem_description, input_output, dependencies


def extract_problem_code_single_step(ordered_content: OrderedContent) -> str:
    """Extract the code from ordered_content by searching keyword "code" in the code cells and extract the entire cell."""
    for content_type, content in ordered_content:
        if content_type == "code" and re.search(r"# code", content.lower()):
            return content
    raise ValueError("Code cell not found.")


def extract_problem_steps_code(ordered_content: OrderedContent) -> list[str]:
    """Extract the test code from ordered_content by searching keyword "code" in the code cells and extract the entire cell."""
    problem_steps_code = []
    for content_type, content in ordered_content:
        if content_type == "code" and re.search(r"# code 1\.\d+", content):
            problem_steps_code.append(content)
    return problem_steps_code


def extract_class_templates_and_docs(
        code: str, include_default_args: bool = True
) -> str:
    """
    similar to extract_function_headers_and_docs, but for classes.
    We will extract class headers and EACH method headers and their docstrings from the given Python code.
    """
    code = dedent(code)
    parsed_code = ast.parse(code)

    class_header = ""
    method_headers = []
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.ClassDef):
            class_header = f"class {node.name}:"
        if isinstance(node, ast.FunctionDef):
            method_headers.append(
                _extract_function_headers_and_docs(
                    node, include_default_args=include_default_args, method_indent=True
                )
            )

    # combine class header and method headers
    class_combined = f"{class_header}"
    for method in method_headers:
        class_combined += f"\n{method}"

    return class_combined


def extract_problem_steps_description(
        ordered_content: OrderedContent, include_background: bool
) -> list[str]:
    """
    Extract the solution cells from the ordered_content.
    we need to extract all numbered md cells, e.g. "1.1", "1.2", "1.3" etc.
    """
    problem_steps_description = []
    for content_type, content in ordered_content:
        if content_type == "markdown":
            # if there is a background section, we need to include it in the problem steps
            if (
                    "### background" in content.lower()
                    or "#### background" in content.lower()
            ):
                if include_background:
                    # Match from the "## 1.1" heading through the background section
                    regex = r"(## 1\.\d+[\s\S]+?(?=(## \d+|$)))"
                else:
                    # Match from the "## 1.1" heading up to just before "### Background" or "#### Background"
                    regex = r"(## 1\.\d+[\s\S]+?)(?=(###|####) Background)"
            else:
                # Match from the "## 1.1" heading to the end of the cell
                regex = r"(## 1\.\d+[\s\S]+)"

            match = re.search(regex, content, flags=re.DOTALL)
            if match:
                res = match.group(0).strip()
                problem_steps_description.append(res)

    return problem_steps_description


def extract_background(problem_steps_description: list[str]) -> list[str]:
    """
    Extract the background section from the ordered_content.
    """
    if isinstance(problem_steps_description, str):
        if "### background" in problem_steps_description.lower() or "#### background" in problem_steps_description.lower():
            # match from the "### Background" OR "#### Background" heading to the end of the cell
            regex = r"(### Background[\s\S]+|#### Background[\s\S]+)"
            match = re.search(regex, problem_steps_description, flags=re.DOTALL)
            if match:
                background = match.group(0).strip()
                return background
            else:
                return ""
        else:
            return ""
    background_all = []
    for content in problem_steps_description:
        if "### background" in content.lower() or "#### background" in content.lower():
            # match from the "### Background" OR "#### Background" heading to the end of the cell
            regex = r"(### Background[\s\S]+|#### Background[\s\S]+)"
            match = re.search(regex, content, flags=re.DOTALL)
            if match:
                background = match.group(0).strip()
                background_all.append(background)
            else:
                background_all.append("")
        else:
            background_all.append("")
    return background_all


def extract_test_from_cells(ordered_content: OrderedContent) -> list[str]:
    tests = []
    for content_type, content in ordered_content:
        if content_type == "code" and re.search(r"# test", content.lower()):
            tests.append(content)

    return tests


def extract_general_test(ordered_content: OrderedContent) -> list[str] | None:
    """find the markdown cell that contains lowercased "test case" and extract every cell after that"""
    if not ordered_content:
        return None
    for i, (content_type, content) in enumerate(ordered_content):
        if content_type == "markdown" and re.search(r"2\. test case", content.lower()):
            # return the content of all cells after the "test case" cell and not including cell type
            return [
                content[1]
                for content in ordered_content[i + 1:]
                if content[0] == "code" and "# test" in content[1].lower()
            ]


def extract_intermediate_test(
        ordered_content: OrderedContent, problem_steps_code: list[str], step: int
) -> list[str] | None:
    """extract intermediate test cases for multi-step problems given a step
    find the markdown cell that contains lowercased "test case" and extract every cell after that
    """
    if step == len(problem_steps_code):
        return extract_general_test(ordered_content)

    start_marker = f"# code 1.{step}"
    end_marker = f"# code 1.{step + 1}"
    end_marker_2 = f"2. test"  # the markdown cell that contains the general test
    test_marker = f"# test"

    start_index = 0
    end_index = len(ordered_content)

    for i, (content_type, content) in enumerate(ordered_content):
        if content_type == "code" and start_marker in content.lower():
            start_index = i
        if content_type == "code" and end_marker in content.lower():
            end_index = i
            break
        elif content_type == "markdown" and end_marker_2 in content.lower():
            end_index = i
            break

    tests = []

    for i, (content_type, content) in enumerate(ordered_content[start_index:end_index]):
        if content_type == "code" and test_marker in content.lower():
            tests.append(content)

    return tests


def process_duplicate(problem_step_prompt: str) -> bool:
    """Find whether the step is a duplicate step"""
    if "(Duplicate)" in problem_step_prompt or "[Duplicate]" in problem_step_prompt:
        return True
    return False


def process_problem_description(problem_step_prompt: str) -> str:
    """remove indicators from the prompt"""
    remove_list = ["# ", "#", "(Duplicate) ", "(Duplicate)", "[Duplicate] ", "[Duplicate]",
                   "__(Do not test)__ ", "__(Do not test)__", "(Do not Test) ", "(Do not Test)",
                   "(Main function) ", "(Main function)"]
    for item in remove_list:
        problem_step_prompt = problem_step_prompt.replace(item, "")
    return re.sub(r'\b1\.\d*\s*', '', problem_step_prompt)


def extract_function_name(function_header):
    pattern = r'\bdef\s+(\w+)\s*\('
    match = re.search(pattern, function_header)
    if match:
        return match.group(1)
    else:
        pattern = r'\bclass\s+(\w+)\s*\('
        match = re.search(pattern, function_header)
        if match:
            return match.group(1)
        else:
            raise ValueError('Function name or class name not found.')

def get_function_from_code(code_string, function_name):
    """
    Extracts and returns the source code of the specified function from a given source code string.

    :param code_string: String containing Python source code
    :param function_name: Name of the function to extract
    :return: String containing the source code of the function, or None if the function is not found
    """
    if code_string is None:
        return None
    try:
        # Parse the code into an AST
        tree = ast.parse(code_string)
        # Iterate through all nodes in the AST
        for node in ast.walk(tree):
            # Check if the node is a function definition
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                # Convert the AST back to a string containing the Python code for the function
                return ast.unparse(node)
    except Exception as e:
        print(f'{function_name} not found with error: {e}')
        return code_string


def read_from_json(file_path: Path | str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as file:
        # Load JSON data from file
        data = json.load(file)
        return data

def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def rm_comments(string: str) -> str:
    ret_lines = []
    lines = string.split('\n')
    for line in lines:
        if 'matplotlib' in line:
            continue
        if not line.startswith('#'):
            ret_lines.append(line)
    return '\n'.join(ret_lines)


def process_code_list(problem_code_lst: list) -> list:
    ret_lst = []
    for function_code in problem_code_lst:
        if "# test" in function_code.lower():
            if "assert" not in function_code:
                print(function_code)
                raise ValueError("Unexpected result!")
        lines = function_code.split('\n')
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                ret_lst.append('\n'.join(lines[i:]))
                break
    for i in range(len(ret_lst)):
        text = ret_lst[i]
        non_empty_lines = [line for line in text.split('\n') if line.strip() != '']
        ret_lst[i] = '\n'.join(non_empty_lines)
    return ret_lst


def parse_single_step_problem_to_dict(
        prob_num: int,
        ordered_content, problem_name: str
) -> dict[str, Any]:
    problem_description_with_background, problem_input_output, dependencies = extract_problem(
        ordered_content, include_background=True
    )
    problem_description_without_background, _, _ = extract_problem(
        ordered_content, include_background=False
    )
    problem_steps_code = process_code_list([extract_problem_code_single_step(ordered_content)])
    problem_background = extract_background(problem_description_with_background)

    json_dict = {
        "problem_name": problem_name,
        "problem_id": str(prob_num),
        "problem_description_main": process_problem_description(problem_description_without_background),
        "problem_background_main": process_problem_description(problem_background),
        "problem_io": problem_input_output,
        "required_dependencies": rm_comments(dependencies),
        "sub_steps": [],
        "general_solution": "\n".join(problem_steps_code),
        "general_tests": process_code_list(extract_test_from_cells(ordered_content)),
    }
    json_dict["sub_steps"].append(
        {
            "step_number": f"{prob_num}.1",
            "step_description_prompt": json_dict["problem_description_main"],
            "step_background": json_dict["problem_background_main"],
            "ground_truth_code": json_dict["general_solution"],
            "function_header": extract_function_headers_and_docs(
                json_dict["general_solution"], include_default_args=True
            ),
            "test_cases": json_dict["general_tests"],
            # "duplicate": process_duplicate(problem_description_without_background)
        }
    )
    return json_dict


def parse_multi_step_problem_to_dict(
        prob_num: int,
        ordered_content: OrderedContent, problem_name: str
) -> dict[str, Any]:
    problem_description_with_background, problem_input_output, dependencies = extract_problem(
        ordered_content, include_background=True
    )
    problem_description_without_background, _, _ = extract_problem(
        ordered_content, include_background=False
    )
    problem_steps_code = process_code_list(extract_problem_steps_code(ordered_content))
    problem_steps_description_with_background = extract_problem_steps_description(
        ordered_content, include_background=True
    )
    problem_steps_description_without_background = extract_problem_steps_description(
        ordered_content, include_background=False
    )
    problem_background = extract_background(problem_description_with_background)
    background = extract_background(problem_steps_description_with_background)

    json_dict = {
        "problem_name": problem_name,
        "problem_id": str(prob_num),
        "problem_description_main": process_problem_description(problem_description_without_background),
        "problem_background_main": process_problem_description(problem_background),
        "problem_io": problem_input_output,
        "required_dependencies": rm_comments(dependencies),
        "sub_steps": [],
        "general_solution": "\n".join(problem_steps_code),
        "general_tests": process_code_list(extract_general_test(ordered_content))
    }

    for i in range(len(problem_steps_code)):
        json_dict["sub_steps"].append(
            {
                "step_number": f"{prob_num}.{i + 1}",
                "step_description_prompt": process_problem_description(problem_steps_description_without_background[i]),
                "step_background": process_problem_description(background[i]),
                "ground_truth_code": problem_steps_code[i],
                "function_header": extract_function_headers_and_docs(
                    problem_steps_code[i], include_default_args=True
                ),
                "test_cases": process_code_list(extract_intermediate_test(
                    ordered_content, problem_steps_code, i + 1
                )),
                # "duplicate": process_duplicate(problem_steps_description_without_background[i])
            }
        )

    return json_dict


def parse_problem_to_dict(prob_num: int,
                          ordered_contnet: OrderedContent, problem_name: str, is_single_step: bool) -> dict[str, Any]:
    if is_single_step:
        return parse_single_step_problem_to_dict(prob_num, ordered_contnet, problem_name)
    else:
        return parse_multi_step_problem_to_dict(prob_num, ordered_contnet, problem_name)


def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num):
    data_lst = []
    with h5py.File(H5PY_FILE, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst


def save_data_to_hdf5(key, value, h5file):
    if isinstance(value, dict):
        subgroup = h5file.create_group(key)
        save_dict_to_hdf5(value, subgroup)
    elif isinstance(value, (list, tuple)):
        try:
            h5file.create_dataset(key, data=np.array(value))
        except Exception:
            group = h5file.create_group(key)
            subgroup = group.create_group('list')
            for i in range(len(value)):
                save_data_to_hdf5(f'var{i + 1}', value[i], subgroup)
    elif isinstance(value, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                            scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
        group = h5file.create_group(key)
        subgroup = group.create_group('sparse_matrix')
        subgroup.create_dataset('data', data=value.data)
        subgroup.create_dataset('shape', data=value.shape)
        if isinstance(value, scipy.sparse.coo_matrix):
            subgroup.create_dataset('row', data=value.row)
            subgroup.create_dataset('col', data=value.col)
        elif isinstance(value, scipy.sparse.bsr_matrix):
            subgroup.create_dataset('indices', data=value.indices)
            subgroup.create_dataset('indptr', data=value.indptr)
            subgroup.create_dataset('blocksize', data=value.blocksize)
        else:
            subgroup.create_dataset('indices', data=value.indices)
            subgroup.create_dataset('indptr', data=value.indptr)
    elif isinstance(value, (int, float, str, complex, bool,
                            np.bool_, np.int_, np.ndarray)):
        h5file.create_dataset(key, data=value)
    else:
        print(type(value))
        h5file.create_dataset(key, data=str(value))


def save_dict_to_hdf5(data_dict, h5file):
    for key, value in data_dict.items():
        if isinstance(key, (Symbol, np.float_)):
            key = str(key)
        if isinstance(value, dict):
            subgroup = h5file.create_group(key)
            save_dict_to_hdf5(value, subgroup)
        elif isinstance(value, (list, tuple)):
            h5file.create_dataset(key, data=np.array(value))
        elif isinstance(value, (int, float, str, complex, bool,
                                np.bool_, np.ndarray)):
            h5file.create_dataset(key, data=value)
        elif isinstance(value, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix,
                                scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            maxtrix_group = h5file.create_group(key)
            subgroup = maxtrix_group.create_group('sparse_matrix')
            subgroup.create_dataset('data', data=value.data)
            subgroup.create_dataset('shape', data=value.shape)
            if isinstance(value, scipy.sparse.coo_matrix):
                subgroup.create_dataset('row', data=value.row)
                subgroup.create_dataset('col', data=value.col)
            elif isinstance(value, scipy.sparse.bsr_matrix):
                subgroup.create_dataset('indices', data=value.indices)
                subgroup.create_dataset('indptr', data=value.indptr)
                subgroup.create_dataset('blocksize', data=value.blocksize)
            else:
                subgroup.create_dataset('indices', data=value.indices)
                subgroup.create_dataset('indptr', data=value.indptr)
        else:
            h5file.create_dataset(key, data=str(value))
