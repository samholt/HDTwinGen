import subprocess
import pickle
import base64
import traceback
import ast
import tempfile
import os
import json
from timeout_decorator import timeout
from llm_utils import num_tokens_from_messages
import numpy as np
from pathlib import Path

def extract_exception(output):
    """Try to extract exception information from the output. If found, raise it in the main process."""
    try:
        # Try to interpret the last line of output as a serialized exception
        exception_data = ast.literal_eval(output.splitlines()[-1])
        exception_message = exception_data.get('exception_message', '')
        exception_traceback = exception_data.get('exception_traceback', '')
        return exception_message, exception_traceback
    except (SyntaxError, ValueError):
        # If we can't interpret the output as a serialized exception, return None values
        return None, None

class Executor:
    def __init__(self, prepend_code_libraries="", variables=None):
        self.prepend_code_libraries = prepend_code_libraries
        if variables is None:
            self.variables = {}
        else:
            self.variables = variables

    def execute_user_code_lines(self, code_string):
        serialized_vars = base64.b64encode(pickle.dumps(self.variables)).decode('utf-8')
        lines = code_string.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if not (last_line.startswith("print(") or "=" in last_line or "import" in last_line or "#" in last_line):
                code_string += f"\nprint({last_line})"

        def auto_indent(code, spaces=4):
            indentation = ' ' * spaces
            return '\n'.join([indentation + line for line in code.split('\n')])

        full_code = f"""
import pickle
import base64
import traceback
import sys

# Deserialize variables
variables = pickle.loads(base64.b64decode('{serialized_vars}'.encode('utf-8')))
locals().update(variables)

try:
{auto_indent(self.prepend_code_libraries)}
{auto_indent(code_string)}
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    formatted_traceback = ''.join(traceback.format_tb(exc_traceback))
    sys.stderr.write(str({{
        'exception_message': str(exc_value),
        'exception_traceback': formatted_traceback
    }}))
    raise SystemExit(1)
"""
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.py') as temp:
            temp.write(full_code)
            temp_file_name = temp.name
            
        try:
            result = subprocess.run(
                ["python3", temp_file_name], 
                capture_output=True, 
                text=True)
        finally:
            os.remove(temp_file_name)
        if result.returncode != 0:
            if "SyntaxError" in result.stderr:
                raise SyntaxError(f"Syntax error in user code:\n{result.stderr.strip()}")
            else:
                exception_message, exception_traceback = extract_exception(result.stderr)
            raise Exception(f"{result.stdout}\nUser code exception: {exception_message}\n\n{exception_traceback}")
        else:
            captured_output = result.stdout.strip()
            return captured_output

if __name__ == '__main__':
    pass
    # unittest.main()
    # TestExecutor().test_pytest_too_much_error_out()
