with open('solution.tex', 'r') as f:
    content = f.read()

# Replace all tables with resizebox
import re

# Function to wrap table in resizebox
def wrap_table(match):
    table_code = match.group(0)
    # If already wrapped, leave it
    if r'\resizebox' in table_code:
        return table_code
    # Find table caption and label
    return r'\begin{table}[H]' + '\n' + r'\centering' + '\n' + r'\small' + '\n' + r'\resizebox{\textwidth}{!}{%' + '\n' + match.group(1) + '\n' + r'}' + '\n' + match.group(2) + '\n' + r'\end{table}'

pattern = re.compile(r'\\begin\{table\}\[H\]\s*\\centering\s*\\small\s*(.*?)\\caption\{([^}]+)\}(.*?)\\end\{table\}', re.DOTALL)

# Let's write the whole file cleanly with resizebox on tables
