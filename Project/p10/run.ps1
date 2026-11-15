param([Parameter(ValueFromRemainingArguments=$true)]$RemainingArgs)
& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\main.py" @RemainingArgs
