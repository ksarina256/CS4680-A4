# Prompt iteration results
Model `gemini-3.5-flash-lite` | 21 test requests x 1 run(s) per prompt version | 2026-11-05

| Prompt version | First-try valid JSON | Valid after repair loop | Correct tools | Risk label correct | Avg LLM calls | Avg secs |
|---|---|---|---|---|---|---|
| v1_baseline | 0% | 24% | 10% | n/a | 2.81 | 4.9 |
| v2_tools_schema | 100% | 100% | 100% | 76% | 1.0 | 3.7 |
| v3_rules | 100% | 100% | 100% | 88% | 1.0 | 3.5 |
| v4_final | 100% | 100% | 100% | 94% | 1.0 | 2.7 |

## Failures (error analysis: decide the next prompt change from these)

### v1_baseline: 19 failing
- `save a grocery list with milk, eggs and bread`: expected ['write_file']; The model could not produce a valid plan. Try rephrasing your request.
- `create a file called ideas.txt that says build a robot`: expected ['write_file']; The model could not produce a valid plan. Try rephrasing your request.
- `add the line 'call mom' to the end of notes.txt`: expected ['append_file']; The model could not produce a valid plan. Try rephrasing your request.
- `make a folder called projects`: expected ['make_dir']; The model could not produce a valid plan. Try rephrasing your request.
- `what files do I have?`: expected ['list_files']; The model could not produce a valid plan. Try rephrasing your request.
- `show me what's in notes.txt`: expected ['read_file']; The model could not produce a valid plan. Try rephrasing your request.
- `remind me to submit the assignment on Friday`: expected ['db_add_task']; The model could not produce a valid plan. Try rephrasing your request.
- `what's on my to-do list?`: expected ['db_list_tasks']; The model could not produce a valid plan. Try rephrasing your request.
- `mark task 1 as done`: expected ['db_complete_task']; The model could not produce a valid plan. Try rephrasing your request.
- `what time is it?`: expected ['system_info']; got no actions
- `how much disk space is left?`: expected ['system_info']; got no actions
- `look up the GitHub repo anthropics/anthropic-sdk-python`: expected ['http_get']; got no actions
- ...and 7 more (see eval_results.json)

### v2_tools_schema: 4 failing
- `rename notes.txt to archive.txt`: expected ['move_file']; wrong risk label
- `delete old.txt`: expected ['delete_file']; wrong risk label
- `overwrite notes.txt with the text hello`: expected ['write_file']; wrong risk label
- `delete task 2`: expected ['db_delete_task']; wrong risk label

### v3_rules: 2 failing
- `rename notes.txt to archive.txt`: expected ['move_file']; wrong risk label
- `overwrite notes.txt with the text hello`: expected ['write_file']; wrong risk label

### v4_final: 1 failing
- `rename notes.txt to archive.txt`: expected ['move_file']; wrong risk label
