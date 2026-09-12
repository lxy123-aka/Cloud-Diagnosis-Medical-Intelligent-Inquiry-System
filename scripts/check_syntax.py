import py_compile
import sys

files = [
    'pipeline/symptom_normalize.py',
    'agent/worker/collect_agent.py',
    'pipeline/diagnosis_engine.py',
    'agent/supervisor_agent.py',
    'utils/tools.py',
]

ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  OK: {f}")
    except py_compile.PyCompileError as e:
        print(f"  FAIL: {f}\n    {e}")
        ok = False

if ok:
    print("\nAll files compiled successfully!")
else:
    print("\nSome files have errors!")
    sys.exit(1)
