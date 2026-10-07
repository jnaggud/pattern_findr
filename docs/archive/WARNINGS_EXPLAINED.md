# Streamlit Warnings Explained

## "ScriptRunContext" Warnings

### What They Are:
```
WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext!
This warning can be ignored when running in bare mode.
```

### Why They Happen:
These warnings occur when using **parallel processing** (multiple workers) because:
1. Streamlit UI elements (progress bars, text) need to run in the main thread
2. Parallel worker threads try to update the UI 
3. Streamlit detects this and warns you (but it still works!)

### Are They Harmful?
**NO** - These warnings are cosmetic and can be safely ignored:
- ✅ Your optimization still runs correctly
- ✅ Progress tracking still works
- ✅ Results are still valid
- ✅ No data corruption or crashes

### Why So Many Warnings?
With 7 parallel workers and many indicators, you get warnings because:
- Each indicator check triggers UI-related code
- Each worker thread generates warnings
- With 100 trials × 7 workers × many indicators = lots of warnings

### How to Reduce Warnings:

#### Option 1: Use 1 Worker (No Warnings)
- Set "Parallel Workers" to 1 in sidebar
- No warnings, but slower optimization
- Good for testing with fewer trials

#### Option 2: Ignore Them (Recommended)
- The warnings don't affect functionality
- Look past them to see the actual progress
- Focus on the `tqdm` progress bar which still works

#### Option 3: Suppress Warnings (Not Recommended)
We could suppress these warnings, but that would hide other potentially useful Streamlit warnings.

## Actual Errors vs. Warnings

### Warnings (Safe to Ignore):
```
WARNING streamlit.runtime.scriptrunner_utils.script_run_context
WARNING streamlit.runtime.state.session_state_proxy
```

### Real Errors (Need Attention):
```
ERROR - Error during strategy optimization: ...
TypeError: ...
```

## What You Should See:

### Terminal Output:
```
🚨 OPTIMIZATION START - Trials: 100, Workers: 7, Method: joblib

========================================
🚀 STANDARD OPTIMIZATION STARTING
Data shape: (252, 150)
Columns count: 150
Trials: 100
Workers: 7 (MULTI-THREADED)
========================================

Running 100 trials across 7 workers: [15, 15, 14, 14, 14, 14, 14]
Parallel Trials: 14%|███▌                     | 14/100 [00:45<04:20,  3.03s/it]
```

### Streamlit UI:
- Progress bar filling up
- "Progress: X / 100 trials (Y%)"
- "ETA: Z minutes remaining"

## The Fixed Error:

The error you just saw:
```
TypeError: run_optimization_distributed() got an unexpected keyword argument 'progress_callback'
```

**This is now FIXED**. The app now uses the standard optimization method which fully supports progress tracking.

## Bottom Line:

✅ **Warnings are normal** - Ignore the "ScriptRunContext" warnings  
✅ **Progress still works** - Watch the tqdm progress bar in terminal  
✅ **Optimization is running** - Your strategies are being discovered  
✅ **Error is fixed** - The TypeError won't happen anymore  

Focus on the **tqdm progress bar** in your terminal - that's the most reliable progress indicator with parallel processing!
