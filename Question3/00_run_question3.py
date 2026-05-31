import os
import subprocess
import sys

print("🚀 Starting Question 3 Pipeline...\n")

scripts = [
    "00_convert_mkv_to_wav.py",
    "00_trim_to_9s.py",
    "02_entropy_delta_01.py",
    "03_sensitivity_analysis.py",
    "04_generate_spectrograms.py",
    "05_mutual_information_bonus.py"
]

for script in scripts:
    script_path = script
    if not os.path.exists(script_path):
        print(f"⚠️  File {script} not found! Skipping...")
        continue
    
    print(f"🔄 Running {script} ...")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        print(f"✅ Finished {script}\n")
        if result.stdout:
            print(result.stdout[-500:])  # Show last part of output
        if result.stderr:
            print("Error:", result.stderr[-300:])
            
    except Exception as e:
        print(f"❌ Error running {script}: {e}\n")

print("🎉 Question 3 Pipeline Completed!")
