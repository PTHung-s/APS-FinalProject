import os
import subprocess
import sys

print("="*70)
print("🚀 APS Final Project - Question 3 Full Pipeline")
print("="*70 + "\n")

scripts = [
    "00_convert_mkv_to_wav.py",
    "00_trim_to_9s.py",
    "02_entropy_delta_01.py",
    "03_sensitivity_analysis.py",
    "04_generate_spectrograms.py",
    "05_mutual_information_bonus.py"
]

success_count = 0
base_dir = os.getcwd()

for script in scripts:
    script_path = os.path.join(base_dir, script)
    
    if not os.path.exists(script_path):
        print(f"⚠️  File not found: {script}")
        continue

    print(f"🔄 Running: {script}")
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            cwd=base_dir,
            timeout=300  # timeout 5 phút
        )
        
        if result.returncode == 0:
            print(f"✅ Success: {script}")
            success_count += 1
        else:
            print(f"❌ Failed: {script} (Return code: {result.returncode})")
        
        # Hiển thị output quan trọng
        if result.stdout:
            print("Output:", result.stdout.strip()[:500] + "..." if len(result.stdout) > 500 else result.stdout.strip())
        if result.stderr:
            print("Error:", result.stderr.strip()[-400:])
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {script} took too long")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("-" * 60 + "\n")

print(f"🎉 Pipeline Completed! Success: {success_count}/{len(scripts)} scripts")
