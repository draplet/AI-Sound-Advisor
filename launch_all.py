"""
Launch both Fake X32 and Audio Pilot together.

Usage:
    python launch_all.py

Then open: http://127.0.0.1:8001
"""
import subprocess
import time
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    python = r"C:\Users\dr_ap\AppData\Local\Programs\Python\Python310\python.exe"

    print("=" * 70)
    print("AUDIO PILOT DEMO LAUNCHER")
    print("=" * 70)
    print()
    print("Starting Fake X32 and Audio Pilot...")
    print()

    try:
        # Start Fake X32
        print("[1/2] Starting Fake X32 OSC Server...")
        fake_x32_proc = subprocess.Popen(
            [python, str(project_root / "fake_x32" / "run_fake_x32.py")],
            cwd=str(project_root)
        )
        time.sleep(2)  # Give it time to start

        # Start Audio Pilot
        print("[2/2] Starting Audio Pilot Dashboard...")
        dashboard_proc = subprocess.Popen(
            [python, "run_dashboard.py"],
            cwd=str(project_root)
        )
        time.sleep(2)

        print()
        print("=" * 70)
        print("✅ Both services started!")
        print("=" * 70)
        print()
        print("Open your browser:")
        print("  http://127.0.0.1:8001")
        print()
        print("To connect in Audio Pilot:")
        print("  1. Go to X32 Network panel")
        print("  2. IP: 10.0.0.19")
        print("  3. Port: 10023")
        print("  4. Click Test or Connect")
        print()
        print("Press Ctrl+C to stop both services")
        print("=" * 70)
        print()

        # Wait for both processes
        fake_x32_proc.wait()
        dashboard_proc.wait()

    except KeyboardInterrupt:
        print()
        print()
        print("Shutting down...")
        fake_x32_proc.terminate()
        dashboard_proc.terminate()

        # Give them time to shutdown gracefully
        time.sleep(1)

        # Force kill if needed
        try:
            fake_x32_proc.kill()
            dashboard_proc.kill()
        except:
            pass

        print("Done.")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
