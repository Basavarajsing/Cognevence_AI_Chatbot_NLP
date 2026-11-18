"""
Automation Script to Populate Real Chat Data and Capture Authentic Screenshots.
Starts Flask server in background, simulates user queries, takes screenshots
via headless Chrome/Edge, and shuts down cleanly.
"""

import os
import sys
import subprocess
import time
import requests

def main():
    os.makedirs("screenshots", exist_ok=True)
    
    print("[*] Launching Flask server with sys.executable...")
    server = subprocess.Popen([sys.executable, "app.py"])
    
    try:
        # Retry waiting for server up to 10 seconds
        connected = False
        for attempt in range(10):
            time.sleep(1)
            try:
                health = requests.get("http://127.0.0.1:5000/health", timeout=2)
                if health.status_code == 200:
                    print(f"[✓] Server healthy: {health.json()}")
                    connected = True
                    break
            except Exception:
                print(f"[*] Waiting for server (attempt {attempt + 1}/10)...")

        if not connected:
            print("[!] Could not connect to Flask server.")
            return

        # Populate user queries
        queries = [
            "What is Artificial Intelligence?",
            "Explain Natural Language Processing",
            "What is Machine Learning?",
            "How does Python work?",
            "What are top final year project ideas?",
            "Tips for technical interview preparation"
        ]

        session = requests.Session()
        for q in queries:
            try:
                resp = session.post("http://127.0.0.1:5000/chat", json={"message": q}, timeout=5)
                data = resp.json()
                print(f"[*] Query: '{q}' -> Intent: {data.get('intent')} ({data.get('confidence')})")
            except Exception as e:
                print(f"[!] Error querying chat: {e}")

        time.sleep(1)

        # Locate Chrome or Edge
        browser_exe = None
        for candidate in [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        ]:
            if os.path.exists(candidate):
                browser_exe = candidate
                break

        if not browser_exe:
            print("[!] Warning: Neither Chrome nor Edge executable found.")
            return

        print(f"[*] Using browser binary: {browser_exe}")

        # Capture Chat Interface Screenshot
        shot1 = os.path.abspath(os.path.join("screenshots", "chat_interface.png"))
        subprocess.run([
            browser_exe,
            "--headless=new",
            "--disable-gpu",
            f"--screenshot={shot1}",
            "--window-size=1280,820",
            "http://127.0.0.1:5000"
        ], check=True)
        print(f"[✓] Saved chat screenshot: {shot1}")

        # Capture Analytics Dashboard Screenshot
        shot2 = os.path.abspath(os.path.join("screenshots", "dashboard_view.png"))
        subprocess.run([
            browser_exe,
            "--headless=new",
            "--disable-gpu",
            f"--screenshot={shot2}",
            "--window-size=1280,950",
            "http://127.0.0.1:5000/dashboard"
        ], check=True)
        print(f"[✓] Saved dashboard screenshot: {shot2}")

    finally:
        print("[*] Terminating Flask server...")
        server.terminate()
        server.wait()
        print("[✓] Flask server stopped.")

if __name__ == "__main__":
    main()
