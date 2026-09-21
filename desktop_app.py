"""Launches the API and Streamlit servers as background processes and shows
the UI in a native desktop window (via pywebview) instead of a browser tab."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import webview

APP_DIR = Path(__file__).resolve().parent
STREAMLIT_EXE = APP_DIR / "virtual_env" / "Scripts" / "streamlit.exe"
UVICORN_EXE = APP_DIR / "virtual_env" / "Scripts" / "uvicorn.exe"
ICON_PATH = APP_DIR / "app_icon.ico"
WINDOW_TITLE = "Infor Support Assistant"
API_PORT = 8000
UI_PORT = 8501
URL = f"http://localhost:{UI_PORT}"

# The codebase prints emoji (e.g. "✅") to stdout/stderr. Windows' default
# console codepage (cp1252) can't encode those and crashes the process on
# startup, so force UTF-8 I/O for the subprocesses.
SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def set_taskbar_icon() -> None:
    import win32api
    import win32con
    import win32gui

    hwnd = None
    for _ in range(20):
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if hwnd:
            break
        time.sleep(0.25)
    if not hwnd:
        return

    hicon = win32gui.LoadImage(
        0, str(ICON_PATH), win32con.IMAGE_ICON, 0, 0,
        win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE,
    )
    win32api.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, hicon)
    win32api.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, hicon)


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("localhost", port)) == 0


def start_api() -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(UVICORN_EXE),
            "api.main:app",
            "--host", "127.0.0.1",
            "--port", str(API_PORT),
        ],
        cwd=str(APP_DIR),
        env=SUBPROCESS_ENV,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def start_streamlit() -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(STREAMLIT_EXE),
            "run",
            str(APP_DIR / "streamlit_app.py"),
            "--server.port", str(UI_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(APP_DIR),
        env=SUBPROCESS_ENV,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def wait_for_server(proc: subprocess.Popen, port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if port_open(port):
            return True
        time.sleep(0.3)
    return False


def main() -> None:
    api_proc = start_api()
    ui_proc = None
    try:
        if not wait_for_server(api_proc, API_PORT):
            sys.exit("Infor Support Assistant failed to start (API server).")

        ui_proc = start_streamlit()
        if not wait_for_server(ui_proc, UI_PORT):
            sys.exit("Infor Support Assistant failed to start (UI server).")

        window = webview.create_window(
            WINDOW_TITLE,
            URL,
            width=1280,
            height=860,
            min_size=(900, 600),
        )
        webview.start(set_taskbar_icon)
    finally:
        api_proc.terminate()
        if ui_proc is not None:
            ui_proc.terminate()


if __name__ == "__main__":
    main()
