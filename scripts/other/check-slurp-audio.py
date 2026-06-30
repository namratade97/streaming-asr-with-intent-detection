import os
import subprocess

BASE_DIR = os.path.expanduser("~/Downloads/audio")

SUBDIRS = ["train", "test", "val"]

def find_audio_file(filename):
    if not filename.endswith(".flac"):
        filename += ".flac"

    for subdir in SUBDIRS:
        dir_path = os.path.join(BASE_DIR, subdir)
        for root, _, files in os.walk(dir_path):
            if filename in files:
                return os.path.join(root, filename)
    return None

def play_audio(file_path):
    print(f"\n... Playing: {file_path}\n")
    # macOS
    if os.uname().sysname == "Darwin":
        subprocess.run(["afplay", file_path])
    # Linux
    else:
        subprocess.run(["aplay", file_path])

def main():
    while True:
        filename = input("\nEnter audio filename (or 'q' to quit): ").strip()
        if filename.lower() == 'q':
            break

        file_path = find_audio_file(filename)
        if file_path:
            print(f"Found: {file_path}")
            play_audio(file_path)
        else:
            print("File not found in train/test/val folders.")

if __name__ == "__main__":
    main()
