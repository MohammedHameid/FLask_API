import subprocess
import sys

def install_spacy_model():
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])

if __name__ == '__main__':
    install_spacy_model()
