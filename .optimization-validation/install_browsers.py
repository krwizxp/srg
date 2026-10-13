"""Download pinned official browser binaries for this validation cycle."""
import argparse
from pathlib import Path
import tarfile
import urllib.request
import zipfile

kind = argparse.ArgumentParser()
kind.add_argument('--browser', choices=['chrome','firefox'], required=True)
args = kind.parse_args()
if args.browser == 'chrome':
    urls = [('chrome.zip','https://storage.googleapis.com/chrome-for-testing-public/155.0.8059.39/linux64/chrome-linux64.zip'),
            ('chromedriver.zip','https://storage.googleapis.com/chrome-for-testing-public/155.0.8059.39/linux64/chromedriver-linux64.zip')]
else:
    urls = [('firefox.tar.xz','https://archive.mozilla.org/pub/firefox/releases/157.0.1/linux-x86_64/en-US/firefox-157.0.1.tar.xz'),
            ('geckodriver.tar.gz','https://github.com/mozilla/geckodriver/releases/download/v0.37.0/geckodriver-v0.37.0-linux64.tar.gz')]
for filename,url in urls:
    urllib.request.urlretrieve(url,filename)
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as archive:
            for entry in archive.infolist():
                assert not Path(entry.filename).is_absolute() and '..' not in Path(entry.filename).parts
            archive.extractall('.')
    else:
        with tarfile.open(filename) as archive:archive.extractall('.',filter='data')
    Path(filename).unlink()
for executable in ['chrome-linux64/chrome','chrome-linux64/chrome_crashpad_handler','chromedriver-linux64/chromedriver','firefox/firefox','firefox/firefox-bin','geckodriver']:
    path=Path(executable)
    if path.exists():path.chmod(0o755)
