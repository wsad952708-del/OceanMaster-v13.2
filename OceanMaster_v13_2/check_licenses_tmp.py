# flake8: noqa
import sys
import re
import urllib.request
import json
import time

def get_pkg_info(pkg_name):
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            version = data['info'].get('version', 'N/A')
            license_str = data['info'].get('license', '')
            if not license_str:
                license_str = ''
            classifiers = data['info'].get('classifiers', [])
            return version, license_str, classifiers
    except Exception as e:
        return 'N/A', f'Error: {e}', []

def parse_req(file_path):
    packages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            match = re.match(r'^([a-zA-Z0-9_\-]+)', line)
            if match:
                packages.append(match.group(1))
    return packages

def classify_license(license_str, classifiers):
    license_str = license_str.lower() if license_str else ''
    classifier_str = " ".join(classifiers).lower()
    
    if 'agpl' in license_str or 'agpl' in classifier_str or 'affero' in classifier_str:
        return 'AGPL'
    if 'lgpl' in license_str or 'lgpl' in classifier_str or 'lesser gpl' in classifier_str:
        return 'LGPL'
    if 'gpl' in license_str or 'gpl' in classifier_str or 'general public license' in classifier_str:
        return 'GPL'
    if 'mit' in license_str or 'mit' in classifier_str:
        return 'MIT'
    if 'apache' in license_str or 'apache' in classifier_str:
        return 'Apache'
    if 'bsd' in license_str or 'bsd' in classifier_str:
        return 'BSD'
    if 'mpl' in license_str or 'mpl' in classifier_str:
        return 'MPL'
    if 'psf' in license_str or 'psf' in classifier_str or 'python software foundation' in classifier_str:
        return 'PSF (Python)'
    
    if license_str and license_str != "unknown":
        return license_str.split('\n')[0][:50]
    
    for c in classifiers:
        if 'License ::' in c:
            return c.split('::')[-1].strip()
    return 'Unknown'

req_file = "requirements.txt"
packages = parse_req(req_file)

optional_pkgs = ['catboost', 'tabpfn', 'torch']
for p in optional_pkgs:
    if p not in packages:
        packages.append(p)

print("Package|Version|Classified")
for pkg in packages:
    ver, lic, classifiers = get_pkg_info(pkg)
    cls = classify_license(lic, classifiers)
    try:
        print(f"{pkg}|{ver}|{cls}")
    except:
        print(f"{pkg}|{ver}|EncodingError|{cls}")
    time.sleep(0.3)
