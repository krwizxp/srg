"""Inspect exported XCTest attachments; all ROIs are in native Safari chrome."""
from pathlib import Path
import json
import re
import numpy as np
from PIL import Image

ROOT = Path('downloaded-native')
EXPECTED = {'light': np.array([245,245,247]), 'dark': np.array([17,17,19])}


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def medians(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    h,w = image.shape[:2]
    output = {}
    for label,(x0,x1) in {'left':(.005,.02),'right':(.98,.995)}.items():
        crop = image[round(.92*h):round(.965*h),round(x0*w):round(x1*w),:]
        output[label] = np.median(crop, axis=(0,1)).tolist()
    return output


def main():
    mapping = {}
    for manifest in ROOT.rglob('manifest.json'):
        data = json.loads(manifest.read_text())
        for obj in objects(data):
            strings = {k:v for k,v in obj.items() if isinstance(v,str)}
            names = [v for v in strings.values() if '__' in v and any(v.startswith(x) for x in ['baseline-','clear-','inset-'])]
            files = [v for v in strings.values() if v.lower().endswith('.png')]
            for name in names:
                for file in files:
                    path = manifest.parent/file
                    if path.exists():
                        match = re.match(r'(?:baseline|clear|inset)-(?:light|dark)__(?:initial|opposite|returned|transmission-(?:light|dark)-(?:a|b))', name)
                        if match:
                            mapping[match.group()] = path
    if not mapping:
        for path in ROOT.rglob('*.png'):
            if re.match(r'(baseline|clear|inset)-(light|dark)__',path.stem):
                mapping[path.stem] = path
    assert mapping, 'Inspect attachment manifest schema before proceeding'
    results = {'files': {name:str(path) for name,path in mapping.items()},'pristine':[],'transmission':[]}
    for variant in ['baseline','clear','inset']:
        for system in ['dark','light']:
            prefix=f'{variant}-{system}__'
            for phase in ['initial','opposite','returned']:
                name=prefix+phase
                if name not in mapping:continue
                theme=('light' if system=='dark' else 'dark') if phase=='opposite' else system
                values=medians(mapping[name])
                error=max(float(np.max(np.abs(np.array(value)-EXPECTED[theme]))) for value in values.values())
                results['pristine'].append(dict(variant=variant,system=system,phase=phase,theme=theme,medians=values,max_error=error,passed=error<=40))
            for theme in ['light','dark']:
                names={pattern:prefix+f'transmission-{theme}-{pattern}' for pattern in ['a','b']}
                if any(name not in mapping for name in names.values()):continue
                values={p:medians(mapping[n]) for p,n in names.items()}
                differences={side:np.abs(np.array(values['a'][side])-np.array(values['b'][side])).tolist() for side in ['left','right']}
                # Red minus blue should reverse between the two screen edges,
                # and reverse again when the ordinary document stripes swap.
                contrasts={p:{side:float(np.array(values[p][side])[0]-np.array(values[p][side])[2]) for side in ['left','right']} for p in ['a','b']}
                passed=all(max(v)>=24 for v in differences.values()) and contrasts['a']['left']>=16 and contrasts['a']['right']<=-16 and contrasts['b']['left']<=-16 and contrasts['b']['right']>=16
                results['transmission'].append(dict(variant=variant,system=system,theme=theme,medians=values,differences=differences,contrasts=contrasts,passed=passed))
    Path('native-analysis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({key:value for key,value in results.items() if key!='files'},indent=2))


if __name__=='__main__':
    main()
