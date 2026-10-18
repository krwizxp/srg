"""Read real-gesture XCTest exports; native ROI thresholds fixed before inspection."""
from pathlib import Path
import json
import re
import numpy as np
from PIL import Image

ROOT=Path('gesture-native')
VARIANTS=['baseline','inset8','inset24']
EXPECTED={'light':np.array([245,245,247]),'dark':np.array([17,17,19])}
PATTERN=re.compile(r'(?:baseline|inset8|inset24)-(?:light|dark)__(?:pristine-(?:initial|up|down|opposite|returned)|fixture-(?:initial|up|down)|after-scroll-(?:light|dark)(?:-[ab])?)')


def objects(value):
    if isinstance(value,dict):
        yield value
        for v in value.values():yield from objects(v)
    elif isinstance(value,list):
        for v in value:yield from objects(v)


def medians(path):
    a=np.asarray(Image.open(path).convert('RGB'));h,w=a.shape[:2]
    return {name:np.median(a[round(.92*h):round(.965*h),round(x0*w):round(x1*w),:],axis=(0,1)).tolist()
            for name,(x0,x1) in {'left':(.005,.02),'right':(.98,.995)}.items()}


def main():
    mapping={};movements={}
    for manifest in ROOT.rglob('manifest.json'):
        for obj in objects(json.loads(manifest.read_text())):
            name=obj.get('suggestedHumanReadableName','');file=obj.get('exportedFileName','')
            match=PATTERN.match(name)
            if file.endswith('.png') and match:
                mapping[match.group()]=manifest.parent/file
            if '__gesture-movement' in name and file.endswith('.txt'):
                text=(manifest.parent/file).read_text()
                movements[name.split('__')[0]]=text
    results={'files':{k:str(v) for k,v in mapping.items()},'movement':movements,'themes':[],'optics':[]}
    for variant in VARIANTS:
        for system in ['dark','light']:
            prefix=f'{variant}-{system}__'
            phases={'pristine-initial':system,'pristine-down':system,
                    'pristine-opposite':'light' if system=='dark' else 'dark',
                    'pristine-returned':system,'after-scroll-dark':'dark','after-scroll-light':'light'}
            for phase,theme in phases.items():
                if prefix+phase not in mapping:continue
                values=medians(mapping[prefix+phase]);error=max(float(np.max(np.abs(np.array(v)-EXPECTED[theme]))) for v in values.values())
                results['themes'].append(dict(variant=variant,system=system,phase=phase,theme=theme,medians=values,max_error=error,passed=error<=40))
            for theme in ['dark','light']:
                names={p:prefix+f'after-scroll-{theme}-{p}' for p in ['a','b']}
                if any(n not in mapping for n in names.values()):continue
                values={p:medians(mapping[n]) for p,n in names.items()}
                differences={side:np.abs(np.array(values['a'][side])-np.array(values['b'][side])).tolist() for side in ['left','right']}
                contrasts={p:{side:float(v[0]-v[2]) for side,v in s.items()} for p,s in values.items()}
                passed=all(max(v)>=24 for v in differences.values()) and contrasts['a']['left']>=16 and contrasts['a']['right']<=-16 and contrasts['b']['left']<=-16 and contrasts['b']['right']>=16
                results['optics'].append(dict(variant=variant,system=system,theme=theme,medians=values,differences=differences,contrasts=contrasts,passed=passed))
    import sys
    if '--require-selected' in sys.argv:
        selected_themes=[r for r in results['themes'] if r['variant']=='inset8']
        selected_optics=[r for r in results['optics'] if r['variant']=='inset8']
        assert len(selected_themes)==12 and all(r['passed'] for r in selected_themes), 'Missing or failed selected theme case'
        assert len(selected_optics)==4 and all(r['passed'] for r in selected_optics), 'Missing or failed selected optical case'
        for variant in ['baseline','inset8']:
            for system in ['dark','light']:
                key=f'{variant}-{system}'
                assert abs(float(movements[key].split('displacement=')[1]))>30, f'No actual gesture: {key}'
        for system in ['dark','light']:
            reproduced=[r for r in results['themes'] if r['variant']=='baseline' and r['system']==system and r['phase']=='pristine-opposite']
            assert len(reproduced)==1 and not reproduced[0]['passed'], f'Baseline bug not reproduced: {system}'
        baseline_optics=[r for r in results['optics'] if r['variant']=='baseline']
        assert len(baseline_optics)==4 and all(not r['passed'] for r in baseline_optics)
    Path('gesture-native-analysis.json').write_text(json.dumps(results,indent=2))
    print('Images',len(mapping),'movements',movements)
    for key in ['themes','optics']:
        for v in VARIANTS:
            a=[x for x in results[key] if x['variant']==v]
            print(key,v,sum(x['passed'] for x in a),'/',len(a))
    for x in results['themes']:print('THEME',x)
    for x in results['optics']:print('OPTIC',x)


if __name__=='__main__':main()
