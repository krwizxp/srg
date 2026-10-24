from pathlib import Path
import os,sys,json,subprocess,hashlib,tarfile,platform
r=Path.cwd();s=Path(__file__).resolve().parent;target=sys.argv[1];e=r/'native-evidence';e.mkdir(exist_ok=True);summary=dict(target=target,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),runner_image={k:os.environ.get(k) for k in ['ImageOS','ImageVersion','RUNNER_OS','RUNNER_ARCH']},compiler='1.99.1',checks=[],versions={});env=dict(os.environ,RNG_ROOT=str(r),RNG_TARGET=target,RNG_EVIDENCE=str(e));suffix='.exe' if os.name=='nt' else ''
def run(label,args,cwd=r,timeout=600):
 p=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout);(e/(label+'.stdout.txt')).write_bytes(p.stdout);(e/(label+'.stderr.txt')).write_bytes(p.stderr);summary['checks'].append(dict(label=label,exit=p.returncode,command=list(map(str,args))));print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+':'+p.stderr.decode(errors='replace')[-2000:])
 return p
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected(checkout):
 names=subprocess.check_output(['git','ls-files','-z'],cwd=checkout).decode().strip('\0').split('\0');return {name:sha(checkout/name) for name in names if not name.startswith(('src/','.rng-validation/')) and name!='.github/workflows/optimization-rng.yml'}
def main():
 run('compiler',['rustc','+1.99.1','-Vv'])
 if os.name=='nt':run('cpu',['pwsh','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'])
 elif sys.platform=='darwin':run('cpu',['sysctl','-n','hw.model'])
 else:run('cpu',['lscpu'])
 for role in ['baseline','candidate']:
  checkout=r/'baseline' if role=='baseline' else r/'srg';before=protected(checkout);v=summary['versions'][role]=dict(protected=before,source={str(p.relative_to(checkout)).replace('\\','/'):sha(p) for p in (checkout/'src').rglob('*.rs')})
  run(role+'-fmt',['cargo','+1.99.1','fmt','--all','--','--check'],checkout)
  run(role+'-clippy',['cargo','+1.99.1','clippy','--frozen','--all-targets','--target',target,'--','-D','warnings'],checkout)
  run(role+'-tests',['cargo','+1.99.1','test','--release','--frozen','--target',target],checkout)
  run(role+'-distribution',['cargo','+1.99.1','build','--release','--frozen','--bin','srg','--target',target],checkout)
  binary=checkout/'target'/target/'release'/('srg'+suffix);v['executable_bytes']=binary.stat().st_size
  saved=env.get('CARGO_TARGET_DIR');env['CARGO_TARGET_DIR']=str(checkout/'target'/target)
  run(role+'-package',['cargo','+1.99.1','run','--frozen','--example','package_artifact','--','srg-native'],checkout)
  if saved is None:env.pop('CARGO_TARGET_DIR',None)
  else:env['CARGO_TARGET_DIR']=saved
  package=checkout/'artifacts'/('srg-native'+('.exe' if os.name=='nt' else '.tar'));v['package_bytes']=package.stat().st_size
  if os.name=='nt':assert package.read_bytes()==binary.read_bytes()
  else:
   with tarfile.open(package) as z:
    members=z.getmembers();assert len(members)==1 and z.extractfile(members[0]).read()==binary.read_bytes()
  assert protected(checkout)==before
 a,b=summary['versions']['baseline'],summary['versions']['candidate'];assert a['protected']==b['protected'];assert b['executable_bytes']<=a['executable_bytes']*1.01 and b['package_bytes']<=a['package_bytes']*1.01
 run('actual-cli-menu',[sys.executable,s/'check_runtime.py',target],timeout=600);summary['runtime']=json.loads((e/'runtime.json').read_text())
 if target.startswith('x86_64'):
  run('probe-builds',[sys.executable,s/'build_probes.py'])
  run('deterministic-corpus',[sys.executable,s/'check_corpus.py']);summary['corpus']=dict(passed=True,scenarios=json.loads((e/'corpus.json').read_text())['scenarios'])
  run('measurements',[sys.executable,s/'measure.py'],timeout=1200);summary['measurements']=json.loads((e/'measurements.json').read_text())
 else:summary['rng_corpus']='inapplicable to existing ARM RNG-disabled configuration; actual disabled CLI/menu paths passed'
 summary['passed']=True
try:main()
except BaseException as error:
 summary['passed']=False;summary['error']=repr(error);raise
finally:(e/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
