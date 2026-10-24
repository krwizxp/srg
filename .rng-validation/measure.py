from pathlib import Path
import os,subprocess,json,sys,random,statistics
r=Path(os.environ.get('RNG_ROOT',Path(__file__).resolve().parent.parent));e=Path(os.environ.get('RNG_EVIDENCE',r/'evidence'));suffix='.exe' if os.name=='nt' else '';backends=os.environ.get('RNG_BACKENDS','mock hardware').split();result=dict(method='3warmups+31 alternating pairs,calibrated100ms baseline,paired-median10000bootstrap95 seed20261025,<=5% median and upper95',workloads=[])
cases=[('int-cli-narrow','integer','cli','1','10'),('int-menu-narrow','integer','menu','1','10'),('int-cli-wide','integer','cli',str(-(1<<63)+1),str((1<<63)-1)),('float-cli','float','cli','1','10'),('float-menu','float','menu','1','10'),('float-cli-equal','float','cli','3.5','3.5'),('int-cli-singleton','integer','cli','7','7'),('int-menu-singleton','integer','menu','7','7'),('ladder-cli-2','ladder','cli','0','2'),('ladder-menu-2','ladder','menu','0','2'),('ladder-cli-64','ladder','cli','0','64'),('ladder-menu-64','ladder','menu','0','64'),('ladder-cli-512','ladder','cli','0','512')]
def run(role,backend,case,count,alloc=False):
 label,kind,mode,a,b=case;args=[r/'probes'/(role+'-'+backend+suffix),'bench',kind,mode,a,b,'123456789abcdef',str(count)];env=dict(os.environ)
 for name in ['MOCK_VALUES','MOCK_FAIL','MOCK_CONSTANT','COUNT_ALLOCATIONS']:env.pop(name,None)
 if alloc:env['COUNT_ALLOCATIONS']='1'
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=120)
 if p.returncode:raise RuntimeError((role,backend,label,p.returncode,p.stderr.decode()))
 fields=dict(line.split('\t',1) for line in p.stdout.decode().splitlines());return dict(ns=int(fields['elapsed']),calls=fields['calls'],allocations=list(map(int,fields['allocations'].split())),source=p.stderr.decode())
for backend in backends:
 for case in cases:
  first=run('baseline',backend,case,100);count=max(10,min(3000000,(100000000*100)//max(1,first['ns'])));cal=run('baseline',backend,case,count);count=max(10,min(3000000,(100000000*count)//max(1,cal['ns'])))
  allocations={v:run(v,backend,case,100,True) for v in ['baseline','candidate']}
  assert allocations['candidate']['allocations'][0]<=allocations['baseline']['allocations'][0] and allocations['candidate']['allocations'][1]<=allocations['baseline']['allocations'][1],allocations
  samples={v:[] for v in ['baseline','candidate']};sources={v:set() for v in samples}
  for repeat in range(34):
   for v in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
    row=run(v,backend,case,count);sources[v].add(row['source'])
    if repeat>=3:samples[v].append(row['ns'])
  med={v:statistics.median(xs) for v,xs in samples.items()};paired=[(b/a-1)*100 for a,b in zip(samples['baseline'],samples['candidate'])];rng=random.Random(20261025);boot=sorted(statistics.median(rng.choices(paired,k=31)) for _ in range(10000));delta=(med['candidate']/med['baseline']-1)*100
  record=dict(name=case[0],backend=backend,count=count,samples_ns=samples,median_ns=med,delta_percent=delta,paired_median_percent=statistics.median(paired),lower95=boot[249],upper95=boot[9749],passed=delta<=5 and boot[9749]<=5,resources=allocations,sources={v:sorted(xs) for v,xs in sources.items()});result['workloads'].append(record)
  print(backend,case[0],round(delta,2),round(record['upper95'],2),record['passed'],flush=True)
  result['passed']=all(x['passed'] for x in result['workloads']);result['complete']=False;(e/'measurements.json').write_text(json.dumps(result,indent=2))
result['complete']=True;(e/'measurements.json').write_text(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(1)
