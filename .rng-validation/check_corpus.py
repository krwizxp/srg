from pathlib import Path
import os,subprocess,json,itertools,struct,random
from fractions import Fraction
r=Path(os.environ.get('RNG_ROOT',Path(__file__).resolve().parent.parent));e=Path(os.environ.get('RNG_EVIDENCE',r/'evidence'));suffix='.exe' if os.name=='nt' else '';records=[];D=1<<64
def trace(role,kind,a,b,mode='menu',modifier=0,words=None,fail=None,constant=None,limit=1000000):
 env=dict(os.environ)
 for name in ['MOCK_VALUES','MOCK_FAIL','MOCK_CONSTANT']:env.pop(name,None)
 if words is not None:env['MOCK_VALUES']=','.join(f'{w:x}' for w in words)
 if fail is not None:env['MOCK_FAIL']=str(fail)
 if constant is not None:env['MOCK_CONSTANT']=f'{constant:x}'
 p=subprocess.run([r/'probes'/(role+'-mock'+suffix),'trace',kind,mode,str(a),str(b),f'{modifier:x}',str(limit)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10);assert p.returncode==0,p.stderr
 f=dict(line.split('\t',1) for line in p.stdout.decode().splitlines());f['calls']=int(f['calls']);f['output']=bytes.fromhex(f['output']).decode();return f
def record(label,result):records.append(dict(case=label,**result))
def mapped(words,width,modifier=0):
 threshold=D%width
 for i,w in enumerate(words):
  product=(w^modifier)*width
  if product%D>=threshold:return product//D,i+1
 raise AssertionError('oracle ran out of words')
rng=random.Random(20261025)
for width in [1,2,3,5,7,10,257,(1<<32)+1,(1<<63)+1,D-1]:
 for value in [0,1,(1<<63)-1,1<<63,D-2,D-1,*[rng.randrange(D) for _ in range(6)]]:
  words=[value,D-1];expected,calls=mapped(words,width)
  old=trace('baseline','bounded',0,width-1,words=words);new=trace('candidate','bounded',0,width-1,words=words)
  assert new['status']=='ok' and int(new['output'])==expected
  assert new['calls']==(0 if width==1 else calls)
  assert old['status']=='ok' and old['output']==new['output'] and old['calls']==calls
  record(f'bounded-{width}-{value}',dict(expected=expected,baseline=old,candidate=new))
for mode in ['menu','cli']:
 for a,b in [(0,0),(7,7),(-7,-7),(1,10),(-100,100),(-(1<<63)+1,(1<<63)-1),((1<<63)-2,(1<<63)-1)]:
  words=[0x123456789abcdef0,D-1,1<<63]
  new=trace('candidate','integer',a,b,mode,modifier=0xffff,words=words);offset,n=mapped(words,b-a+1)
  assert new['status']=='ok' and new['calls']==(0 if a==b else n);expected=a+offset
  assert new['output']==f'무작위 정수({a} ~ {b}): {expected} (0x{expected & (D-1):X})\n'
  record(f'integer-{mode}-{a}-{b}',dict(expected=expected,candidate=new,baseline=trace('baseline','integer',a,b,mode,modifier=0xffff,words=words)))
def bits(x):return struct.pack('>d',x)
def float_oracle(word,a,b):
 if bits(a)==bits(b):return a
 scale=float(word)*2.0**-64
 value=float(Fraction.from_float(scale)*Fraction.from_float(b-a)+Fraction.from_float(a))
 return min(b,max(a,value))
for a,b in [(0.,1.),(-100.,100.),(7.,7.),(-0.,-0.),(-0.,0.),(1e-300,1e-200),(1e307,1e308),(-1e308,-1e307)]:
 for word in [0,1,(1<<53)-1,(1<<63)+1,D-1]:
  for mode in ['cli','menu']:
   n=trace('candidate','float',a,b,mode,modifier=D-1,words=[word]);assert n['status']=='ok' and n['calls']==1
   observed=float(n['output'].rsplit(': ',1)[1]);expected=float_oracle(word,a,b);assert bits(observed)==bits(expected),(a,b,word,observed,expected)
   record(f'float-{mode}-{a}-{b}-{word}',dict(expected=expected,candidate=n))
permutations=set()
for j2,j1 in itertools.product(range(3),range(2)):
 words=[max(1,(j2*D+2)//3),(j1*D)//2]
 n=trace('candidate','ladder',0,3,words=words);rows=n['output'].splitlines()[1:];observed=tuple(row.split(' -> ')[1] for row in rows);expected=['r0','r1','r2'];expected[2],expected[j2]=expected[j2],expected[2];expected[1],expected[j1]=expected[j1],expected[1]
 assert n['status']=='ok' and n['calls']==2 and observed==tuple(expected);permutations.add(observed);record(f'ladder-permutation-{j2}-{j1}',dict(candidate=n,expected=expected))
assert len(permutations)==6
for mode in ['cli','menu']:
 for count in [0,1,2,5,64,512]:
  n=trace('candidate','ladder',0,count,mode,modifier=0x0123456789abcdef);o=trace('baseline','ladder',0,count,mode,modifier=0x0123456789abcdef)
  assert n['status']=='ok' and n['calls']==max(0,count-1)
  rows=n['output'].splitlines()[1:];assert len(rows)==count and {row.split(' -> ')[1] for row in rows}=={f'r{i}' for i in range(count)}
  assert o['status']=='ok';record(f'ladder-count-{mode}-{count}',dict(baseline=o,candidate=n))
for kind,a,b in [('bounded',0,2),('integer',1,3)]:
 n=trace('candidate',kind,a,b,constant=0);assert n['calls']==1024 and 'rejection sampling' in n['status'];assert not n['output'];record('retry-limit-'+kind,dict(candidate=n))
 for fail in [0,1,1023]:
  n=trace('candidate',kind,a,b,constant=0,fail=fail);assert n['calls']==fail+1 and n['status']=='error:injected entropy failure' and not n['output'];record(f'rejected-failure-{kind}-{fail}',dict(candidate=n))
for kind,a,b in [('integer',1,10),('integer',7,7),('float',1.,10.),('float',7.,7.),('ladder',0,5)]:
 for mode in ['cli','menu']:
  o=trace('baseline',kind,a,b,mode,fail=0);n=trace('candidate',kind,a,b,mode,fail=0)
  allowed=kind=='integer' and a==b
  assert n['status']==('ok' if allowed else 'error:injected entropy failure');assert n['calls']==(0 if allowed else 1)
  record(f'first-failure-{kind}-{a}-{mode}',dict(approved_difference=allowed,baseline=o,candidate=n))
  for limit in [0,16]:
   o=trace('baseline',kind,a,b,mode,limit=limit);n=trace('candidate',kind,a,b,mode,limit=limit)
   assert n['status']=='error:injected output failure' and len(n['output'].encode())<=limit;assert o['status']==n['status'];record(f'output-failure-{kind}-{a}-{mode}-{limit}',dict(baseline=o,candidate=n))
for mode in ['cli','menu']:
 for kind,a,b in [('integer',1,10),('float',1.,10.),('ladder',0,5)]:
  left=trace('candidate',kind,a,b,mode,modifier=0);right=trace('candidate',kind,a,b,mode,modifier=D-1);assert left==right
  record(f'modifier-independence-{kind}-{mode}',dict(candidate=left))
for kind,a,b in [('integer',7,7),('integer',1,10),('float',1.,10.),('ladder',0,5)]:
 for mode in ['cli','menu']:
  versions={role:trace(role,kind,a,b,mode) for role in ['baseline','early','direct','candidate']}
  assert all(x['status']=='ok' for x in versions.values())
  if kind=='integer' and a==b:
   assert [versions[v]['calls'] for v in ['baseline','early','direct','candidate']]==([2,1,1,0] if mode=='cli' else [1,0,1,0])
  elif kind=='ladder':assert [versions[v]['calls'] for v in ['baseline','early','direct','candidate']]==([9,9,4,4] if mode=='cli' else [8,8,4,4])
  else:assert [versions[v]['calls'] for v in ['baseline','early','direct','candidate']]==([2,2,1,1] if mode=='cli' else [1,1,1,1])
  record(f'isolated-work-{kind}-{a}-{mode}',dict(versions=versions))
result=dict(passed=True,seed=20261025,scenarios=len(records),small_shuffle_permutations=len(permutations),method='Actual production number/ladder modules in isolated probe; Python exact-integer and Fraction rounding oracle; failure and resource bounds; mock is not evidence of hardware entropy quality',cases=records)
(e/'corpus.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('Passed',len(records),'scenarios,6 independently expected shuffle permutations')
