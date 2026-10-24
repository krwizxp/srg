from pathlib import Path
import os,subprocess,tempfile,json,re,sys,hashlib
r=Path(os.environ.get('RNG_ROOT',Path(__file__).resolve().parent.parent));e=Path(os.environ.get('RNG_EVIDENCE',r/'evidence'));target=sys.argv[1];suffix='.exe' if os.name=='nt' else '';supported=target.startswith('x86_64');records={};logs=e/'runtime';logs.mkdir(exist_ok=True)
invalid=[['--unknown'],['--help','extra'],['--version','extra'],['random-integer'],['random-integer','a','10'],['random-integer','10','1'],['random-integer',str(-(1<<63)),'0'],['random-integer','0',str(1<<63)],['random-integer','1','2','extra'],['random-float','nan','1'],['random-float','0','inf'],['random-float','1e-320','1'],['random-float','10','1'],['random-float','-1e308','1e308'],['ladder','a','x'],['ladder','a,b','x'],['ladder','a,,b','x,y,z'],['ladder','a,b','x,'],['ladder',','.join('p'+str(i) for i in range(513)),','.join('r'+str(i) for i in range(513))]]
def invoke(binary,args,label,work,stdin=None,timeout=30):
 p=subprocess.run([binary,*args],cwd=work,input=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout);(logs/(label+'.stdout.txt')).write_bytes(p.stdout);(logs/(label+'.stderr.txt')).write_bytes(p.stderr);return p
for role in ['baseline','candidate']:
 checkout=r/'baseline' if role=='baseline' else r/'srg';binary=checkout/'target'/target/'release'/('srg'+suffix);row=dict(invalid=[],cli=[],menu=[])
 with tempfile.TemporaryDirectory(prefix='srg-rng-runtime-') as directory:
  work=Path(directory)
  for i,args in enumerate([['--help'],['--version'],*invalid]):
   p=invoke(binary,args,f'{role}-grammar-{i}',work);assert (p.returncode==0)==(i<2);row['invalid'].append(dict(exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode()))
  if not supported:
   p=invoke(binary,['random-integer','7','7'],role+'-unsupported-cli',work);assert p.returncode!=0 and 'x86_64' in p.stderr.decode();row['unsupported_cli']=dict(exit=p.returncode,stdout=p.stdout.decode(),stderr=p.stderr.decode())
   p=invoke(binary,[],role+'-unsupported-menu',work,b'1\n2\nq\n');assert p.returncode==0 and '비활성화' in p.stdout.decode();row['unsupported_menu']=dict(exit=0,stdout=p.stdout.decode(),stderr=p.stderr.decode())
  else:
   ints=[];floats=[]
   for i in range(8):
    p=invoke(binary,['random-integer','-1000000','1000000'],f'{role}-live-int-{i}',work);assert p.returncode==0,p.stderr
    value=int(re.search(r': (-?\d+) \(0x',p.stdout.decode())[1]);assert -1000000<=value<=1000000;ints.append(value)
    p=invoke(binary,['random-float','-1000','1000'],f'{role}-live-float-{i}',work);assert p.returncode==0,p.stderr;value=float(p.stdout.decode().rsplit(': ',1)[1]);assert -1000<=value<=1000;floats.append(value)
   assert len(set(ints))>1 and len(set(floats))>1
   row['live_values']=dict(integer=ints,floating=floats,qualification='Observed variation and bounds, not a hardware entropy certification')
   for kind,a,b in [('random-integer','7','7'),('random-integer',str(-(1<<63)+1),str((1<<63)-1)),('random-float','3.5','3.5'),('random-float','1e307','1e308')]:
    p=invoke(binary,[kind,a,b],f'{role}-boundary-{len(row["cli"])}',work);assert p.returncode==0,p.stderr;row['cli'].append(dict(kind=kind,a=a,b=b,output=p.stdout.decode()))
   for n in [3,512]:
    p=invoke(binary,['ladder',','.join('p'+str(i) for i in range(n)),','.join('r'+str(i) for i in range(n))],f'{role}-live-ladder-{n}',work);assert p.returncode==0,p.stderr;lines=p.stdout.decode().splitlines()[1:];assert len(lines)==n;assert {line.split(' -> ')[1] for line in lines}=={f'r{i}' for i in range(n)};row['cli'].append(dict(kind='ladder',entries=n,bijective=True))
   cases=[('integer',b'2\n1\n1\n10\n2\n1\n7\n7\nq\n'),('float',b'2\n2\n1\n10\n2\n2\n3.5\n3.5\nq\n'),('ladder',b'1\np0,p1,p2\nr0,r1,r2\n1\np0,p1,p2\nr0,r1,r2\nq\n')]
   for label,data in cases:
    p=invoke(binary,[],role+'-menu-'+label,work,data);assert p.returncode==0,p.stderr;text=p.stdout.decode()
    if label=='integer':
     values=list(map(int,re.findall(r'무작위 정수\([^\n]*?\): (-?\d+)',text)));assert len(values)==2 and 1<=values[0]<=10 and values[1]==7
    elif label=='float':
     values=list(map(float,re.findall(r'무작위 실수\([^\n]*?\): ([^\n]+)',text)));assert len(values)==2 and 1<=values[0]<=10 and values[1]==3.5
    else:
     values=re.findall(r'p[012] -> (r[012])',text);assert len(values)==6 and set(values[:3])==set(values[3:])=={'r0','r1','r2'}
    row['menu'].append(dict(kind=label,invocations=2,without_data_regeneration=True,checked=True))
   row['file']=dict(bytes=(work/'random_data.txt').stat().st_size,sha256=hashlib.sha256((work/'random_data.txt').read_bytes()).hexdigest())
 records[role]=row
assert records['baseline']['invalid']==records['candidate']['invalid']
if not supported:assert records['baseline']['unsupported_cli']==records['candidate']['unsupported_cli'] and records['baseline']['unsupported_menu']==records['candidate']['unsupported_menu']
(e/'runtime.json').write_text(json.dumps(dict(passed=True,target=target,supported_rng=supported,versions=records),ensure_ascii=False,indent=2));print('Real CLI/menu runtime passed',target)
