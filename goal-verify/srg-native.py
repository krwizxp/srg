from pathlib import Path
import hashlib,json,os,platform,re,subprocess,tarfile,tempfile,time,http.server,threading,email.utils
root=Path.cwd();binary=root/'target/release'/('srg.exe' if os.name=='nt' else 'srg');data=binary.read_bytes();report={'os':platform.platform(),'machine':platform.machine(),'compiler':subprocess.check_output(['rustc','+1.98.1','-Vv'],text=True),'binary_bytes':len(data),'binary_sha256':hashlib.sha256(data).hexdigest(),'checks':[]}
def invoke(args,directory=root,timeout=15):return subprocess.run([str(binary),*args],cwd=directory,capture_output=True,timeout=timeout)
class Handler(http.server.BaseHTTPRequestHandler):
 mode='valid';requests=0
 def log_message(self,*args):pass
 def do_HEAD(self):
  type(self).requests+=1
  if self.mode=='stall':time.sleep(6);return
  self.send_response_only(200);self.send_header('Date','invalid' if self.mode=='bad-date' else email.utils.formatdate(usegmt=True))
  if self.mode=='stale':self.send_header('Age','1')
  if self.mode=='duplicate-date':self.send_header('Date',email.utils.formatdate(usegmt=True))
  self.send_header('Content-Length','0');self.end_headers()
server=None
try:
 version=re.search(r'^version\s*=\s*"([^"]+)"',Path('Cargo.toml').read_text(),re.M).group(1)
 a=invoke(['--help']);b=invoke(['-h']);assert a.returncode==b.returncode==0 and a.stdout==b.stdout and not a.stderr and not b.stderr;report['checks'].append('help equivalence')
 p=invoke(['--version']);assert p.returncode==0 and p.stdout==f'srg {version}\n'.encode() and not p.stderr;report['checks'].append('exact version')
 for args in [['--unknown'],['--help','extra'],['--version','extra'],['generate','0'],['generate','-1'],['time-observe','localhost','0']]:
  p=invoke(args);assert p.returncode!=0 and p.stderr;report['checks'].append('reject '+repr(args))
 entry=os.environ['GOAL_ARTIFACT'];artifact=root/'artifacts'/(entry+('.exe' if os.name=='nt' else '.tar'))
 if os.name=='nt':assert artifact.read_bytes()==data
 else:
  with tarfile.open(artifact) as tar:
   members=tar.getmembers();assert len(members)==1;member=members[0];assert member.name==entry and member.isfile() and member.mode==0o755;assert tar.extractfile(member).read()==data
 report['artifact_bytes']=artifact.stat().st_size;report['checks'].append('package bytes and executable permissions')
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True;threading.Thread(target=server.serve_forever,daemon=True).start();report['http']=[]
 for mode in ['valid','bad-date','stale','duplicate-date','stall']:
  Handler.mode=mode;Handler.requests=0;start=time.monotonic();p=invoke(['time-observe',f'http://127.0.0.1:{server.server_port}','1'],timeout=8);seconds=time.monotonic()-start;text=(p.stdout+p.stderr).decode('utf8','replace');assert p.returncode==0 and Handler.requests>=1 and .8<seconds<3,(mode,p.returncode,seconds,text)
  if mode not in ['valid','stall']:assert any(token in text for token in ['실패','오류']),text
  report['http'].append({'mode':mode,'exit':p.returncode,'seconds':seconds,'requests':Handler.requests,'stdout':p.stdout.decode('utf8','replace'),'stderr':p.stderr.decode('utf8','replace')})
  report['checks'].append('native time observation '+mode)
 report['passed']=True
finally:
 if server:server.shutdown()
 print('GOAL_NATIVE_JSON='+json.dumps(report,ensure_ascii=False),flush=True)
 result=Path(os.environ['RUNNER_TEMP'])/'goal-native-result.json';result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
