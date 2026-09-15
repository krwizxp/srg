from pathlib import Path
import os,subprocess,json,hashlib,resource,time,statistics,random,shutil
root=Path.cwd();work=Path(os.environ['RUNNER_TEMP'])/'goal-batch-perf';work.mkdir(exist_ok=True);cpu=min(os.sched_getaffinity(0));phases=['baseline','candidate'];base='e6ffef5b2e638204f37ee8a8521dcb5550cdfc59'
plan={'seed':20260915,'warmup':5,'pairs':96,'count':65536,'order':'AB/BA','cpu':cpu,'time_upper_ci_pct':5,'rss_increase_max':'max(5%,512KiB)','reason':'Final complete Rust bundle; one dedicated fresh runner after builds, replacing local measurements contaminated by concurrent builds and forked Python RSS. Original inconclusive samples retained.','scope':'actual release batch with identical fixed entropy at hardware boundary; inner batch wall clock and /usr/bin/time CPU and peak RSS','refinement_limit':'one fixed run; do not enlarge budget or change threshold after results'}
(work/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
hook='''
    if std::env::var_os("GOAL_BATCH").is_some() {
        let rng = HardwareRng::new();
        let mut file = OutputFile::try_from(Path::new(FILE_NAME))?;
        let began = std::time::Instant::now();
        let result = batch::regenerate_with_count(&mut file, &rng, 65536, false, &mut io::sink());
        eprintln!("GOAL_BATCH_NS={}", began.elapsed().as_nanos());
        return result.map(|_| ());
    }
'''
for phase in phases:
 dest=work/phase;dest.mkdir(exist_ok=True)
 for path in subprocess.check_output(['git','ls-tree','-r','--name-only',base if phase=='baseline' else 'HEAD'],text=True).splitlines():
  if not (path.startswith('src/') or path in ['Cargo.toml','Cargo.lock','clippy.toml']):continue
  target=dest/path;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(subprocess.check_output(['git','show',base+':'+path]) if phase=='baseline' else (root/path).read_bytes())
 p=dest/'src/main.rs';p.write_text(p.read_text().replace('fn main() -> Result<()> {','fn main() -> Result<()> {'+hook,1))
 p=dest/'src/hardware_rng.rs';s=p.read_text().replace('pub(super) struct HardwareRng {','pub(super) struct HardwareRng {\n    goal_fixed: bool,',1).replace('Self {\n            fallback_notice_pending','Self {\n            goal_fixed: std::env::var_os("GOAL_BATCH").is_some(),\n            fallback_notice_pending',1).replace('pub(super) fn next_u64(&self) -> Result<u64> {','pub(super) fn next_u64(&self) -> Result<u64> {\n        if self.goal_fixed {return Ok(0x1122_3344_5566_7788);}',1);p.write_text(s)
 subprocess.run(['cargo','+1.98.1','build','--release','--frozen'],cwd=dest,env=os.environ|{'CARGO_TARGET_DIR':str(work/('target-'+phase))},check=True)
raw=[];directory=work/'output';directory.mkdir(exist_ok=True)
for i in range(-plan['warmup'],plan['pairs']):
 pair={}
 for phase in phases if i%2==0 else phases[::-1]:
  (directory/'random_data.txt').unlink(missing_ok=True);metrics=directory/'metrics'
  p=subprocess.run(['taskset','-c',str(cpu),'/usr/bin/time','-f','%U %S %M','-o',str(metrics),str(work/('target-'+phase)/'release/srg')],cwd=directory,env=os.environ|{'GOAL_BATCH':'1'},capture_output=True,timeout=30);assert p.returncode==0,p.stderr
  ns=int(p.stderr.split(b'GOAL_BATCH_NS=')[1].splitlines()[0]);user,system,rss=map(float,metrics.read_text().split());data=(directory/'random_data.txt').read_bytes();assert data.count('64비트 난수: '.encode())==plan['count'];pair[phase]={'wall_ns':ns,'cpu_seconds':user+system,'rss_kib':rss,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 assert pair['baseline']['sha256']==pair['candidate']['sha256']
 if i>=0:raw.append(pair)
metrics={}
for key in ['wall_ns','cpu_seconds']:
 ratios=[(p['candidate'][key]/p['baseline'][key]-1)*100 for p in raw];rng=random.Random(plan['seed']);boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000));ci=[boot[249],boot[9749]];metrics[key]={'baseline':statistics.median(p['baseline'][key] for p in raw),'candidate':statistics.median(p['candidate'][key] for p in raw),'change_pct':statistics.median(ratios),'ci95':ci,'pass':ci[1]<=5}
a=statistics.median(p['baseline']['rss_kib'] for p in raw);b=statistics.median(p['candidate']['rss_kib'] for p in raw);metrics['rss_kib']={'baseline':a,'candidate':b,'pass':b-a<=max(a*.05,512)}
report={'plan':plan,'metrics':metrics,'raw':raw,'passed':all(v['pass'] for v in metrics.values())};(Path(os.environ['RUNNER_TEMP'])/'goal-batch-perf.json').write_text(json.dumps(report,indent=2)+'\n');print('GOAL_BATCH_JSON='+json.dumps(report),flush=True);assert report['passed'],metrics
