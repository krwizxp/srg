from pathlib import Path
import os,subprocess,json,shutil
r=Path(os.environ.get('RNG_ROOT',Path(__file__).resolve().parent.parent));s=Path(__file__).resolve().parent;target=os.environ.get('RNG_TARGET','x86_64-unknown-linux-gnu');e=Path(os.environ.get('RNG_EVIDENCE',r/'evidence'));e.mkdir(exist_ok=True);env=dict(os.environ,PATH='/root/.cargo/bin'+os.pathsep+os.environ['PATH']);records=[]
if not target.startswith('x86_64'):raise SystemExit('RNG functions are x86_64-only by existing product policy')
out=r/'probes';out.mkdir(exist_ok=True)
for role in ['baseline','candidate','early','direct']:
 origin=r/'baseline' if role in ['baseline','early'] else r/'srg'
 if role in ['early','direct']:
  origin=out/'variants'/role;(origin/'src').mkdir(parents=True,exist_ok=True)
  start=r/'baseline' if role=='early' else r/'srg'
  for name in ['hardware_rng.rs','random_number.rs','ladder.rs']:shutil.copy2(start/'src'/name,origin/'src'/name)
  f=origin/'src/random_number.rs';code=f.read_text();branch='    if inclusive_max == 0 {\n        return Ok(0);\n    }\n'
  code=code.replace('    let range_value = inclusive_max.strict_add(1);',branch+'    let range_value = inclusive_max.strict_add(1);') if role=='early' else code.replace(branch,'');f.write_text(code)
 for backend in (['mock'] if role in ['early','direct'] else ['mock','hardware']):
  text=(s/'probe.rs.in').read_text();rng=s/'mock_rng.rs' if backend=='mock' else origin/'src/hardware_rng.rs'
  signature='fn invoke(task:&Task,cli:bool,modifier:u64,rng:&HardwareRng,out:&mut dyn Write)->Result<()> {' if role in ['baseline','early'] else 'fn invoke(task:&Task,rng:&HardwareRng,out:&mut dyn Write)->Result<()> {'
  seed='if cli {rng.next_u64()?} else {modifier},' if role in ['baseline','early'] else ''
  invoke=signature+'''match task {
    Task::Integer(a,b)=>random_number::generate_random_integer(*a,*b,SEEDout,rng),
    Task::Float(a,b)=>random_number::generate_random_float(*a,*b,SEEDout,rng),
    Task::Ladder(players,results)=>ladder::write_ladder_results(players.iter().map(String::as_str),results.iter().map(String::as_str),SEEDrng,out),
    Task::Bounded(max)=>{let value=random_number::random_bounded_inclusive(*max,MODIFIERrng)?;writeln!(out,"{value}")?;Ok(())},
  }}'''.replace('SEED',seed).replace('MODIFIER','modifier,' if role in ['baseline','early'] else '')
  call='invoke(std::hint::black_box(&task),'+('cli,modifier,' if role in ['baseline','early'] else '')+'&rng,&mut out)'
  mapping={'RNG_SOURCE':str(rng.resolve()).replace('\\','/'),'NUMBER_SOURCE':str((origin/'src/random_number.rs').resolve()).replace('\\','/'),'LADDER_SOURCE':str((origin/'src/ladder.rs').resolve()).replace('\\','/'),'INVOKE_TRACE':call,'INVOKE_BENCH':call,'INVOKE':invoke,'BIND_MODE':'let cli=args[3]=="cli";let modifier=u64::from_str_radix(&args[6],16).unwrap();' if role in ['baseline','early'] else '', 'OBSERVED_CALLS':'rng.calls.get()' if backend=='mock' else '"unmeasured"','PREFLIGHT':'' if backend=='mock' else '''let source=match rng.source(){hardware_rng::HardwareRandomSource::None=>return Err("hardware RNG unsupported".into()),hardware_rng::HardwareRandomSource::RdSeed=>"RDSEED",hardware_rng::HardwareRandomSource::RdRand=>"RDRAND"};eprintln!("source:{source}");''','AFTER':'' if backend=='mock' else 'rng.write_rdseed_fallback_notice(&mut std::io::stderr())?;'}
  for a,b in mapping.items():text=text.replace(a,b)
  source=out/f'{role}-{backend}.rs';source.write_text(text);binary=out/(f'{role}-{backend}'+('.exe' if os.name=='nt' else ''))
  args=['rustc','+1.99.1','--edition=2024','-C','opt-level=3','-C','lto=fat','-C','codegen-units=1','-C','panic=abort','--target',target,source,'-o',binary]
  p=subprocess.run(list(map(str,args)),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(e/f'{role}-{backend}-probe-build.txt').write_bytes(p.stdout);records.append(dict(role=role,backend=backend,exit=p.returncode,command=list(map(str,args))));print(role,backend,p.returncode,flush=True)
  if p.returncode:raise RuntimeError(p.stdout.decode()[-3000:])
(e/'probe-builds.json').write_text(json.dumps(records,indent=2))
