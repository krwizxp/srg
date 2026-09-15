from pathlib import Path
import os,shutil,json,subprocess
root=Path.cwd(); work=Path(os.environ.get('GOAL_WORK',root/'work3/remote-work'));work.mkdir(parents=True,exist_ok=True)
(work/'baseline.html').write_bytes(subprocess.check_output(['git','show','e6ffef5b2e638204f37ee8a8521dcb5550cdfc59:revgeo.html']))
(work/'candidate.html').write_bytes((root/'revgeo.html').read_bytes())
probe=work/'probe';shutil.copytree(root,probe,ignore=shutil.ignore_patterns('.git','goal-verify'),dirs_exist_ok=True)
p=probe/'src/main.rs';s=p.read_text().replace('fn main() -> Result<()> {','''fn main() -> Result<()> {
    if let Some(path) = env::var_os("GOAL_COMMON_CORPUS") {
        let mut out = io::stdout().lock();
        for (index, line) in std::fs::read_to_string(path)?.lines().enumerate() {
            let mut tokens = line.split(',');
            let num_64: u64 = tokens.next().ok_or("missing num64")?.parse().map_err(|e: core::num::ParseIntError| e.to_string())?;
            let mut consumed = 0_usize;
            let data = random_data::RandomDataSet { num_64, ..Default::default() }.populate(&mut |_| {
                consumed += 1;
                Ok(tokens.next().ok_or("supp exhausted")?.parse::<u64>().map_err(|e| e.to_string())?)
            })?;
            let mut buffer = [0_u8; BUFFER_SIZE];
            let len = output::format_data_into_buffer(&data, &mut buffer, output::OutputTarget::File);
            writeln!(out, "@@{index},{consumed}")?;
            out.write_all(&buffer[..len])?;
        }
        return Ok(());
    }''',1);p.write_text(s)
mask=(1<<64)-1;seed=0x2026091412345678
def nxt():
 global seed
 seed=(seed+0x9e3779b97f4a7c15)&mask;z=seed
 z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask;z=((z^(z>>27))*0x94d049bb133111eb)&mask
 return z^(z>>31)
values=[0,mask,1<<63,(1<<63)-1,0x1122334455667788]+[1<<i for i in range(64)]+[mask^(1<<i) for i in range(64)]+[nxt() for _ in range(512)]
corpus=[[str(v)]+[str(nxt()) for _ in range(32)] for v in values]
(work/'manual.csv').write_text('\n'.join(','.join(r) for r in corpus)+'\n');(work/'manual.json').write_text(json.dumps(corpus))
