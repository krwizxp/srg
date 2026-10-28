import os,sys,subprocess,struct,json,shutil,tempfile
from pathlib import Path
candidate=Path(sys.argv[1]).resolve();baseline=Path(sys.argv[2]).resolve();work=Path(tempfile.mkdtemp(prefix='srg-native-'))
index=(candidate/'data/korea-boundary-v1/index.bin').read_bytes();ends=[struct.unpack_from('<Q',index,i)[0] for i in range(32,len(index),24)];starts=[0]+[n+1 for n in ends[:-1]]
corpus=(candidate/'tools/validation/values.bin').read_bytes()+b''.join(struct.pack('<Q',v) for pair in zip(starts,ends) for v in pair);(work/'inputs.bin').write_bytes(corpus)
probe_template=r'''mod diagnostic {pub type Result<T> = std::result::Result<T,Box<dyn std::error::Error+Send+Sync>>;}
mod random_data {pub struct Coordinates {pub latitude:f64,pub longitude:f64}}
#[path="INFLATE"] mod boundary_inflate;
#[path="BOUNDARY"] mod korea_boundary;
fn calculate(bytes:&[u8])->diagnostic::Result<Vec<u8>>{
 let mut out=Vec::with_capacity(bytes.len()*2);
 for value in bytes.as_chunks::<8>().0 {
  let p=korea_boundary::sample(u64::from_le_bytes(*value))?;
  out.extend_from_slice(&p.latitude.to_bits().to_le_bytes());out.extend_from_slice(&p.longitude.to_bits().to_le_bytes());
 } Ok(out)
}
fn main()->diagnostic::Result<()> {
 let args:Vec<String>=std::env::args().collect();let bytes=std::fs::read(&args[1])?;
 let out=if args.get(3).is_some_and(|s|s=="threads") {
  std::thread::scope(|scope|->diagnostic::Result<Vec<u8>>{
   let width=(bytes.len()/8).div_ceil(8)*8;
   let handles:Vec<_>=bytes.chunks(width).map(|slice|scope.spawn(move||calculate(slice))).collect();
   let mut out=Vec::new();for handle in handles{out.extend(handle.join().map_err(|_|"Worker panic")??);}Ok(out)
  })?
 }else{calculate(&bytes)?};
 std::fs::write(&args[2],out)?;Ok(())
}
'''
outputs=[]
for label,source in [('baseline',baseline),('candidate',candidate)]:
 module=work/(label+'_boundary.rs');s=(source/'src/korea_boundary.rs').read_text(encoding="utf-8").replace('mod data;', '#[path="'+(source/'src/korea_boundary/data.rs').as_posix()+'"]\nmod data;');module.write_text(s,encoding="utf-8")
 probe=work/(label+'.rs');probe.write_text(probe_template.replace('INFLATE',(source/'src/boundary_inflate.rs').as_posix()).replace('BOUNDARY',module.as_posix()),encoding='utf-8')
 exe=work/(label+('.exe' if os.name=='nt' else ''))
 subprocess.run(['rustc','+1.99.1','--edition=2024','-O','-C','panic=abort',str(probe),'-o',str(exe)],check=True)
 for mode in ['single','threads']:
  output=work/(label+'_'+mode+'.bin');subprocess.run([str(exe),str(work/'inputs.bin'),str(output),mode],check=True);outputs.append(output.read_bytes())
assert all(v==outputs[0] for v in outputs)
print(json.dumps({'coordinate_cases':len(corpus)//8,'exact_coordinate_bits':True,'threads':8,'baseline_commit':'1300842994d74d96076237c60f498384a5fa029e'}))
