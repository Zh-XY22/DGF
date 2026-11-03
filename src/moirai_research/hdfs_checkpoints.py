"""Optional cluster storage adapter; disabled in ordinary public runs."""
import atexit
import base64
import io
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import tempfile
import time
import zipfile

_CLASS_DATA='yv66vgAAADQBiAoAcwCbBwCcCgACAJsJAEsAnQoAngCfCgCgAKEJAEsAogoAowCkCQBLAKUHAKYKAKcAqAoAqQCqCQCrAKwKAAoArQcArgcArwoAEACwCgAPALEDAEAAAAoAsgCzCgC0ALUHALYIALcKABYAuAgAuQoAtAC6CAC7CgCyALwHAL0KAB0AvgoACgC/CADACgAKAMEIAMIHAMMKACMAuAoAIwDECADFCgAjAMYKACMAxwcAyAoAKQC4CgApAMkKAKAAygoAoADLCgCgAMwKAM0AzggAzwoAoADQCgBLANEKALQAvAgA0gcA0woANQCbCADUCgA1ANUKACkA1ggA1woA2ADZCgA1ANoKADUA2woAKQDcCgCgAN0KAN4A3woA3gDgCgDeALwIAOEIAOIHAOQJAEUA5wkARQDoCgCjAOkIAOoKAKAA6wcA7AoASwCbBwDtBwDuCQDvAPAKAE4A8QoATQDyBwDzBwD0CQDvAPUKAFMA9goAUgD3CgBNAPgIAPkKAAoA+ggA+woASwD8CAD9CgAKAP4KAEsA/wgBAAoANQEBCgBSAQIHAQMKAHMBBAoBBQEGCAEHCgBiAQgKAAoBCQoACgEKCAELCgCnAQwKAAoBDQoBDgEPCAEQCgDvAREIARIIARMKAAoBFAgBFQcBFgEABlBSRUZJWAEAEkxqYXZhL2xhbmcvU3RyaW5nOwEAAmZzAQAhTG9yZy9hcGFjaGUvaGFkb29wL2ZzL0ZpbGVTeXN0ZW07AQAHY29udGV4dAEAIkxvcmcvYXBhY2hlL2hhZG9vcC9mcy9GaWxlQ29udGV4dDsBAAY8aW5pdD4BAAMoKVYBAARDb2RlAQAPTGluZU51bWJlclRhYmxlAQAKRXhjZXB0aW9ucwEABmRlY29kZQEAJihMamF2YS9sYW5nL1N0cmluZzspTGphdmEvbGFuZy9TdHJpbmc7AQAHY29tcGFyZQEAOShMamF2YS9pby9GaWxlO0xvcmcvYXBhY2hlL2hhZG9vcC9mcy9GU0RhdGFJbnB1dFN0cmVhbTspVgEADVN0YWNrTWFwVGFibGUHAMMHARcHARgHAL0HARkBAAdwdWJsaXNoAQApKExqYXZhL2xhbmcvU3RyaW5nO0xqYXZhL2xhbmcvU3RyaW5nO1paKUoHAMgHAOwHAKYHARoHARsHARwHAOQBAARtYWluAQAWKFtMamF2YS9sYW5nL1N0cmluZzspVgcA7QcA8wcBHQcBAwEACDxjbGluaXQ+AQAKU291cmNlRmlsZQEAGE1vaXJhaUFyY2hpdmVCcmlkZ2UuamF2YQwAegB7AQAkb3JnL2FwYWNoZS9oYWRvb3AvY29uZi9Db25maWd1cmF0aW9uDAB0AHUHAR4MAR8BIAcBIQwBIgEjDAB2AHcHARsMASQBJQwAeAB5AQAQamF2YS9sYW5nL1N0cmluZwcBJgwBJwEpBwEqDAB/ASsHASwMAS0BLgwAegEvAQAbamF2YS9pby9CdWZmZXJlZElucHV0U3RyZWFtAQAXamF2YS9pby9GaWxlSW5wdXRTdHJlYW0MAHoBMAwAegExBwEYDAEyATMHARcMATIBNAEAE2phdmEvaW8vSU9FeGNlcHRpb24BABpSZW1vdGUgc2hvcnRlciB0aGFuIHNvdXJjZQwAegE1AQAWUmVtb3RlIGNvbnRlbnQgZGlmZmVycwwBMgE2AQAZUmVtb3RlIGxvbmdlciB0aGFuIHNvdXJjZQwBNwB7AQATamF2YS9sYW5nL1Rocm93YWJsZQwBOAE5DAE6ATsBAAQvLi4vDAE8AT0BABVPdXRzaWRlIG93bmVkIGFyY2hpdmUBAAxqYXZhL2lvL0ZpbGUMAT4BPwEAHkNvbXBsZXRlZCBsb2NhbCBzb3VyY2UgbWlzc2luZwwBQAFBDAFCAUEBABlvcmcvYXBhY2hlL2hhZG9vcC9mcy9QYXRoDAFDAUQMAUUBRgwBRwFGDAFIAUkHAUoMAUsBQQEAJEV4aXN0aW5nIGRlc3RpbmF0aW9uIGxlbmd0aCBjb25mbGljdAwBTAFNDACBAIIBACBTb3VyY2UgY2hhbmdlZCBkdXJpbmcgY29tcGFyaXNvbgEAF2phdmEvbGFuZy9TdHJpbmdCdWlsZGVyAQABLgwBTgFPDAFQAVEBAAkucGVuZGluZy0HAVIMAVMBVAwBTgFVDAFWAVEMAHoBVwwBHwFYBwEaDAFZAVoMAVsAewEAFVVwbG9hZCBsZW5ndGggZGlmZmVycwEAHlNvdXJjZSBjaGFuZ2VkIGR1cmluZyB0cmFuc2ZlcgcBXAEAI29yZy9hcGFjaGUvaGFkb29wL2ZzL09wdGlvbnMkUmVuYW1lAQAGUmVuYW1lAQAMSW5uZXJDbGFzc2VzDAFdAV4MAV8BXgwBYAFhAQAYUHVibGlzaGVkIGxlbmd0aCBkaWZmZXJzDAFiAWMBABNNb2lyYWlBcmNoaXZlQnJpZGdlAQAWamF2YS9pby9CdWZmZXJlZFJlYWRlcgEAGWphdmEvaW8vSW5wdXRTdHJlYW1SZWFkZXIHAWQMAWUBZgwAegFnDAB6AWgBABNqYXZhL2lvL1ByaW50V3JpdGVyAQAaamF2YS9pby9PdXRwdXRTdHJlYW1Xcml0ZXIMAWkBagwAegFrDAB6AWwMAW0BUQEAAlx0DAFuAW8BABlJbnZhbGlkIHRyYW5zcG9ydCByZXF1ZXN0DAB/AIABAAExDAFwAXEMAIkAigEAA09LCQwBTgFyDAFzATUBABNqYXZhL2xhbmcvRXhjZXB0aW9uDAF0AXUHAXYMAXcBUQEAAjogDAF4AVEMAXkBegwBewF8AQAGRVJST1IJDAF9AX8MAYABgQcBggwBgwGEAQAXTU9JUkFJX0hERlNfQVJDSElWRV9VUkkMAYUAgAEAAy8rJAEAAAwBhgGHAQABLwEAEGphdmEvbGFuZy9PYmplY3QBACZvcmcvYXBhY2hlL2hhZG9vcC9mcy9GU0RhdGFJbnB1dFN0cmVhbQEAE2phdmEvaW8vSW5wdXRTdHJlYW0BAAJbQgEAJ29yZy9hcGFjaGUvaGFkb29wL2ZzL0ZTRGF0YU91dHB1dFN0cmVhbQEAIG9yZy9hcGFjaGUvaGFkb29wL2ZzL0ZpbGVDb250ZXh0AQAmW0xvcmcvYXBhY2hlL2hhZG9vcC9mcy9PcHRpb25zJFJlbmFtZTsBABNbTGphdmEvbGFuZy9TdHJpbmc7AQAMamF2YS9uZXQvVVJJAQAGY3JlYXRlAQAiKExqYXZhL2xhbmcvU3RyaW5nOylMamF2YS9uZXQvVVJJOwEAH29yZy9hcGFjaGUvaGFkb29wL2ZzL0ZpbGVTeXN0ZW0BAANnZXQBAFcoTGphdmEvbmV0L1VSSTtMb3JnL2FwYWNoZS9oYWRvb3AvY29uZi9Db25maWd1cmF0aW9uOylMb3JnL2FwYWNoZS9oYWRvb3AvZnMvRmlsZVN5c3RlbTsBAA5nZXRGaWxlQ29udGV4dAEAWChMamF2YS9uZXQvVVJJO0xvcmcvYXBhY2hlL2hhZG9vcC9jb25mL0NvbmZpZ3VyYXRpb247KUxvcmcvYXBhY2hlL2hhZG9vcC9mcy9GaWxlQ29udGV4dDsBABBqYXZhL3V0aWwvQmFzZTY0AQAKZ2V0RGVjb2RlcgEAB0RlY29kZXIBABwoKUxqYXZhL3V0aWwvQmFzZTY0JERlY29kZXI7AQAYamF2YS91dGlsL0Jhc2U2NCREZWNvZGVyAQAWKExqYXZhL2xhbmcvU3RyaW5nOylbQgEAIWphdmEvbmlvL2NoYXJzZXQvU3RhbmRhcmRDaGFyc2V0cwEABVVURl84AQAaTGphdmEvbmlvL2NoYXJzZXQvQ2hhcnNldDsBAB8oW0JMamF2YS9uaW8vY2hhcnNldC9DaGFyc2V0OylWAQARKExqYXZhL2lvL0ZpbGU7KVYBABgoTGphdmEvaW8vSW5wdXRTdHJlYW07KVYBAARyZWFkAQAFKFtCKUkBAAcoW0JJSSlJAQAVKExqYXZhL2xhbmcvU3RyaW5nOylWAQADKClJAQAFY2xvc2UBAA1hZGRTdXBwcmVzc2VkAQAYKExqYXZhL2xhbmcvVGhyb3dhYmxlOylWAQAKc3RhcnRzV2l0aAEAFShMamF2YS9sYW5nL1N0cmluZzspWgEACGNvbnRhaW5zAQAbKExqYXZhL2xhbmcvQ2hhclNlcXVlbmNlOylaAQAGaXNGaWxlAQADKClaAQAGbGVuZ3RoAQADKClKAQAMbGFzdE1vZGlmaWVkAQAJZ2V0UGFyZW50AQAdKClMb3JnL2FwYWNoZS9oYWRvb3AvZnMvUGF0aDsBAAZta2RpcnMBAB4oTG9yZy9hcGFjaGUvaGFkb29wL2ZzL1BhdGg7KVoBAAZleGlzdHMBAA1nZXRGaWxlU3RhdHVzAQA+KExvcmcvYXBhY2hlL2hhZG9vcC9mcy9QYXRoOylMb3JnL2FwYWNoZS9oYWRvb3AvZnMvRmlsZVN0YXR1czsBAB9vcmcvYXBhY2hlL2hhZG9vcC9mcy9GaWxlU3RhdHVzAQAGZ2V0TGVuAQAEb3BlbgEARShMb3JnL2FwYWNoZS9oYWRvb3AvZnMvUGF0aDspTG9yZy9hcGFjaGUvaGFkb29wL2ZzL0ZTRGF0YUlucHV0U3RyZWFtOwEABmFwcGVuZAEALShMamF2YS9sYW5nL1N0cmluZzspTGphdmEvbGFuZy9TdHJpbmdCdWlsZGVyOwEAB2dldE5hbWUBABQoKUxqYXZhL2xhbmcvU3RyaW5nOwEADmphdmEvdXRpbC9VVUlEAQAKcmFuZG9tVVVJRAEAEigpTGphdmEvdXRpbC9VVUlEOwEALShMamF2YS9sYW5nL09iamVjdDspTGphdmEvbGFuZy9TdHJpbmdCdWlsZGVyOwEACHRvU3RyaW5nAQAwKExvcmcvYXBhY2hlL2hhZG9vcC9mcy9QYXRoO0xqYXZhL2xhbmcvU3RyaW5nOylWAQBHKExvcmcvYXBhY2hlL2hhZG9vcC9mcy9QYXRoO1opTG9yZy9hcGFjaGUvaGFkb29wL2ZzL0ZTRGF0YU91dHB1dFN0cmVhbTsBAAV3cml0ZQEAByhbQklJKVYBAAVoc3luYwEAHG9yZy9hcGFjaGUvaGFkb29wL2ZzL09wdGlvbnMBAAlPVkVSV1JJVEUBACVMb3JnL2FwYWNoZS9oYWRvb3AvZnMvT3B0aW9ucyRSZW5hbWU7AQAETk9ORQEABnJlbmFtZQEAXyhMb3JnL2FwYWNoZS9oYWRvb3AvZnMvUGF0aDtMb3JnL2FwYWNoZS9oYWRvb3AvZnMvUGF0aDtbTG9yZy9hcGFjaGUvaGFkb29wL2ZzL09wdGlvbnMkUmVuYW1lOylWAQAGZGVsZXRlAQAfKExvcmcvYXBhY2hlL2hhZG9vcC9mcy9QYXRoO1opWgEAEGphdmEvbGFuZy9TeXN0ZW0BAAJpbgEAFUxqYXZhL2lvL0lucHV0U3RyZWFtOwEAMihMamF2YS9pby9JbnB1dFN0cmVhbTtMamF2YS9uaW8vY2hhcnNldC9DaGFyc2V0OylWAQATKExqYXZhL2lvL1JlYWRlcjspVgEAA291dAEAFUxqYXZhL2lvL1ByaW50U3RyZWFtOwEAMyhMamF2YS9pby9PdXRwdXRTdHJlYW07TGphdmEvbmlvL2NoYXJzZXQvQ2hhcnNldDspVgEAFChMamF2YS9pby9Xcml0ZXI7WilWAQAIcmVhZExpbmUBAAVzcGxpdAEAKChMamF2YS9sYW5nL1N0cmluZztJKVtMamF2YS9sYW5nL1N0cmluZzsBAAZlcXVhbHMBABUoTGphdmEvbGFuZy9PYmplY3Q7KVoBABwoSilMamF2YS9sYW5nL1N0cmlu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'

def native_command(name):
    override=os.environ.get('MOIRAI_'+('HDFS_CLI' if name=='hdfs' else 'HADOOP_CLI'))
    candidates=[override,shutil.which(name)]
    for variable in ['HADOOP_HOME','HADOOP_HDFS_HOME']:
        if os.environ.get(variable):candidates.append(str(Path(os.environ[variable])/'bin'/name))
    for value in candidates:
        if value:
            resolved=shutil.which(value)
            if resolved:return resolved
    raise RuntimeError('Native '+name+' client is unavailable; configure its executable explicitly or use ordinary filesystem storage')


class NativeCheckpointWriter:
    def __init__(self,work,archive_uri=None):
        self.archive_uri=(archive_uri or os.environ['MOIRAI_HDFS_ARCHIVE_URI']).rstrip('/')
        self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        jar=self.work/'archive-transport.jar'
        with zipfile.ZipFile(jar,'w') as archive:
            archive.writestr('MoiraiArchiveBridge.class',base64.b64decode(_CLASS_DATA))
        hadoop=native_command('hadoop')
        self.log=(self.work/'transport.log').open('a')
        # The installed Hadoop wrapper supplies required site/JVM authentication settings.
        self.process=subprocess.Popen([hadoop,'jar',str(jar),'MoiraiArchiveBridge'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,env=dict(os.environ,MOIRAI_HDFS_ARCHIVE_URI=self.archive_uri))
        self.ready=selectors.DefaultSelector();self.ready.register(self.process.stdout,selectors.EVENT_READ)
        self.received=b''
        atexit.register(self.close)
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.terminate()
        if not self.process.stdin.closed:self.process.stdin.close()
        self.process.stdout.close();self.ready.close();self.log.close()
    def publish(self,source,uri,overwrite=False,verify=True):
        if not uri.startswith(self.archive_uri+'/') or '..' in Path(uri).parts:raise ValueError('Outside owned experiment archive')
        fields=[base64.b64encode(str(x).encode()).decode() for x in [source,uri]]+[str(int(overwrite)),str(int(verify))]
        self.process.stdin.write(('\t'.join(fields)+'\n').encode());self.process.stdin.flush()
        deadline=time.monotonic()+1800
        while True:
            while b'\n' not in self.received:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not self.ready.select(timeout=remaining):raise TimeoutError('HDFS checkpoint publication did not finish')
                chunk=os.read(self.process.stdout.fileno(),65536)
                if not chunk:
                    self.log.flush()
                    detail=(self.work/'transport.log').read_text(errors='replace')[-4000:]
                    import re
                    detail='\n'.join(line for line in detail.splitlines() if not re.search(r'token|password|secret|credential|authorization|cookie|keytab',line,re.I))
                    raise OSError('HDFS transport exited before receipt; '+detail[-2200:])
                self.received+=chunk
            raw,self.received=self.received.split(b'\n',1);response=raw.decode(errors='replace')
            if response.startswith('OK\t'):return int(response.split('\t')[1])
            if response.startswith('ERROR\t'):raise OSError(base64.b64decode(response.split('\t')[1]).decode())
            # Site wrappers may emit ordinary startup messages to stdout.
            with (self.work/'wrapper_stdout.log').open('ab') as diagnostic:diagnostic.write(raw+b'\n')


_WRITER=None

def _writer():
    global _WRITER
    if _WRITER is None:
        work=Path(os.environ['RESEARCH_WORKER_DIR'])/'hdfs_transport' if os.environ.get('RESEARCH_WORKER_DIR') else Path(tempfile.mkdtemp(prefix='moirai-hdfs-client-'))
        _WRITER=NativeCheckpointWriter(work)
    return _WRITER

def destination(path,worker,category):
    path=Path(path);worker=Path(worker)
    relative=path.relative_to(worker)
    if relative.parts[0]!='result' or '..' in relative.parts:raise ValueError('Only this worker result can publish checkpoints')
    parts=Path(category).parts
    if not parts or Path(category).is_absolute() or any(x in ['.','..'] for x in parts):raise ValueError('Invalid checkpoint category')
    return '/'.join([category,*relative.parts])

def publish_cli(source,uri):
    import filecmp
    hdfs=native_command('hdfs')
    work=Path(os.environ['RESEARCH_WORKER_DIR'])/'hdfs_transport';work.mkdir(exist_ok=True)
    def run(arguments):
        with (work/'native_cli.log').open('a') as log:
            result=subprocess.run([hdfs,'dfs',*arguments],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if result.returncode:raise OSError('Native HDFS file command failed; original state retained; see worker native_cli.log')
    run(['-mkdir','-p',uri.rsplit('/',1)[0]])
    # The destination is a new immutable generation. Hadoop -put publishes its
    # completed .COPYING upload before we change the logical checkpoint link.
    run(['-put',str(source),uri])
    with tempfile.TemporaryDirectory(prefix='moirai-checkpoint-verify-') as scratch:
        copy=Path(scratch)/'remote.pt';run(['-get',uri,str(copy)])
        if not filecmp.cmp(source,copy,shallow=False):raise OSError('Remote checkpoint differs; original state retained')
    return Path(source).stat().st_size

def write_if_enabled(path,buffer):
    import uuid
    category=os.environ.get('MOIRAI_HDFS_CHECKPOINT_CATEGORY')
    if not category or Path(path).suffix.lower() not in {'.pt','.pth','.ckpt','.safetensors'}:return False
    worker=Path(os.environ['RESEARCH_WORKER_DIR']);path=Path(path)
    logical=destination(path,worker,category)
    relative=logical+'.generations/'+uuid.uuid4().hex+'.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='moirai-checkpoint-') as scratch:
        source=Path(scratch)/path.name
        with source.open('wb') as stream:
            stream.write(buffer.getbuffer());stream.flush();os.fsync(stream.fileno())
        size=publish_cli(source,os.environ['MOIRAI_HDFS_ARCHIVE_URI'].rstrip('/')+'/'+relative)
    target=Path(os.environ['MOIRAI_HDFS_ARCHIVE_MOUNT'])/relative
    temporary=path.with_name('.'+path.name+'.hdfs-link-'+str(os.getpid()))
    temporary.unlink(missing_ok=True);temporary.symlink_to(target);os.replace(temporary,path)
    with (worker/'checkpoint_storage.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(epoch=time.time(),logical_path=str(path),storage='hdfs',archive_relative=relative,bytes=size,publication_complete=True,backend='native-cli-immutable-generations-v1'))+'\n')
    return True


def remote_uri(path):
    uri=os.environ.get('MOIRAI_HDFS_ARCHIVE_URI');mount=os.environ.get('MOIRAI_HDFS_ARCHIVE_MOUNT')
    if not uri or not mount:return None
    if not isinstance(path,(str,os.PathLike)):return None
    if str(path).startswith(uri.rstrip('/')+'/'):
        if '..' in Path(str(path)).parts:raise ValueError('Invalid owned archive URI')
        return str(path)
    p=Path(path)
    target=Path(os.readlink(p)) if p.is_symlink() else p
    try:relative=target.relative_to(mount)
    except ValueError:return None
    if not relative.parts or '..' in relative.parts:raise ValueError('Invalid owned archive path')
    return uri.rstrip('/')+'/'+str(relative)

from contextlib import contextmanager
@contextmanager
def local_checkpoint(path):
    uri=remote_uri(path)
    if uri is None:yield path;return
    with tempfile.TemporaryDirectory(prefix='moirai-checkpoint-read-') as scratch:
        local=Path(scratch)/'state.pt'
        hadoop=native_command('hdfs')
        with (Path(scratch)/'transfer.log').open('w') as log:
            r=subprocess.run([hadoop,'dfs','-get',uri,str(local)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        if r.returncode or not local.is_file():raise OSError('Native HDFS checkpoint download failed; original archive retained')
        yield local

def checkpoint_exists(path):
    return bool(remote_uri(path)) or Path(path).is_file()


def worker_category(experiment,runtime_config):
    import re
    model_text=str(runtime_config.get('weights') or runtime_config.get('model_id') or '')
    match=re.search(r'moirai-[0-9.]+-R-(?:small|base|large)',model_text,re.I)
    model=match.group(0) if match else 'model_unrecorded'
    dataset=runtime_config.get('protocol',{}).get('dataset') or runtime_config.get('dataset')
    if dataset is None:
        match=re.search(r'ETT[hm][12]',experiment['node_id']);dataset=match.group(0) if match else 'dataset_unrecorded'
    node=experiment['node_id'];attempt=experiment['attempt']
    for part in [model,dataset,node]:
        if not isinstance(part,str) or re.fullmatch(r'[A-Za-z0-9_.-]+',part) is None:raise ValueError('Invalid archive identity')
    return '/'.join(['point_opd_confirmation_20261024_130931',model,dataset,node.split('-')[0],node,'attempt-'+str(int(attempt))])


def audit(config_path,output,device='cuda',seed=1):
    # Every tensor operation in this qualification executes in its HOPE worker.
    import torch
    import numpy as np
    from .io_utils import buffered_torch_save,equal_state,load_checkpoint
    from .runtime import atomic_json
    config=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
    os.environ['MOIRAI_HDFS_ARCHIVE_URI']=config['archive_uri']
    os.environ['MOIRAI_HDFS_ARCHIVE_MOUNT']=config['archive_mount']
    source=config['source_checkpoint'];source_uri=remote_uri(source)
    if source_uri is None:
        raise ValueError('Selected source must resolve inside the configured owned archive base; no mounted checkpoint fallback')
    with local_checkpoint(source) as materialized:
        source_bytes=Path(materialized).stat().st_size
        state=load_checkpoint(materialized,map_location='cpu',weights_only=False)
    required={'policy_state','optimizer_state','rng','cursor','config'}
    if not isinstance(state,dict) or not required.issubset(state):
        raise ValueError('Storage qualification requires a real complete optimizer/RNG/cursor state')
    if not {'python','numpy','torch','cuda'}.issubset(state['rng']):
        raise ValueError('Selected source is missing required continuation RNG streams')
    category=config.get('qualification_category','point_opd_confirmation_20261024_130931/qualification/worker-write/'+config['qualification_id'])
    if not category.startswith('point_opd_confirmation_20261024_130931/qualification/'):
        raise ValueError('New qualification outputs must remain inside this campaign category')
    os.environ['MOIRAI_HDFS_CHECKPOINT_CATEGORY']=category
    clone=out/'full_state_roundtrip.pt';buffered_torch_save(state,clone)
    remote=remote_uri(clone)
    if remote is None or not clone.is_symlink():raise RuntimeError('New checkpoint did not publish through the native archive backend')
    equal_state(state,load_checkpoint(remote,map_location='cpu',weights_only=False))
    direct_mount=os.readlink(clone)
    if remote_uri(direct_mount)!=remote:raise RuntimeError('Direct owned-mount path does not resolve to the published native URI')
    equal_state(state,load_checkpoint(direct_mount,map_location='cpu',weights_only=False))
    original=state.get('policy_state',state.get('state_dict',state))
    tensors=[v for v in original.values() if torch.is_tensor(v)]
    assert tensors,'Real checkpoint contains no model tensors'
    tensor=tensors[0].to(device);assert torch.equal(tensor.cpu(),tensors[0])
    atomic_json(out/'result.json',dict(status='PASS',source_checkpoint=str(source),source_remote_uri=source_uri,
        remote_checkpoint=str(remote),qualification_category=category,
        original_remote_and_reloaded_state_equal=True,native_read_without_mount=True,new_checkpoint_is_hdfs_link=clone.is_symlink(),storage_backend='native-cli-immutable-generations-v1',
        source_read_backend='native CLI materialization, no mounted checkpoint read',
        independent_native_URI_reload=True,direct_owned_mount_path_native_reload=True,
        complete_state_fields=sorted(required),source_cursor=state['cursor'],
        optimizer_rng_cursor_preserved=True,optimizer_state_entries=len(state['optimizer_state'].get('state',{})),
        roundtrip_bytes=source_bytes,optimizer_trajectories=0,optimizer_steps=0,
        actual_gpu=torch.cuda.get_device_name(),test_used=False,model_parameters_changed=False,
        selected_old_source_written=False,CUDA_model_tensor_roundtrip_equal=True))
