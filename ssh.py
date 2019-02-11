import paramiko
import fnmatch
import os

'Connect'
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname="bjth.itu.dk",username="fkp",password="")
sftp = ssh_client.open_sftp()

'Execute command'
cmd = "cd /30T/Music && ls"
stdin,stdout,stderr=ssh_client.exec_command(cmd)
'print(stdout.readlines())'

def sftp_walk(remotepath):
    path = remotepath
    folders = []
    files = []
    for f in sftp.listdir_attr(remotepath):
        if os.stat.S_ISDIR(file.st_mode):
            folders.append(f.filename)
        else:
            if f.filename.endswith('.mp3'):
                files.append(f.filename)
    print(files)
    for folder in folders:
        new_path = os.path.join(remotepath, folder)
        sftp_walk(new_path)
    

sftp_walk("/30T/Music/MSD")