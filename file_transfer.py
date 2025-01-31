import paramiko
import os
import posixpath  # POSIX 경로 처리를 위해 추가
from datetime import datetime
import stat

class SFTPManager:
    def __init__(self, hostname, pem_path, port=30483):
        self.hostname = hostname
        self.pem_path = pem_path
        self.port = port
        self.ssh = None
        self.sftp = None
    
    def connect(self):
        try:
            # PEM 키 로드
            private_key = paramiko.RSAKey.from_private_key_file(self.pem_path)
            
            # SSH 클라이언트 생성
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # PEM 키를 사용하여 서버 연결
            self.ssh.connect(
                hostname=self.hostname,
                username='root',
                pkey=private_key,
                port=self.port
            )
            
            # SFTP 클라이언트 생성
            self.sftp = self.ssh.open_sftp()
            print("서버 연결 성공")
            
        except Exception as e:
            print(f"서버 연결 실패: {str(e)}")
            self.close()
    
    def upload_file(self, local_path, remote_path):
        try:
            self.sftp.put(local_path, remote_path)
            print(f"파일 업로드 성공: {local_path} -> {remote_path}")
        except Exception as e:
            print(f"파일 업로드 실패: {str(e)}")
    
    def download_file(self, remote_path, local_path):
        try:
            self.sftp.get(remote_path, local_path)
            print(f"파일 다운로드 성공: {remote_path} -> {local_path}")
        except Exception as e:
            print(f"파일 다운로드 실패: {str(e)}")
    
    def upload_directory(self, local_dir, remote_dir):
        try:
            # 원격 디렉토리가 없으면 생성
            try:
                self.sftp.stat(remote_dir)
            except FileNotFoundError:
                self.sftp.mkdir(remote_dir)
            
            # 로컬 디렉토리의 모든 파일 업로드
            for root, dirs, files in os.walk(local_dir):
                for dir_name in dirs:
                    local_path = os.path.join(root, dir_name)
                    remote_path = posixpath.join(remote_dir, os.path.relpath(local_path, local_dir))
                    try:
                        self.sftp.mkdir(remote_path)
                    except:
                        pass
                
                for file_name in files:
                    local_path = os.path.join(root, file_name)
                    remote_path = posixpath.join(remote_dir, os.path.relpath(local_path, local_dir))
                    self.upload_file(local_path, remote_path)
            
            print(f"디렉토리 업로드 성공: {local_dir} -> {remote_dir}")
        except Exception as e:
            print(f"디렉토리 업로드 실패: {str(e)}")
    
    def download_directory(self, remote_dir, local_dir):
        try:
            # 로컬 디렉토리가 없으면 생성
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            # 원격 디렉토리의 모든 파일과 폴더 가져오기
            def download_recursive(remote_path, local_path):
                try:
                    # 원격 경로의 파일/폴더 목록 가져오기
                    items = self.sftp.listdir_attr(remote_path)
                    
                    for item in items:
                        remote_item_path = posixpath.join(remote_path, item.filename)
                        local_item_path = os.path.join(local_path, item.filename)
                        
                        if stat.S_ISDIR(item.st_mode):  # 디렉토리인 경우
                            # 로컬에 해당 디렉토리 생성
                            if not os.path.exists(local_item_path):
                                os.makedirs(local_item_path)
                            # 재귀적으로 하위 디렉토리 다운로드
                            download_recursive(remote_item_path, local_item_path)
                        else:  # 파일인 경우
                            self.download_file(remote_item_path, local_item_path)
                            
                except Exception as e:
                    print(f"다운로드 중 오류 발생: {str(e)}")
            
            # 재귀적 다운로드 시작
            if self.sftp:
                download_recursive(remote_dir, local_dir)
                print(f"디렉토리 다운로드 성공: {remote_dir} -> {local_dir}")
            else:
                print("SFTP 연결이 설정되지 않았습니다.")
            
        except Exception as e:
            print(f"디렉토리 다운로드 실패: {str(e)}")
    
    def close(self):
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        print("서버 연결 종료")

def main():
    # 서버 접속 정보
    hostname = "10.196.197.26"
    pem_path = "C:\\Users\\user\\.ssh\\nlp_stage.pem"  # PEM 파일의 경로
    
    # 전송할 파일/폴더 경로
    local_review_dir = "C:\\Users\\user\\OneDrive\\경희대\\3학년2학기(휴학)\\UPSTAGE AI LAB 5기\\제출파일\\nlp\\code"
    remote_review_dir = "/home/code/files"  # root 사용자의 홈 디렉토리
    
    # 현재 날짜로 백업 폴더 생성
    current_date = datetime.now().strftime("%Y%m%d")
    remote_backup_dir = f"/home/fine_data_{current_date}"
    
    # SFTP 매니저 생성 및 연결
    sftp_manager = SFTPManager(hostname, pem_path)
    sftp_manager.connect()
    
    try:
        # 리뷰 데이터 폴더 전체를 서버로 다운로드
        sftp_manager.download_directory(remote_review_dir, local_review_dir)
        
    finally:
        sftp_manager.close()

if __name__ == "__main__":
    main()