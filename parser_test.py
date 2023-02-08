import argparse
from pathlib import Path


# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='사용법 테스트입니다.')

# 입력받을 인자값 등록
parser.add_argument('config', help='Path to config file')

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

# 입력받은 인자값 출력
print(Path(args.config).stem)