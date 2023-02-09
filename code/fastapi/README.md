# FastAPI
FastAPI을 이용한 모델 온라인 서빙 

## Getting Started
0. Python requirements  
   `Python`: 3.6.2 이상이 필요합니다  
   `가상환경`: poetry(>=1.0.0)를 사용(권장), 또는 virtualenv, pyenv-virtualenv 등의 방법을 사용할 수 있습니다.
1. Installation
   1. 가상 환경을 설정합니다
      - Poetry
         1. Poetry 설치하기
            - Mac OSX / Linux
              ```shell
              > curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
              ```
         2. Poetry shell  
            Poetry로 가상환경을 만듭니다. 
            ```shell
            # 프로젝트 루트에서
            > cd /opt/ml/input/code/fastapi
            # (optional)
            > poetry config virtualenvs.create true # 로컬에 .venv 폴더를 생성해서, IDE에서 interpreter 설정이 편해집니다
            > poetry shell
            ```
   2. 프로젝트의 의존성을 설치합니다
      - Using Poetry
        ```shell
        > poetry install
        ```
      - 나머지
        ```shell
        > pip install -r requirements.txt 
        
        > apt-get update
        > apt-get install -y make
        > pip install black

        # module 'click' has no attribute 'command'에러가 뜨면
        > pip install -U click==8.0.0

        > pip install pdf2img
        > apt-get install poppler-utils
        > pip install pycocotools
        > pip install stqdm

        > pip install -U openmim
        > mim install mmcv-full
        > pip install mmdet
        
        # opencv 관련에러가 뜨면 headless와 opencv 버전을 맞춰서 다시 설치합니다.
        > pip install opencv-python-headless==4.7.0.68
        > pip install mmcls
        ``` 
   3. 애플리케이션을 실행합니다
      ```shell
      > cd /opt/ml/input/code/fastapi
      > python -m app
       INFO:     Started server process [11467]
       INFO:     Waiting for application startup.
       INFO:     Application startup complete.
       INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)

      ```  
   4. Frontend(Streamlit)와 Server를 같이 실행합니다
      ```shell
      make -j 2 run_app
      # or
      
      python3 -m app
      # in other shell
      python3 -m streamlit run app/frontend.py
      ```
