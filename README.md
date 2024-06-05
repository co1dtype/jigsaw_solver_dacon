# jigsaw_solver_dacon
[대학 대항전 : 퍼즐 이미지 AI 경진대회 1등 솔루션](https://dacon.io/competitions/official/236207/codeshare/9670?page=1&dtype=recent)  

![image](https://github.com/co1dtype/jigsaw_solver_dacon/assets/76248669/48b9d203-cafa-4a1a-8f60-a87bea8fcaa4)
  
  
baseline에서 코드를 수정 및 추가하였습니다.
> Model을 SegFormer를 사용했습니다.  
> Adam에서 AdamW, learning rate, weight decay를 튜닝하였습니다. 
> Augmentation을 적용했습니다. (PPT 및 코드 참고)  
> Ensemble 및 중복 제거 코드를 추가하였습니다.

**.py파일은 지금 실행이 안됩니다. 시간 나면 재업로드 하겠습니다.**


학습 환경
> CPU: Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz  
> RAM: 1000GB  
> GPU: RTX 3090 24GB * 8 (학습에는 1장만 사용)  
> OS: Linux 5.4.0-167-generic  
> Docker: ufoym/deepo:latest

| 이름            | 버전                            |
|-----------------|---------------------------------|
| Python          | 3.8.10                          |
| Numpy           | 1.22.3                          |
| Pandas          | 1.4.1                           |
| Matplotlib      | 3.5.1                           |
| PyTorch         | 1.12.0.dev20220327+cu113        |
| Torchvision     | 0.13.0.dev20220327+cu113        |
| Transformers    | 4.36.2                          |
| Albumentations  | 1.3.1                           |
| OpenCV          | 4.9.0                           |
| tqdm            | 4.63.1                          |
| PIL             | 9.0.1                           |


# Dataset
![image](https://github.com/co1dtype/jigsaw_solver_dacon/assets/76248669/f7aa2201-a05d-4aed-93cc-b8d439d17b6d)
 4x4의 격자 형태의 16개의 조각으로 구성된 순서가 뒤바뀐 퍼즐 이미지로 이루어져 있다.

 

 



