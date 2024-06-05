# jigsaw_solver_dacon

![Example Image]([https://dacon.s3.ap-northeast-2.amazonaws.com/attach/talkboard/236207/495721/1708044003398627.png](https://github.com/co1dtype/jigsaw_solver_dacon/blob/main/img/1708044003398627%20(1).png?raw=true))

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




