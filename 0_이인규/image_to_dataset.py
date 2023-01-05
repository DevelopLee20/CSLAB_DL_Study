import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as scalers

def image_to_input_data(dir_path:str, grayscale:bool=True, processing:int=0, validation:float=0.0):
    '''이미지 파일을 X_train, y_train, X_test, y_test 데이터로 변환해주는 함수
    
    폴더 구조 예시
    - train(dir_path)
        - label1
            - 1.jpg
            - 2.jpg
        - label2
            - 1.jpg
            - 2.jpg
            - 3.jpg
    
    Params
    
    @param dir_path:str -> 파일 위치
    @param grayscale:bool default=True -> 이미지 그레이스케일 적용 여부
    @param processing:int -> 스케일러 종류 default=0.0 (없음), 1:MinMax, 2:Standard
    @param validation:float (0.0 ~ 1.0), default=0.0 -> 테스트 데이터 분할 비율
    
    Returns
    
    if validation > 0.0
    @return X_train:list -> 훈련 타겟 데이터
    @return X_test:list -> 테스트 타겟 데이터
    @return y_train:list -> 훈련 라벨 데이터
    @return y_test:list -> 테스트 타겟 데이터
    
    if validation == 0
    @return X_train:list -> 훈련 타겟 데이터
    @return y_train:list -> 테스트 타겟 데이터
    '''
    
    # Params value error check part
    if type(dir_path) is not str:
        raise TypeError(
            f"Param dir_path -> {dir_path} is not string type."
            "@param dir_path:str -> 파일 위치"
        )
    
    if type(processing) is not int:
        raise TypeError(
            f"Param processing -> {processing} is not int type."
            "@param processing:int -> 스케일러 종류 default=0.0 (없음), 1:MinMax, 2:Standard"
        )
        
    elif processing > 2 or processing < 0:
        raise ValueError(
            f"Param processing -> {processing}."
            "It must be a value between 0 and 2"
        )
    
    if type(validation) is not float:
        raise TypeError(
            f"Param validation -> {validation} is not float type."
            "@param validation:float (0.0 ~ 1.0), default=0.0 -> 테스트 데이터 분할 비율"
        )
        
    elif validation < 0 or validation > 1.0:
        raise ValueError(
            f"Param validation -> {validation}."
            "it must be a value between 0.0 and 1.0."
            "@param validation:float (0.0 ~ 1.0), default=0.0 -> 테스트 데이터 분할 비율"
        )
        
    if type(grayscale) is not bool:
        raise TypeError(
            f"Param grayscale -> {grayscale} is not bool type."
            "@param grayscale:bool default=True -> 이미지 그레이스케일 적용 여부"
        )
        
    # Image to array part
    X_array = np.array([])
    y_array = []
    classes_name_list = os.listdir(dir_path)
    image_count = 0
    
    classes = dict((c,i) for i, c in enumerate(classes_name_list))
    
    # grayscale part
    if grayscale:
        for classes_name in classes_name_list:
            picture_name_list_path = os.path.join(dir_path, classes_name)
            picture_name_list = os.listdir(picture_name_list_path)
            
            for picture_name in picture_name_list:
                image_path = os.path.join(picture_name_list_path, picture_name)
                image = cv2.imread(image_path, 0)
                X_array = np.append(X_array, image)
                y_array.append(classes[classes_name])
                image_count = image_count + 1
                
    else:
        # still making 아직 만드는 중
        return 'grayscale True Please'
    
    X_array = X_array.reshape(image_count, -1)
    
    # scaler part
    if processing == 1:
        ScaleType = scalers.MinMaxScaler()
        X_array = ScaleType.fit_transform(X_array)
        
    elif processing == 2:
        ScaleType = scalers.StandardScaler()
        X_array = ScaleType.fit_transform(X_array)
    
    # Validation part
    if validation:
        X_train, X_test, y_train, y_test = \
            train_test_split(X_array, y_array, stratify=y_array, test_size=validation)     
        # Return part
        return X_train, X_test, y_train, y_test
    
    else:        
        X_train = X_array
        y_train = y_array
        # Return part
        return X_train, y_train


'''Test part
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = image_to_input_data(
        dir_path='./train',
        processing=1,
        validation=0.5,
        grayscale=True)

    print(X_train.shape, len(X_test), y_train.shape, len(y_test))
'''