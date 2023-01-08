''' 노션 주소
https://www.notion.so/leeingyu/4d1c748776804815a9c4747e8e30b6a7
'''

import image_to_dataset as itd
import torch

def model_train(X_train, y_train, X_test, y_test):
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1,1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1,1)
    
    # 로지스틱 회귀 모델 클래스
    class LogisticClassifer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(400, 128)
            self.linear2 = torch.nn.Linear(128, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            output = self.linear1(x)
            output = self.linear2(output)
            output = self.sigmoid(output)
            return output

    model = LogisticClassifer()

    # 파라미터 설정
    # SGD 확률적 경사 하강법
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    nb_epochs = 300 # 반복 횟수

    for epoch in range(nb_epochs + 1):
        hypothesis = model(X_train)
        loss = torch.nn.functional.binary_cross_entropy(hypothesis, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 0.5는 임계값
    pred = model(X_train) >= torch.FloatTensor([0.5])
    accuracy = (pred == y_train).sum() / len(y_train)
    print(f'Train Data : Epoch {epoch}/{nb_epochs} Loss: {loss:.4f} Accuracy {accuracy*100:.2f}%')

    pred = model(X_test) >= torch.FloatTensor([0.5])
    accuracy = (pred == y_test).sum() / len(y_test)
    print(f'Test Data : Accuracy {accuracy*100:.2f}%')

# 데이터 생성하기
print("None")
# itd는 직접 만든 이미지 파일을 학습용 데이터로 만들어주는 모듈
X_train, X_test, y_train, y_test = itd.image_to_input_data(
    dir_path='./data/logistic_data',    # 파일 위치
    grayscale=True,                     # 그레이스케일 여부
    processing=0,                       # 전처리 : 0는 없음
    validation=0.1                      # 훈련데이터 비율
)

model_train(X_train, y_train, X_test, y_test)
print("===========================================================")

print("Min-Max")
X_train, X_test, y_train, y_test = itd.image_to_input_data(
    dir_path='./data/logistic_data',    # 파일 위치
    grayscale=True,                     # 그레이스케일 여부
    processing=1,                       # 전처리 : 1는 최대 최소 전처리
    validation=0.1                      # 훈련데이터 비율
)

model_train(X_train, y_train, X_test, y_test)
print("===========================================================")
print("Standard")
X_train, X_test, y_train, y_test = itd.image_to_input_data(
    dir_path='./data/logistic_data',    # 파일 위치
    grayscale=True,                     # 그레이스케일 여부
    processing=2,                       # 전처리 : 표준화 전처리
    validation=0.1                      # 훈련데이터 비율
)

model_train(X_train, y_train, X_test, y_test)
print("===========================================================")