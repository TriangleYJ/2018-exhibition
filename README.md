# A Development of Non-API based Platform Game AI using reinforcement learning

2018 과학전람회 작품 연구 주제로 Game Api를 필요로 하지 않는 플랫폼 게임 AI에 대한 연구를 진행하고 있습니다. Platform 게임으로는 Geometry dash를 선정하였습니다!

## 작품에 대한 설명
![Alt text](/image/diagram1.PNG)

다른 게임 AI들은 거의 대부분 API를 사용해서 작동된다. 따라서 게임으로부터 진행되고 있는 화면으로부터 정보를 가져오거나 게임에 정보를 보내는 것이 모두 AI 내 소스 몇줄로 가능해 진다. 하지만 우리가 진행하고자 하는 게임 “Geometry Dash”는 API가 제공되지 않는다. 따라서 위의 방식으로 AI를 제작하는 것이 불가능하다. 따라서 데이터를 입력받거나 데이터를 제공하는 작업을 일일이 처리해야 한다. 우리는 이러한 작업을 직접 처리하기 위해서 컴퓨터 내 화면을 캡쳐하는 라이브러리 Mss 라이브러리와 게임에 터치 정보를 제공하기 위해서 Win32GUI를 사용할 계획이다.