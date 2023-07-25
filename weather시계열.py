import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 7/1~7/23의 기상 데이터 (풍속, 습도, 기온, 파향)
wind_speed = [4.5, 6.5, 1.0, 5.1, 9.5, 6.9, 7.3, 4.6, 1.7, 4.8, 7.2, 11.2, 
              0.7, 8.0, 11.7, 4.7, 6.5, 3.1, 3.8, 2.4, 2.8, 2.7, 7.5]
humidity = [91, 92, 90, 87, 89, 78, 85, 85, 90, 96, 88, 86, 93, 91, 87, 96, 
            88, 93, 95, 87, 82, 84, 93]
temperature = [22.8, 23.2, 21.9, 23.8, 24.1, 23.8, 25.4, 21.6, 
               23.8, 24.6, 23.6, 24.3, 24.8, 23.6, 25.2, 23.6, 22.6, 24.9, 
               23.0, 24.6, 25.6, 25.8, 26.0]
wave_direction = [189, 44, 38, 236, 120, 194, 211, 217, 130, 193, 187, 199, 
                  211, 186, 176, 189, 51, 38, 48, 166, 68, 166, 116]

# 7/1~7/23의 수온 데이터
sea_temperature = [21.5, 21.5, 22.8, 23.2, 23.0, 23.0, 23.2, 23.1, 23.4, 23.4, 23.4, 
                   23.2, 24.0, 23.6, 22.6, 20.9, 22.3, 23.7, 23.7, 23.9, 25.0, 25.1, 25.1]

# 데이터를 1차원 배열로 변환
sea_temperature = np.array(sea_temperature)

# ARIMA 모델 학습
model = ARIMA(sea_temperature, order=(7, 1, 0))  # (p, d, q) 설정으로 ARIMA 모델을 생성
model_fit = model.fit()

# 다음날의 기상 요소와 파향 데이터
next_day_wind_speed = 8.3
next_day_humidity = 90
next_day_temperature = 26.7
next_day_wave_direction = 214

# 다음날의 수온 예측
next_day_sea_temperature = model_fit.forecast(steps=1)[0]
print(f"다음날의 수온 예측값: {next_day_sea_temperature:.1f}도")
